from __future__ import division
import os
import random
import argparse
import time
import math
import numpy as np

import paddle
import paddle.optimizer as optim
paddle.disable_static()
from data import *
import tools
# import paddlex
# from paddlex.det import transforms
from utils.augmentations import SSDAugmentation
from utils.vocapi_evaluator import VOCAPIEvaluator
import logging
# from visualdl import LogWriter
logging.basicConfig(filename='./TrainLog.txt',level=logging.INFO)
def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-v', '--version', default='yolo',
                        help='yolo')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')  
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    # parser.add_argument('-r', '--resume', default='./checkpoints/yolo-model-best.pdparams', type=str,
    #                     help='keep training')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='use gpu.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')

    return parser.parse_args()


def train():
    args = parse_args()

    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # use hi-res backbone
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True
    else:
        hr = False
    
    if args.gpu:
        print('use gpu')
        device = paddle.device.set_device("gpu")
    else:
        device = paddle.device.set_device("cpu")

    # multi-scale
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = [640, 640]
        val_size = [416, 416]
    else:
        train_size = [416, 416]
        val_size = [416, 416]

    cfg = train_cfg
    # dataset and evaluator
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    print('Loading the dataset...')

    if args.dataset == 'voc':
        data_dir = VOC_ROOT
        num_classes = 20
        print(data_dir)

        dataset = VOCDetection(root=data_dir, 
                                img_size=train_size[0],
                                transform=SSDAugmentation(train_size)
                                )

        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=VOC_CLASSES
                                    )
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)
    
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # dataloader
    dataloader = paddle.io.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,drop_last=True
                    )

    # build model
    if args.version == 'yolo':
        from models.yolo import myYOLO
        yolo_net = myYOLO(device, input_size=train_size, num_classes=num_classes, trainable=True)
        print('Let us train yolo on the %s dataset ......' % (args.dataset))

    else:
        print('We only support YOLO !!!')
        exit()


    model = yolo_net
    model.train()

    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_dict(paddle.load(args.resume))

    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    # optimizer = paddle.optimizer.SGD(model.parameters(),args.lr,
    #                         weight_decay=args.weight_decay
    #                         )
    # optimizer=paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=0.000001, momentum=args.momentum,weight_decay=args.weight_decay,use_nesterov=True)
    # optimizer = paddle.optimizer.SGD(parameters=model.parameters(),weight_decay=args.weight_decay)
    max_epoch = cfg['max_epoch']
    epoch_size = len(dataset) // args.batch_size

    # start training loop
    t0 = time.time()

    for epoch in range(args.start_epoch, max_epoch):
        optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=args.lr,
                                              momentum=args.momentum, weight_decay=args.weight_decay, use_nesterov=True)

        # use cos lr
        if args.cos and epoch > 20 and epoch <= max_epoch - 20:
            # use cos lr
            tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1./ (max_epoch-20)))
            set_lr(optimizer, tmp_lr)

        elif args.cos and epoch > max_epoch - 20:
            tmp_lr = 0.00001
            set_lr(optimizer, tmp_lr)
        
        # use step lr
        else:
            if epoch in cfg['lr_epoch']:
                tmp_lr = tmp_lr * 0.1
                set_lr(optimizer, tmp_lr)
    

        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
                    # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)
            # # to device
            # images = images.to(device)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                size = random.randint(10, 19) * 32
                train_size = [size, size]
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = paddle.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            # make train label
            targets = [label.tolist() for label in targets]
            targets = tools.gt_creator(input_size=train_size, stride=yolo_net.stride, label_lists=targets)
            targets = paddle.to_tensor(targets)
            
            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets)

            # backprop
            total_loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            print(model.parameters()[0][0][0])

            # optimizer.clear_grad()


            # display
            if iter_i % 10 == 0:

                logging.info('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), train_size[0]))

                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d ]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), train_size[0]),
                        flush=True)


        # evaluation
        if (epoch + 1) % 1 == 0:
            model.trainable = False
            model.set_grid(val_size)
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            # convert to training mode.
            model.trainable = True
            model.set_grid(train_size)
            model.train()

        # save model
        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            paddle.save(model.state_dict(), './checkpoints/yolo-model.pdparams')


def set_lr(optimizer, lr):
    optimizer.set_lr(lr)



if __name__ == '__main__':
    train()
