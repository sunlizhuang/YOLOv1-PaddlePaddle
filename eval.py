import paddle.device
import paddle
from data import *
import argparse
from utils.vocapi_evaluator import VOCAPIEvaluator



parser = argparse.ArgumentParser(description='YOLO Detector Evaluation')
parser.add_argument('-v', '--version', default='yolo',
                    help='yolo.')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val, coco-test.')
parser.add_argument('--trained_model', type=str,
                    default='./weight/voc/yolo_64.4_68.5_71.5.pth',
                    help='Trained state_dict file path to open')
parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='input_size')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')

args = parser.parse_args()



def voc_test(model, device, input_size):
    evaluator = VOCAPIEvaluator(data_root=VOC_ROOT,
                                img_size=input_size,
                                device=device,
                                transform=BaseTransform(input_size),
                                labelmap=VOC_CLASSES,
                                display=True
                                )

    # VOC evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        device = paddle.device.set_device("cuda")
    else:
        device = paddle.device.set_device("cpu")

    # input size
    input_size = [args.input_size, args.input_size]

    # build model
    if args.version == 'yolo':
        from models.yolo import myYOLO
        net = myYOLO(device, input_size=input_size, num_classes=num_classes, trainable=False)

    else:
        print('Unknown Version !!!')
        exit()

    # load net
    # net.load_state_dict(paddle.load(args.trained_model, map_location='cuda'))
    net.load_state_dict(paddle.load(args.trained_model,map_location='cpu'))
    net.eval()
    print('Finished loading model!')
    net = net.to(device)
    
    # evaluation
    with paddle.no_grad():
        if args.dataset == 'voc':
            voc_test(net, device, input_size)
