import numpy as np
from data import *
import paddle.nn as nn
import paddle
paddle.disable_static()
CLASS_COLOR = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(len(VOC_CLASSES))]


class MSELoss(nn.Layer):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets):
        targets_np=targets.numpy()
        pos_id=(targets_np==1.0).astype('float')
        neg_id=(targets_np==0.0).astype('float')

        pos_id = paddle.to_tensor(pos_id).astype('float32')
        neg_id = paddle.to_tensor(neg_id).astype('float32')
        
        i_t_square=paddle.square(inputs - targets)
        pos_id=pos_id.numpy()
        i_t_square=i_t_square.numpy()
        pos_loss=pos_id*i_t_square
        pos_loss=paddle.to_tensor(pos_loss)
        i_square=paddle.square(inputs)
        neg_id=neg_id.numpy()
        i_square=i_square.numpy()
        neg_loss = neg_id * i_square
        neg_loss=paddle.to_tensor(neg_loss)

        if self.reduction == 'mean':
            pos_loss = paddle.mean(paddle.sum(pos_loss, 1))
            neg_loss = paddle.mean(paddle.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss


def generate_dxdywh(gt_label, w, h, s):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # compute the center, width and height
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1. or box_h < 1.:
        # print('A dirty data !!!')
        return False    

    # map center point of box to the grid cell
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # compute the (x, y, w, h) for the corresponding grid cell
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)
    th = np.log(box_h)
    weight = 2.0 - (box_w / w) * (box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight


def gt_creator(input_size, stride, label_lists=[], name='VOC'):
    assert len(input_size) > 0 and len(label_lists) > 0
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    w = input_size[1]
    h = input_size[0]
    
    # We  make gt labels by anchor-free method and anchor-based method.
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, 1+1+4+1])

    # generate gt whose style is yolo-v1
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_class = int(gt_label[-1])
            result = generate_dxdywh(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight = result

                if grid_x < gt_tensor.shape[2] and grid_y < gt_tensor.shape[1]:
                    gt_tensor[batch_index, grid_y, grid_x, 0] = 1.0
                    gt_tensor[batch_index, grid_y, grid_x, 1] = gt_class
                    gt_tensor[batch_index, grid_y, grid_x, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[batch_index, grid_y, grid_x, 6] = weight


    gt_tensor = gt_tensor.reshape(batch_size, -1, 1+1+4+1)

    return gt_tensor


def loss(pred_conf, pred_cls, pred_txtytwth, label):
    obj = 5.0
    noobj = 1.0

    # create loss_f
    conf_loss_function = MSELoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='mean')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    pred_conf = paddle.nn.functional.sigmoid(pred_conf[:, :, 0])
    pred_cls=paddle.fluid.layers.transpose(pred_cls,perm=[0,2,1])
    # pred_cls = pred_cls.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]
        
    gt_obj = label[:, :, 0]
    gt_cls = label[:, :, 1]
    gt_txtytwth = label[:, :, 2:-1]
    gt_box_scale_weight = label[:, :, -1]

    # objectness loss

    pred_conf=pred_conf.astype("float32")
    gt_obj=gt_obj.astype("float32")
    pos_loss, neg_loss = conf_loss_function(pred_conf,gt_obj)

    conf_loss = obj * pos_loss + noobj * neg_loss


    
    # class loss
    pred_cls=pred_cls.astype('float32')
    gt_cls=gt_cls.astype('int64')
    pred_cls_t=paddle.fluid.layers.transpose(pred_cls,perm=[2,0,1])
    gt_cls_t=paddle.fluid.layers.transpose(gt_cls,perm=[1,0])

    cls_loss_total=[]
    for i in range(169):
        temp=[]
        for j in range(32):
            temp.append(cls_loss_function(pred_cls_t[i][j],gt_cls_t[i][j]).numpy())
               
        cls_loss_total.append(temp)
    cls_loss_total=paddle.to_tensor(cls_loss_total)
    cls_loss_total=paddle.reshape(cls_loss_total,shape=[32,169])
    
    cls_loss_total=cls_loss_total.numpy()
    gt_obj=gt_obj.numpy()
    temp_result=cls_loss_total * gt_obj
    temp_result=paddle.to_tensor(temp_result).astype('float32')
    

    cls_loss = paddle.mean(paddle.sum(temp_result, 1))

    
    # box loss
    pred_txty=pred_txty.astype('float32')
    gt_txtytwth=gt_txtytwth.astype('float32')

    temp_result1_np=paddle.sum(txty_loss_function(pred_txty, gt_txtytwth[:, :, :2]), 2).numpy()
    gt_box_scale_weight_np=gt_box_scale_weight.numpy()
    gt_obj_np=gt_obj
    temp_result_2=temp_result1_np*gt_box_scale_weight_np*gt_obj_np
    temp_result_2=paddle.to_tensor(temp_result_2)
    
    temp_result3_np=paddle.sum(twth_loss_function(pred_twth, gt_txtytwth[:, :, 2:]), 2).numpy()
    temp_result4=temp_result3_np*gt_box_scale_weight_np*gt_obj_np
    temp_result4=paddle.to_tensor(temp_result4)



    txty_loss = paddle.mean(paddle.sum(temp_result_2,1))
    twth_loss = paddle.mean(paddle.sum(temp_result4, 1))

    txtytwth_loss = txty_loss + twth_loss

    total_loss = conf_loss + cls_loss + txtytwth_loss

    return conf_loss, cls_loss, txtytwth_loss, total_loss


if __name__ == "__main__":
    pass