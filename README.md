# YOLOv1-PaddlePaddle

# 如果需要验证mAP是否符合精度，在AIStudio上Fork后，在work/datasets解压好测试数据集后，直接运行python eval.py 即可，已经上传checkpoints

# Network
This is a a new version of YOLOv1 built by PaddlePaddle:
- Backbone: resnet18


# Train
- Batchsize: 32
- Base lr: 1e-3
- Max epoch: 160
- LRstep: 60, 90
- optimizer: SGD
采用动态调整学习率的策略，mAP达到65.42，大于原文的63.4


## Experiment
Environment:

- Python3.6, opencv-python, PaddlePaddle 2.1.0

- VOC:
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> mAP </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC2007 test</th><td bgcolor=white> 416 </td><td bgcolor=white> 65.42</td></tr>
</table></tbody>


## Try On AI Studio
https://aistudio.baidu.com/aistudio/projectdetail/2259467




You will see a ```VOCdevkit.zip``` in ```datasets/```, then what you need to do is just to unzip it. After that, the whole path to VOC dataset is:

- ```datasets/VOCdevkit/VOC2007```


#### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```


# Train
```Shell
python train.py
```


## Test
```Shell
python test.py
```

## Evaluation
```Shell
python eval.py
```


