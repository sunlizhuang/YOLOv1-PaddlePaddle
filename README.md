# YOLOv1-PaddlePaddle


# Network
This is a a new version of YOLOv1 built by PaddlePaddle:
- Backbone: resnet18
- Head: SPP, SAM

# Train
- Batchsize: 32
- Base lr: 1e-3
- Max epoch: 160
- LRstep: 60, 90
- optimizer: SGD


## Experiment
Environment:

- Python3.6, opencv-python, PaddlePaddle 2.1.0
- 
- VOC:
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> mAP </td><td bgcolor=white> FPS </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC2007 test</th><td bgcolor=white> 320 </td><td bgcolor=white> 64.4 </td><td bgcolor=white> - </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC2007 test</th><td bgcolor=white> 416 </td><td bgcolor=white> 68.5 </td><td bgcolor=white> - </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC2007 test</th><td bgcolor=white> 608 </td><td bgcolor=white> 71.5 </td><td bgcolor=white> - </td></tr>
</table></tbody>


## Try On AI Studio
https://aistudio.baidu.com/aistudio/projectdetail/2259467




You will get a ```VOCdevkit.zip```, then what you need to do is just to unzip it and put it into ```data/```. After that, the whole path to VOC dataset is:

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


