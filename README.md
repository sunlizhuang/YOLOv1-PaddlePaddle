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





You will get a ```VOCdevkit.zip```, then what you need to do is just to unzip it and put it into ```data/```. After that, the whole path to VOC dataset is:

- ```data/VOCdevkit/VOC2007```
- ```data/VOCdevkit/VOC2012```.

#### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

#### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

# Train
```Shell
python train.py -d voc --cuda -v [select a model] -ms
```

You can run ```python train.py -h``` to check all optional argument.

## Test
```Shell
python test.py -d voc --cuda -v [select a model] --trained_model [ Please input the path to model dir. ]
```

## Evaluation
```Shell
python eval.py -d voc --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```


