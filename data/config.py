# config.py
import os.path


# new yolo config
train_cfg = {
    'lr_epoch': (60, 90, 160),
    'max_epoch': 160,
    'min_dim': [416, 416]
}
