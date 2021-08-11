import paddle.device
from thop import profile
from models.yolo_v1 import myYOLOv1 as net_1



device = paddle.device.set_device("gpu")

def main():
    input_image = paddle.randn(1, 3, 320, 640).to(device)
    input_size = [320, 640]
    num_classes = 20
    model = net_1(device, input_size=input_size, num_classes=num_classes, trainable=True).to(device)
    flops, params = profile(model, inputs=(input_image, ))
    print('FLOPs : ', flops / 1e9, ' B')
    print('Params : ', params / 1e6, ' M')


if __name__ == "__main__":
    main()
