# ADVSDK
ADVSDK是一款针对PaddlePaddle框架定制的轻量级SDK。目前支持学术届和工业界公认的基线测试算法，并且支持输出$L_0$、$L_2$和$L_(inf)$并可视化。
## 支持算法

- PGD
- FGSM	

## 使用教程
全部教程使用jupter编写，方便使用和阅读。

- [攻击AlexNet](sdk_demo_alexnet.ipynb)

- [攻击ResNet](sdk_demo.ipynb)

## 模型文件地址
	https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification
	
	wget http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar
	tar -xvf ResNet50_pretrained.tar
	
[更多模型](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV)	
	
## 初始化环境

	pip install opencv-python
	pip install paddlepaddle==1.5

