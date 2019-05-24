# Advbox Family

[English](README.md)

![logo](pic/logo.png)

Advbox Family 是一系列AI模型安全的开源工具，由百度安全开源，包括针对AI模型的攻击、防护以及检测。

## AdvBox
AdvBox是一款由百度安全实验室研发，在百度大范围使用的AI模型安全工具箱，目前原生支持PaddlePaddle、PyTorch、Caffe2、MxNet、Keras以及TensorFlow平台，方便广大开发者和安全工程师可以使用自己熟悉的框架。AdvBox同时支持GraphPipe,屏蔽了底层使用的深度学习平台，用户可以零编码，仅通过几个命令就可以对PaddlePaddle、PyTorch、Caffe2、MxNet、CNTK、ScikitLearn以及TensorFlow平台生成的模型文件进行黑盒攻击。

[Homepage of AdvBox](advbox-ch.md)

## AdvDetect
AdvDetect是一款从海量数据中检测对抗样本的工具。

[Homepage of AdvDetect](advbox_family/AdvDetect/README.md)


# AI应用攻防

## 攻击人脸识别

[Homepage of Face Recogniztion Attack](applications/face_recognition_attack/README.md)

## 消失的T恤

defcon上我们演示了可以在智能摄像头下消失的T恤，该子项目下开源了演示使用的智能摄像头的程序以及部署方法。

[FStealth T-shirt](applications/fFStealthTshirt/README.md)

## Advbox Family的ppt和论文

# Issues report
	
[https://github.com/baidu/AdvBox/issues](https://github.com/baidu/AdvBox/issues)

# License

AdvBox support [Apache License 2.0](https://github.com/baidu/AdvBox/blob/master/LICENSE)

# Authors

- Baidu xlab


# 如何引用

If you instead use AdvBox in an academic publication, cite as:

	@misc{advbox,
	 author= {Dou Goodman,Wang Yang,Hao Xin},
	 title = {Advbox:a toolbox to generate adversarial examples that fool neural networks},
	 month = mar,
	 year  = 2019,
	 url   = {https://github.com/baidu/AdvBox}
	}
