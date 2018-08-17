# 使用示例
## 概述
主流的深度学习平台都提供了大量的预训练模型，以PaddlePaddle为例，基于ImageNet 2012数据集提供了AlexNet、ResNet50和SE_ResNeXt50_32x4d模型文件，完整的模型地址为：

[https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification#supported-models-and-performances](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification#supported-models-and-performances)

## 示例1：攻击基于MNIST数据集的CNN模型
首先需要生成攻击用的模型，advbox的测试模型是一个识别MNIST的cnn模型。

	python mnist_model.py
	
运行攻击代码，以基于FGSM算法的演示代码为例。

	python mnist_tutorial_fgsm.py
	
运行攻击脚本，对MNIST数据集进行攻击，测试样本数量为500，其中攻击成功394个，占78.8%。

	attack success, original_label=4, adversarial_label=9, count=498
	attack success, original_label=8, adversarial_label=3, count=499
	attack success, original_label=6, adversarial_label=1, count=500
	[TEST_DATASET]: fooling_count=394, total_count=500, fooling_rate=0.788000
	fgsm attack done
	
完整的攻击算法和代码对应关系如下：

| 攻击算法 | 代码文件 | 
| ------ | ------ | 
| lbfgs | mnist\_tutorial\_lbfgs.py | 
| fgsm | mnist\_tutorial\_fgsm.py | 
| bim | mnist\_tutorial\_bim.py | 
| ilcm | mnist\_tutorial\_ilcm.py | 
| mifgsm | mnist\_tutorial\_mifgsm.py | 
| jsma | mnist\_tutorial\_jsma.py | 
| deepfool | mnist\_tutorial\_deepfool.py | 


## 示例2：攻击基于CIFAR10数据集的ResNet模型
首先需要生成攻击用的模型，advbox的测试模型是一个识别CIFAR10的ResNet模型。

	python cifar10_model.py
	
运行攻击代码，以基于FGSM算法的演示代码为例。

	python cifar10_tutorial_fgsm.py
		
完整的攻击算法和代码对应关系如下：

| 攻击算法 | 代码文件 | 
| ------ | ------ | 
| bim | cifar10\_tutorial\_bim.py | 
| fgsm | cifar10\_tutorial\_fgsm.py | 
| deepfool | cifar10\_tutorial\_deepfool.py | 
| jsma | cifar10\_tutorial\_ jsma.py | 
