# 使用示例
## 概述
主流的深度学习平台都提供了大量的预训练模型，以PaddlePaddle为例，基于ImageNet 2012数据集提供了AlexNet、ResNet50和SE\_ResNeXt50\_32x4d模型文件，完整的模型地址为：

[https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification#supported-models-and-performances](https://github.com/PaddlePaddle/models/tree/develop/fluid/image_classification#supported-models-and-performances)

## 目录

 - 示例1:白盒攻击基于MNIST数据集的CNN模型
 - 示例2:白盒攻击基于CIFAR10数据集的ResNet模型
 - 示例3:白盒攻击caffe下基于MNIST数据集的LeNet模型
 - 示例4:黑盒攻击基于MNIST数据集的CNN模型
 - 示例5:使用FeatureFqueezing加固基于MNIST数据集的CNN模型
 - 示例6:使用GaussianAugmentation加固基于MNIST数据集的CNN模型
 - 示例7:白盒攻击PyTorch下基于MNIST数据集的CNN模型
 - 示例8:白盒攻击PyTorch下基于IMAGENET数据集的AlexNet模型
 - 示例9:白盒攻击MxNet下基于IMAGENET数据集的AlexNet模型
 - 示例10:黑盒攻击graphpipe下的基于tensorflow的squeezenet模型
 - 示例11:黑盒攻击graphpipe下的基于onnx的squeezenet模型
 
 
## 示例1:白盒攻击基于MNIST数据集的CNN模型
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


## 示例2:白盒攻击基于CIFAR10数据集的ResNet模型
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

## 示例3:白盒攻击caffe下基于MNIST数据集的LeNet模型
首先使用Paddle提供的caffe2fluid工具把caffe下在MNIST数据集上训练好的LeNet模型转换成Paddle可以识别的格式。caffe2fluid的地址为[caffe2fluid](https://github.com/PaddlePaddle/models/blob/e7684f07505c172beb4c4d9febb4a48f9fa83b68/fluid/image_classification/caffe2fluid/README.md)

Caffe2Fluid的安装非常简单，在Caffe2Fluid目录下执行。

	bash ./proto/compile.sh
	cd proto/ && wget https://github.com/ethereon/caffe-tensorflow/blob/master/kaffe/caffe/caffepb.py

下载caffe下基于MNIST数据集的LeNet模型，地址如下：

	https://github.com/ethereon/caffe-tensorflow/blob/master/examples/mnist

caffe的模型文件通常有两个组成，假设保存到models.caffe/lenet/目录下。

	models.caffe/lenet/lenet.prototxt
	models.caffe/lenet/lenet.caffemodel

使用Caffe2Fluid工具生成模型代码文件lenet.py以及参数文件lenet.npy

	python convert.py models.caffe/lenet/lenet.prototxt --caffemodel models.caffe/lenet/lenet.caffemodel --data-output-path lenet.npy --code-output-path lenet.py

运行攻击代码，以基于FGSM算法的演示代码为例。

	python cifar10_tutorial_fgsm.py
	
运行结果如下，基于FGSM算法攻击成功率为91.6%。

	[TEST_DATASET]: fooling_count=458, total_count=500, fooling_rate=0.916000
	fgsm attack done
	
## 示例4:黑盒攻击基于MNIST数据集的CNN模型
首先需要生成攻击用的模型，advbox的测试模型是一个识别MNIST的cnn模型。

	python mnist_model.py
	
运行攻击代码，以基于SinglePixelAttack算法的演示代码为例。

	python mnist_tutorial_singlepixelattack.py
	
运行结果如下，基于SinglePixelAttack算法攻击成功率为100%。

	[TEST_DATASET]: fooling_count=10, total_count=10, fooling_rate=1.000000
	SinglePixelAttack attack done
	
	
## 示例5:使用FeatureFqueezing加固基于MNIST数据集的CNN模型
首先需要生成攻击用的模型，advbox的测试模型是一个识别MNIST的cnn模型。

	python mnist_model.py
	
然后运行攻击代码，攻击FeatureFqueezing加固后的CNN模型。

	python mnist_tutorial_defences_feature_squeezing.py
	
运行结果如下，首先攻击的没有加固的CNN模型，攻击成功率为54.6%

	[TEST_DATASET]: fooling_count=273, total_count=500, fooling_rate=0.546000
	fgsm attack done without any defence
	
攻击eatureFqueezing加固后的CNN模型，攻击成功率下降为10.6%。

	[TEST_DATASET]: fooling_count=53, total_count=500, fooling_rate=0.106000
	fgsm attack done with FeatureFqueezingDefence
	
## 示例6:使用GaussianAugmentation加固基于MNIST数据集的CNN模型
首先需要生成攻击用的模型，advbox的测试模型是一个识别MNIST的cnn模型，模型保存在mnist目录下。

	python mnist_model.py
	
接着运行GaussianAugmentation加固的模型，模型保存在mnist-gad目录下。

	python mnist_model_gaussian_augmentation_defence.py


最后运行攻击代码，攻击GaussianAugmentation加固后的CNN模型。

	python mnist_tutorial_defences_gaussian_augmentation.py
	
运行结果如下，首先攻击的没有加固的CNN模型，攻击成功率为54.6%

	[TEST_DATASET]: fooling_count=282, total_count=500, fooling_rate=0.564000 fgsm attack done without any defence
	
攻击加固后的CNN模型，攻击成功率下降为36.2%。

	[TEST_DATASET]: fooling_count=181, total_count=500, fooling_rate=0.362000 fgsm attack done with  GaussianAugmentationDefence
	
## 示例7：白盒攻击PyTorch下基于MNIST数据集的CNN模型
首先需要生成攻击用的模型，advbox的测试模型是一个识别MNIST的cnn模型，模型保存在mnist目录下。
	
	python mnist_model_pytorch.py

运行攻击代码，以基于FGSM算法的演示代码为例。

	python mnist_tutorial_fgsm_pytorch.py
	
## 示例8:白盒攻击PyTorch下基于IMAGENET数据集的AlexNet模型
首先运行工具imagenet\_tools\_pytorch.py，对原始图片cropped_panda.jpg进行预测，预测标签为388

	python imagenet_tools_pytorch.py cropped_panda.jpg 
	image_path:cropped_panda.jpg
	imagenet_tools_pytorch.py[line:46] INFO CUDA Available: False
	label=388

运行攻击脚本，生成对抗样本图片img_adv.png。

	python imagenet_tutorial_fgsm_pytorch.py

首先运行工具imagenet\_tools\_pytorch.py，对图片img_adv.png进行预测，预测标签为538，满足预期。

	python imagenet_tools_pytorch.py img_adv.png 
	image_path:img_adv.png
	imagenet_tools_pytorch.py[line:46] INFO CUDA Available: False
	label=538

## 示例9:白盒攻击MxNet下基于IMAGENET数据集的AlexNet模型
首先运行工具imagenet\_tools\_mxnet.py，对原始图片cropped_panda.jpg进行预测，预测标签为388

	python imagenet_tools_mxnet.py cropped_panda.jpg 
	image_path:cropped_panda.jpg
	388

运行攻击脚本，生成对抗样本图片img_adv.png。

	python imagenet_tutorial_fgsm_mxnet.py

首先运行工具imagenet_tools_mxnet.py，对图片img_adv.png进行预测，预测标签为538，满足预期。

	python imagenet_tools_mxnet.py img_adv.png 
	image_path:img_adv.png
	538

## 示例10:黑盒攻击graphpipe下的基于tensorflow的squeezenet模型
建议在docker环境下运行graphpipe，启动graphpipe下的基于tensorflow的squeezenet模型。

	docker run -it --rm \
	    -e https_proxy=${https_proxy} \
	    -p 9000:9000 \
	    sleepsonthefloor/graphpipe-tf:cpu \
	    --model=https://oracle.github.io/graphpipe/models/squeezenet.pb \
	    --listen=0.0.0.0:9000
	    
安装graphpipe的python客户端库。

	pip install graphpipe
	pip install pillow 
	
在applications/graphpipe下运行攻击代码，攻击成功，label由504变成659。

	python graphpipe_localsearch.py
	localsearch.py[line:293] INFO try 10 times  selected pixel indices:[ 0 16 15 14 13]
	localsearch.py[line:308] INFO adv_label=659 adv_label_pro=0.00122496963013
	attack success, original_label=504, adversarial_label=659
	graphpipe_localsearch.py[line:100] INFO LocalSearchAttack attack done
	
运行验证工具，生成的对抗样本确实识别为659.

	python graphpipe_tool.py
	image_path:adversary_image.jpg
	[659]

## 示例11:黑盒攻击graphpipe下的基于onnx的squeezenet模型
建议在docker环境下运行graphpipe，启动graphpipe下的基于onnx的squeezenet模型。

	docker run -it --rm \
	    -e https_proxy=${https_proxy} \
	    -p 9000:9000 \
	    sleepsonthefloor/graphpipe-onnx:cpu \
	    --model=https://oracle.github.io/graphpipe/models/squeezenet. onnx \
	    --listen=0.0.0.0:9000
	    	
在applications/graphpipe下运行攻击代码，攻击成功，label由504变成659。

	python graphpipe_onnx_localsearch.py
	localsearch.py[line:293] INFO try 10 times  selected pixel indices:[ 0 16 15 14 13]
	localsearch.py[line:308] INFO adv_label=659 adv_label_pro=0.00122496963013
	attack success, original_label=504, adversarial_label=659
	graphpipe_onnx_localsearch.py[line:100] INFO LocalSearchAttack attack done
	
运行验证工具，生成的对抗样本确实识别为659.

	python graphpipe_onnx_tool.py
	image_path:adversary_image.jpg
	[659]

