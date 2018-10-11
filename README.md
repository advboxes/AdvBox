# AdvBox

![logo](pic/logo.png)

AdvBox是一款支持PaddlePaddle、PyTorch、Caffe2、Keras以及TensorFlow平台的AI模型安全工具箱。AdvBox同时支持白盒、黑盒攻击算法以及主流防御算法，支持列表如下。

### 白盒攻击算法

- L-BFGS
- FGSM
- BIM
- ILCM
- MI-FGSM
- JSMA
- DeepFool
- C/W

### 黑盒攻击算法

- Single Pixel Attack
- Local Search Attack


### 防护算法

- Feature Fqueezing
- Spatial Smoothing
- Label Smoothing
- Gaussian Augmentation
- Adversarial Training




对抗样本是深度学习领域的一个重要问题，比如在图像上叠加肉眼难以识别的修改，就可以欺骗主流的深度学习图像模型，产生分类错误，指鹿为马，或者无中生有。这些问题对于特定领域（比如无人车、人脸识别）会产生严重的后果，尤为重要。

![针对图像分类模型的对抗样本](pic/针对图像分类模型的对抗样本.png)

百度安全实验室研发了AdvBox，它能够为安全工程师研究模型的安全性提供极大的便利，免去重复造轮子的精力与时间消耗。AdvBox可以高效地使用最新的生成方法构造对抗样本数据集用于对抗样本的特征统计、攻击全新的AI应用，加固业务AI模型，为模型安全性研究和应用提供重要的支持，当前最新版本为[0.3](doc/RELEASE.cn.md)。

# 安装

## 部署AdvBox代码
直接同步advbox的代码，其中示例代码在tutorials目录下。

	git clone https://github.com/baidu/AdvBox.git  

## 初始化软件环境

为了兼容主流的深度学习平台，AdvBox基于python2.7开发，强烈建议使用Conda管理python软件环境，对应的python安装包安装方式如下。

	pip install -r requirements.txt

# 文档

## Keras示例

以Keras环境为例，代码路径为[tutorials/keras_demo.py](tutorials/keras_demo.py)

使用Keras自带的ResNet50模型进行白盒攻击，并设置为预测模式，加载测试图片。

	#设置为测试模式
    keras.backend.set_learning_phase(0)
    model = ResNet50(weights=modulename)
    img = image.load_img(imagename, target_size=(224, 224))
    original_image = image.img_to_array(img)
    imagedata = np.expand_dims(original_image, axis=0)

获取ResNet50的logit层，并创建keras对象。keras的ResNet50要求对原始图像文件进行标准化处理，mean值为[104, 116, 123]，std为1.

	#获取logit层
    logits=model.get_layer('fc1000').output
    # 创建keras对象
    # imagenet数据集归一化时 标准差为1  mean为[104, 116, 123]
    m = KerasModel(
        model,
        model.input,
        None,
        logits,
        None,
        bounds=(0, 255),
        channel_axis=3,
        preprocess=([104, 116, 123],1),
        featurefqueezing_bit_depth=8)

创建攻击对象，攻击算法使用FGSM的non-targeted attack，攻击步长epsilons设置为静态值。
	
	attack = FGSM(m)
	#静态epsilon
	attack_config = {"epsilons": 1, "epsilons_max": 10, "epsilon_steps": 1, "steps": 100}
	# fgsm non-targeted attack
	adversary = attack(adversary, **attack_config)

对比生成的对抗样本和原始图像的差别。
	
	adversary_image=np.copy(adversary.adversarial_example)
	#强制类型转换 之前是float 现在要转换成uint8
	#BGR -> RGB
	adversary_image=adversary_image[:,:,::-1]
	adversary_image = np.array(adversary_image).reshape([224,224,3])
	original_image=np.array(original_image).reshape([224, 224, 3])
	show_images_diff(original_image,adversary_image)

实际运行代码，原始图像和对抗样本的差别如下图所示。

   ![keras-demo.png](pic/keras-demo.png)

## PaddlePaddle示例
请见[PaddlePaddle示例](paddle.md)


##  学习教程

为了进一步降低学习成本，AdvBox提供大量的[学习教程](tutorials/README.md)。

 - [示例1：白盒攻击基于MNIST数据集的CNN模型](tutorials/README.md#示例1：白盒攻击基于MNIST数据集的CNN模型)
 - [示例2：白盒攻击基于CIFAR10数据集的ResNet模型](tutorials/README.md#示例2：白盒攻击基于CIFAR10数据集的ResNet模型)
 - [示例3：白盒攻击caffe下基于MNIST数据集的LeNet模型](tutorials/README.md#示例3：白盒攻击caffe下基于MNIST数据集的LeNet模型)
 - [示例4：黑盒攻击基于MNIST数据集的CNN模型](tutorials/README.md#示例4：黑盒攻击基于MNIST数据集的CNN模型)
 - [示例5：使用FeatureFqueezing加固基于MNIST数据集的CNN模型](tutorials/README.md#示例5：使用FeatureFqueezing加固基于MNIST数据集的CNN模型)
 - [示例6：使用GaussianAugmentation加固基于MNIST数据集的CNN模型](tutorials/README.md#示例6：使用GaussianAugmentation加固基于MNIST数据集的CNN模型)
 - [示例7：白盒攻击PyTorch下基于MNIST数据集的CNN模型](tutorials/README.md#示例7：白盒攻击PyTorch下基于MNIST数据集的CNN模型)

## 典型应用

基于AdvBox可以针对大量实际使用的AI模型生成对抗样本并给出通用加固方案。

 - [应用1：白盒攻击人脸识别系统](applications/face_recognition_attack/README.md)


# 问题反馈
	
目前支持通过Github提交[issues](https://github.com/baidu/AdvBox/issues)
。国内用户（需要通过国内手机号登录）也可以加入我们的知识星球进行交流，扫描二维码即可。
![知识星球](pic/issue.jpeg)

# 许可

AdvBox循序[Apache License 2.0](https://github.com/baidu/AdvBox/blob/master/LICENSE)

# 作者

- 百度安全实验室 xlab


# 参考文献

- http://www.paddlepaddle.org/docs/develop/documentation/en/build_and_install/pip_install_en.html
- http://paddlepaddle.org/docs/0.14.0/documentation/fluid/zh/new_docs/beginners_guide/install/install_doc.html
- https://github.com/PaddlePaddle/models/tree/develop/fluid/adversarial
- [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199), C. Szegedy et al., arxiv 2014
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), I. Goodfellow et al., ICLR 2015
- [Adversarial Examples In The Physical World](https://arxiv.org/pdf/1607.02533v3.pdf), A. Kurakin et al., ICLR workshop 2017
- [Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081), Yinpeng Dong et al., arxiv 2018
- [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/abs/1511.07528), N. Papernot et al., ESSP 2016
- [DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/abs/1511.04599), S. Moosavi-Dezfooli et al., CVPR 2016
- [Foolbox: A Python toolbox to benchmark the robustness of machine learning models](https://arxiv.org/abs/1707.04131), Jonas Rauber et al., arxiv 2018
- [CleverHans: An adversarial example library for constructing attacks, building defenses, and benchmarking both](https://github.com/tensorflow/cleverhans#setting-up-cleverhans)
- [Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey](https://arxiv.org/abs/1801.00553), Naveed Akhtar, Ajmal Mian, arxiv 2018


