# V0.4版本
- 新增支持平台：
	- PyTorch
	- MxNet
- 支持白盒攻击算法：
	- C/W 
	
# V0.3版本
- 新增支持平台：
	- Keras
- 支持黑盒攻击算法：
	- Single Pixel Attack
	- Local Search Attack

- 支持防护算法：
	- Feature Fqueezing
	- Spatial Smoothing
	- Label Smoothing
	- Gaussian Augmentation
	- Adversarial Training
- 新增了使用案例：
	- 白盒攻击caffe下基于MNIST数据集的LeNet模型
	- 黑盒攻击基于MNIST数据集的CNN模型
	- 使用FeatureFqueezing加固基于MNIST数据集的CNN模型
	- 使用GaussianAugmentation加固基于MNIST数据集的CNN模型

# V0.2版本
- 新增对Caffe平台的支持
- 多GPU支持，目前基于nccl2框架，nccl2支持的NVIDIA GPU均可以使用
- 新增了针对cifar10数据集的ResNet和Vgg16模型的攻击案例

# V0.1版本（盘古开天地版本）
- 支持PaddlePaddle
- 支持白盒攻击算法：
	- L-BFGS
	- FGSM
	- BIM
	- ILCM
	- MI-FGSM
	- JSMA
	- DeepFool
