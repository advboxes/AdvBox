# V0.2版本
- 新增对Caffe以及TensorFlow平台的支持
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
