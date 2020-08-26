# Advbox Family

![logo](pic/logo.png)

Advbox Family is a series of AI model security tools set of Baidu Open Source,including the generation, detection and protection of adversarial examples, as well as attack and defense cases for different AI applications.

Advbox Family  support Python 3.*.

## Our Work

- [Tracking the Criminal of Fake News Based on a Unified Embedding. Blackhat Asia 2020](https://www.blackhat.com/asia-20/briefings/schedule/index.html#tracking-the-criminal-of-fake-news-based-on-a-unified-embedding-18388)
- [Attacking and Defending Machine Learning Applications of Public Cloud. Blackhat Asia 2020](https://www.blackhat.com/asia-20/briefings/schedule/#attacking-and-defending-machine-learning-applications-of-public-cloud-18725)
- [ABSTRACT:Cloud-based Image Classification Service Is Not Robust To Affine Transformation : A Forgotten Battlefield. CCSW 2019: The ACM Cloud Computing Security Workshop of CCS 2019](https://ccsw.io/#speakers)
- [TRANSFERABILITY OF ADVERSARIAL EXAMPLES TO ATTACK REAL WORLD PORN IMAGES DETECTION SERVICE.HITB CyberWeek 2019](https://cyberweek.ae/session/transferability-of-adversarial-examples-to-attack-real-world-porn-images-detection-service/)
- [COMMSEC: Tracking Fake News Based On Deep Learning. HITB GSEC 2019](https://gsec.hitb.org/sg2019/sessions/commsec-tracking-fake-news-based-on-deep-learning/)
- [COMMSEC: Hacking Object Detectors Is Just Like Training Neural Networks. HITB GSEC 2019](https://gsec.hitb.org/sg2019/sessions/commsec-hacking-object-detectors-is-just-like-training-neural-networks/) | See [code](https://github.com/advboxes/AdvBox/blob/master/advbox_family/ODD/README.md)
- [COMMSEC: How to Detect Fake Faces (Manipulated Images) Using CNNs. HITB GSEC 2019](https://gsec.hitb.org/sg2019/sessions/commsec-how-to-detect-fake-faces-manipulated-images-using-cnns/)
- [Transferability of Adversarial Examples to Attack Cloud-based Image Classifier Service. Defcon China 2019](https://www.defcon.org/html/dc-china-1/dc-cn-1-speakers.html)
- [Face Swapping Video Detection with CNN. Defcon China 2019](https://www.defcon.org/html/dc-china-1/dc-cn-1-speakers.html)

 

## AdvSDK

A Lightweight Adv SDK For PaddlePaddle to generate adversarial examples.

[Homepage of AdvSDK](advsdk/README.md) 


## AdversarialBox
Adversarialbox is a toolbox to generate adversarial examples that fool neural networks in PaddlePaddle、PyTorch、Caffe2、MxNet、Keras、TensorFlow and Advbox can benchmark the robustness of machine learning models.Advbox give a command line tool to generate adversarial examples with Zero-Coding. It is inspired and based on FoolBox v1. 

[Homepage of AdversarialBox](adversarialbox.md)

## AdvDetect
AdvDetect is a toolbox to detect adversarial examples from massive data.

[Homepage of AdvDetect](advbox_family/AdvDetect/README.md)


## AdvPoison

Data poisoning

# AI applications

## Face Recognition Attack

[Homepage of Face Recognition Attack](applications/face_recognition_attack/README.md)

## Stealth T-shirt
On defcon, we demonstrated T-shirts that can disappear under smart cameras. Under this sub-project, we open-source the programs and deployment methods of smart cameras for demonstration.

[Homepage of Stealth T-shirt](applications/StealthTshirt/README.md)

![pic1](applications/StealthTshirt/output.gif)

## Fake Face Detect

The restful API is used to detect whether the face in the picture/video is a false face.

[Homepage of Fake Face Detect](applications/fake_face_detect/README.md)

![pic2](pic/deepfake02.png)



## Paper and ppt of Advbox Family




# How to cite

If you use AdvBox in an academic publication, please cite as:

	@misc{goodman2020advbox,
	    title={Advbox: a toolbox to generate adversarial examples that fool neural networks},
	    author={Dou Goodman and Hao Xin and Wang Yang and Wu Yuesheng and Xiong Junfeng and Zhang Huan},
	    year={2020},
	    eprint={2001.05574},
	    archivePrefix={arXiv},
	    primaryClass={cs.LG}
	}

Cloud-based Image Classification Service is Not Robust to Affine Transformation: A Forgotten Battlefield

	@inproceedings{goodman2019cloud,
	  title={Cloud-based Image Classification Service is Not Robust to Affine Transformation: A Forgotten Battlefield},
	  author={Goodman, Dou and Hao, Xin and Wang, Yang and Tang, Jiawei and Jia, Yunhan and Wei, Tao and others},
	  booktitle={Proceedings of the 2019 ACM SIGSAC Conference on Cloud Computing Security Workshop},
	  pages={43--43},
	  year={2019},
	  organization={ACM}
	}

# Who use/cite AdvBox

- Pablo Navarrete Michelini, Hanwen Liu, Yunhua Lu, Xingqun Jiang; A Tour of Convolutional Networks Guided by Linear Interpreters; The IEEE International Conference on Computer Vision (ICCV), 2019, pp. 4753-4762
- Ling, Xiang and Ji, Shouling and Zou, Jiaxu and Wang, Jiannan and Wu, Chunming and Li, Bo and Wang, Ting; Deepsec: A uniform platform for security analysis of deep learning model ; IEEE S\&P, 2019
- Deng, Ting and Zeng, Zhigang; Generate adversarial examples by spatially perturbing on the meaningful area; Pattern Recognition Letters[J], 2019, pp. 632-638 


# Issues report

[https://github.com/baidu/AdvBox/issues](https://github.com/baidu/AdvBox/issues)

# License

AdvBox support [Apache License 2.0](https://github.com/baidu/AdvBox/blob/master/LICENSE)
