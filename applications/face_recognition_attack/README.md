# Face Recogniztion Attack

## White box attack
We chose a pre-trained FaceNet model that is a state-of-the-art and widely used face-recognition system as our white-box attacked model. We used gradient-based attacks methods and modify its loss function using FaceNet embedding distance.

## Install

FaceNet and its pre-trained model can be downloaded from `https://github.com/davidsandberg/facenet`.

	cd thirdparty

### Clone facenet

	git clone https://github.com/davidsandberg/facenet.git

### Download pre-trained model

baidu netdisk 

	https://pan.baidu.com/s/1xWj1wW6MgoOI2MFAejr0oQ 

code: kdxd 

you can get 20180402-114759.zip，unzip it，get 20180402-114759.pb

	cd applications/face_recognition_attack
	total 93600
	drwxr-xr-x 3 root root     4096 Sep 17 16:59 ./
	drwxr-xr-x 3 root root     4096 Sep 17 16:33 ../
	-rw-r--r-- 1 root root 95745767 Apr  9 14:45 20180402-114759.pb
	-rw-r--r-- 1 root root    34639 Sep 17 16:33 Bill_Gates_0001.png
	drwxr-xr-x 9 root root     4096 Sep 17 16:58 facenet/
	-rw-r--r-- 1 root root     6114 Sep 17 16:33 facenet_fr.py
	-rw-r--r-- 1 root root    36618 Sep 17 16:33 Michael_Jordan_0002.png
	-rw-r--r-- 1 root root      388 Sep 17 16:33 README.md
	
## Attack face recogniztion



![Bill_Gates_0001.png](Bill_Gates_0001.png)
![Michael_Jordan_0002.png](Michael_Jordan_0002.png)

	python facenet_fr.py

![Bill_Gates_0001_2_Michael_Jordan_0002.png](Bill_Gates_0001_2_Michael_Jordan_0002.png)