#coding=utf-8

# Copyright 2017 - 2018 Baidu Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FGSM tutorial on mnist using advbox tool.
FGSM method is non-targeted attack while FGSMT is targeted attack.
"""
from __future__ import print_function
import sys
sys.path.append("..")
import logging
logging.basicConfig(level=logging.INFO,format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger=logging.getLogger(__name__)

#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#pip install Pillow

from adversarialbox.adversary import Adversary
#from adversarialbox.attacks.gradient_method import FGSM
#from adversarialbox.attacks.gradient_method import FGSMT
from adversarialbox.attacks.deepfool import DeepFoolAttack
from adversarialbox.models.tensorflow import TensorflowModel
from tutorials.mnist_model_tf import mnist_cnn_model
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main(dirname,imagename):

    #加载解码的图像 这里是个大坑 tf提供的imagenet预训练好的模型pb文件中 包含针对图像的预处理环节 即解码jpg文件 这部分没有梯度
    #需要直接处理解码后的数据
    image=None
    with tf.gfile.Open(imagename, 'rb') as f:
        image = np.array(
            Image.open(f).convert('RGB')).astype(np.float)

    image=[image]


    session=tf.Session()

    def create_graph(dirname):
        with tf.gfile.FastGFile(dirname, 'rb') as f:
            graph_def = session.graph_def
            graph_def.ParseFromString(f.read())

            _ = tf.import_graph_def(graph_def, name='')

    create_graph(dirname)

    # 初始化参数  非常重要
    session.run(tf.global_variables_initializer())

    tensorlist=[n.name for n in session.graph_def.node]

    logger.info(tensorlist)

    #获取logits
    logits=session.graph.get_tensor_by_name('softmax/logits:0')

    x = session.graph.get_tensor_by_name('ExpandDims:0')

    #y = tf.placeholder(tf.int64, None, name='label')

    # advbox demo
    # 因为原始数据没有归一化  所以bounds=(0, 255)
    m = TensorflowModel(
        session,
        x,
        None,
        logits,
        None,
        bounds=(0, 255),
        channel_axis=3,
        preprocess=None)

    attack = DeepFoolAttack(m)
    attack_config = {"iterations": 100, "overshoot": 0.02}

    #y设置为空 会自动计算
    adversary = Adversary(image,None)

    # FGSM non-targeted attack
    adversary = attack(adversary, **attack_config)

    if adversary.is_successful():
        print(
            'attack success, adversarial_label=%d'
            % (adversary.adversarial_label) )

        #对抗样本保存在adversary.adversarial_example
        adversary_image=np.copy(adversary.adversarial_example)

        #print(adversary_image - image)

        #强制类型转换 之前是float 现在要转换成int8
        adversary_image = np.array(adversary_image).astype("uint8").reshape([100,100,3])

        logging.info(adversary_image - image)
        #print(adversary_image - image)

        im = Image.fromarray(adversary_image)
        im.save("adversary_image_nontarget.jpg")

    print("DeepFool non-target attack done")



    attack = DeepFoolAttack(m)
    attack_config = {"iterations": 100, "overshoot": 0.05}

    adversary = Adversary(image,None)
    #麦克风
    tlabel = 651
    adversary.set_target(is_targeted_attack=True, target_label=tlabel)

    # FGSM targeted attack
    adversary = attack(adversary, **attack_config)

    if adversary.is_successful():
        print(
            'attack success, adversarial_label=%d'
            % (adversary.adversarial_label) )

        #对抗样本保存在adversary.adversarial_example
        adversary_image=np.copy(adversary.adversarial_example)
        #强制类型转换 之前是float 现在要转换成int8

        logging.info(adversary_image-image)

        adversary_image = np.array(adversary_image).astype("uint8").reshape([100,100,3])
        im = Image.fromarray(adversary_image)
        im.save("adversary_image_target.jpg")



    print("DeepFool target attack done")



if __name__ == '__main__':
    #从'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'下载并解压到当前路径
    #classify_image_graph_def.pb cropped_panda.jpg
    #imagenet2012 中文标签 https://blog.csdn.net/u010165147/article/details/72848497
    main(dirname="classify_image_graph_def.pb",imagename="cropped_panda.jpg")
