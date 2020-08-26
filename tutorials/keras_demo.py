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
from __future__ import absolute_import
import sys
sys.path.append("..")
import logging
#logging.basicConfig(level=logging.INFO,format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
#logger=logging.getLogger(__name__)

#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#pip install Pillow

from adversarialbox.adversary import Adversary
from adversarialbox.attacks.gradient_method import FGSM
from adversarialbox.models.keras import KerasModel

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array,array_to_img
from keras.applications.resnet50 import decode_predictions

from .utils import show_images_diff


import keras

#pip install keras==2.1

def main(modulename,imagename):
    '''
    Kera的应用模块Application提供了带有预训练权重的Keras模型，这些模型可以用来进行预测、特征提取和finetune
    模型的预训练权重将下载到~/.keras/models/并在载入模型时自动载入
    '''

    # 设置为测试模式
    keras.backend.set_learning_phase(0)

    model = ResNet50(weights=modulename)

    img = image.load_img(imagename, target_size=(224, 224))
    original_image = image.img_to_array(img)
    imagedata = np.expand_dims(original_image, axis=0)



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

    attack = FGSM(m)
    #静态epsilon
    attack_config = {"epsilons": 1, "epsilons_max": 10, "epsilon_steps": 1, "steps": 100}

    #y设置为空 会自动计算
    adversary = Adversary(imagedata[:, :, ::-1],None)

    # fgsm non-targeted attack
    adversary = attack(adversary, **attack_config)

    if adversary.is_successful():
        print(
            'attack success, adversarial_label=%d'
            % (adversary.adversarial_label) )

        adversary_image=np.copy(adversary.adversarial_example)
        #强制类型转换 之前是float 现在要转换成uint8

        #BGR -> RGB
        adversary_image=adversary_image[:,:,::-1]

        #adversary_image = np.array(adversary_image).astype("uint8").reshape([224,224,3])
        #original_image=np.array(original_image).astype("uint8").reshape([224, 224, 3])

        adversary_image = np.array(adversary_image).reshape([224,224,3])
        original_image=np.array(original_image).reshape([224, 224, 3])


        show_images_diff(original_image,adversary_image)

    print("FGSM non-target attack done")






if __name__ == '__main__':
    #从'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'下载并解压到当前路径
    #classify_image_graph_def.pb cropped_panda.jpg
    #imagenet2012 中文标签 https://blog.csdn.net/u010165147/article/details/72848497
    main(modulename='imagenet',imagename="cropped_panda.jpg")
