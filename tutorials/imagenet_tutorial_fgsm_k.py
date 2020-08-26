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
from adversarialbox.attacks.gradient_method import FGSM
from adversarialbox.attacks.gradient_method import FGSMT
from adversarialbox.models.keras import KerasModel

from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array,array_to_img
#from keras.applications.resnet50 import decode_predictions


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
    #model = InceptionV3(weights=modulename)

    logging.info(model.summary())

    img = image.load_img(imagename, target_size=(224, 224))
    raw_imagedata = image.img_to_array(img)
    raw_imagedata = np.expand_dims(raw_imagedata, axis=0)

    # 'RGB'->'BGR'
    imagedata = raw_imagedata[:, :, :, ::-1]

    #logging.info(raw_imagedata)
    #logging.info(imagedata)

    #logit fc1000
    logits=model.get_layer('fc1000').output

    #keras中获取指定层的方法为：
    #base_model.get_layer('block4_pool').output)
    # advbox demo
    # 因为原始数据没有归一化  所以bounds=(0, 255)  KerasMode内部在进行预测和计算梯度时会进行预处理
    # imagenet数据集归一化时 标准差为1  mean为[104, 116, 123]
    # featurefqueezing_bit_depth featurefqueezing防御算法 提高生成攻击样本的质量 为特征数据的bit位 一般8就ok了
    m = KerasModel(
        model,
        model.input,
        None,
        logits,
        None,
        bounds=(0, 255.0),
        channel_axis=3,
        preprocess=([104, 116, 123],1),
        featurefqueezing_bit_depth=8)

    attack = FGSM(m)
    #设置epsilons时不用考虑特征范围 算法实现时已经考虑了取值范围的问题 epsilons取值范围为（0，1）
    #epsilon支持动态调整 epsilon_steps为epsilon变化的个数
    #epsilons为下限 epsilons_max为上限
    #attack_config = {"epsilons": 0.3, "epsilons_max": 0.5, "epsilon_steps": 100}
    #静态epsilons
    attack_config = {"epsilons": 1, "epsilons_max": 10, "epsilon_steps": 1,"steps":100}

    #y设置为空 会自动计算
    adversary = Adversary(imagedata.copy(),None)

    # FGSM non-targeted attack
    adversary = attack(adversary, **attack_config)

    if adversary.is_successful():
        print(
            'attack success, adversarial_label=%d'
            % (adversary.adversarial_label) )

        #对抗样本保存在adversary.adversarial_example
        adversary_image=np.copy(adversary.adversarial_example)

        logging.info("adversary_image label={0} ".format(np.argmax(m.predict(adversary_image)))  )
        #logging.info(adversary_image)

        #强制类型转换 之前是float 现在要转换成uint8
        adversary_image = np.array(adversary_image).astype("uint8").reshape([224,224,3])

        #logging.info(adversary_image)
        adversary_image=adversary_image[:, :, ::-1]
        logging.info(adversary_image-raw_imagedata)

        img=array_to_img(adversary_image)
        img.save('adversary_image_nontarget.jpg')

    print("fgsm non-target attack done")



    attack = FGSMT(m)
    #静态epsilons
    attack_config = {"epsilons": 20, "epsilons_max": 20, "epsilon_steps": 1,"steps":100}

    adversary = Adversary(imagedata,None)

    tlabel = 489
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

        adversary_image = np.array(adversary_image).astype("uint8").reshape([224,224,3])

        adversary_image=adversary_image[:, :, ::-1]
        logging.info(adversary_image - raw_imagedata)

        img=array_to_img(adversary_image)
        img.save('adversary_image_target.jpg')

    print("fgsm target attack done")



if __name__ == '__main__':
    #从'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'下载并解压到当前路径
    #classify_image_graph_def.pb cropped_panda.jpg
    #imagenet2012 中文标签 https://blog.csdn.net/u010165147/article/details/72848497
    main(modulename='imagenet',imagename="cropped_panda.jpg")
