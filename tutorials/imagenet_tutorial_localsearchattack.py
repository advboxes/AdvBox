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

#使用LocalSearchAttack攻击AlexNet 数据集为imagenet2012

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from past.utils import old_div
import sys


import logging
logging.basicConfig(level=logging.INFO,format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger=logging.getLogger(__name__)


import os
import numpy as np
import math
import time

sys.path.append("..")

import paddle.fluid as fluid
try:
    import paddle.v2 as paddle
except ModuleNotFoundError as e:
    import paddle

from PIL import Image


from adversarialbox.adversary import Adversary
from adversarialbox.attacks.localsearch import LocalSearchAttack

from image_classification.alexnet import AlexNet

#from adversarialbox.models.paddle import PaddleModel
from adversarialbox.models.paddleBlackBox import PaddleBlackBoxModel



#通过设置环境变量WITH_GPU 来动态设置是否使用GPU资源 特别适合在mac上开发但是在GPU服务器上运行的情况
#比如在mac上不设置该环境变量，在GPU服务器上设置 export WITH_GPU=1
with_gpu = os.getenv('WITH_GPU', '0') != '0'

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

#图像预处理
def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = old_div((width - size), 2)
        h_start = old_div((height - size), 2)
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img

def get_image(image_file):

    img = Image.open(image_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=224, center=True)
    img = old_div(np.array(img).astype("float32").transpose((2, 0, 1)), 255)

    #imagenet数据训练时进行了标准化，强烈建议图像预处理时也进行预处理
    img -= img_mean
    img /= img_std
    #img=img[np.newaxis, :]
    return img

def main(use_cuda):

    """
    Advbox demo which demonstrate how to use advbox.
    """
    class_dim = 1000
    IMG_NAME = 'img'
    LABEL_NAME = 'label'
    #模型路径 http://paddle-imagenet-models.bj.bcebos.com/resnet_50_model.tar 下载并解压
    #pretrained_model = "models/resnet_50/115"
    pretrained_model = "models/alexnet/116/"
    image_shape = [3,224,224]

    image = fluid.layers.data(name=IMG_NAME, shape=image_shape, dtype='float32')
    label = fluid.layers.data(name=LABEL_NAME, shape=[1], dtype='int64')

    # model definition

    model = AlexNet()

    out = model.net(input=image, class_dim=class_dim)

    # 根据配置选择使用CPU资源还是GPU资源
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    #加载模型参数
    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        logger.info("Load pretrained_model")
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)


    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)


    logging.info("Build advbox")
    # advbox demo 黑盒攻击 直接传入测试版本的program
    m = PaddleBlackBoxModel(
        fluid.default_main_program().clone(for_test=True),
        IMG_NAME,
        LABEL_NAME,
        out.name, (-1, 1),
        channel_axis=0)

    #不定向攻击
    # 形状为[1,28,28] channel_axis=0  形状为[28,28,1] channel_axis=2
    attack = LocalSearchAttack(m)

    attack_config = {"R": 200,"r":1.0}

    test_data = get_image("cat.png")
    original_data=np.copy(test_data)
    # 猫对应的标签 imagenet 2012 对应链接https://blog.csdn.net/LegenDavid/article/details/73335578
    original_label = None
    adversary = Adversary(original_data, original_label)

    logger.info("Non-targeted Attack...")
    adversary = attack(adversary, **attack_config)

    if adversary.is_successful():

        print(
                'attack success, original_label=%d, adversarial_label=%d'
                % (adversary.original_label, adversary.adversarial_label))

        #对抗样本保存在adversary.adversarial_example
        adversary_image=np.copy(adversary.adversarial_example)

        #从[3,224,224]转换成[224,224，3]
        adversary_image*=img_std
        adversary_image+=img_mean

        adversary_image = np.array(adversary_image * 255).astype("uint8").transpose([1, 2, 0])

        im = Image.fromarray(adversary_image)
        im.save("adversary_image.jpg")


    else:
        print('attack failed, original_label=%d' % (adversary.original_label))

    logger.info("LocalSearchAttack attack done")



if __name__ == '__main__':
    main(use_cuda=with_gpu)
