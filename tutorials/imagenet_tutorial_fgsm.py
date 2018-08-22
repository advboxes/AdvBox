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

#使用FGSM攻击resnet 数据集为cifar10

from __future__ import print_function

import sys



import os
import numpy as np
import math
import time

sys.path.append("..")

import paddle.fluid as fluid
import paddle.v2 as paddle
from PIL import Image

import resnet

from advbox.adversary import Adversary
from advbox.attacks.gradient_method import FGSM
from advbox.models.paddle import PaddleModel


%matplotlib inline

import matplotlib.pyplot as plt

#from resnet import resnet_cifar10


#通过设置环境变量WITH_GPU 来动态设置是否使用GPU资源 特别适合在mac上开发但是在GPU服务器上运行的情况
#比如在mac上不设置该环境变量，在GPU服务器上设置 export WITH_GPU=1
with_gpu = os.getenv('WITH_GPU', '0') != '0'


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
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img

def get_image(image_file):
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = Image.open(image_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=224, center=True)
    img = np.array(img).astype("float32").transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std
    #img=img[np.newaxis, :]
    return img


# 展示效果
def show(image,adversarial):
    plt.figure()

    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial')
    plt.imshow(adversarial[:, :, ::-1] / 255)  # ::-1 to convert BGR to RGB
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference')
    difference = adversarial[:, :, ::-1] - image
    plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
    plt.axis('off')

    plt.show()


def main(use_cuda):
    """
    Advbox demo which demonstrate how to use advbox.
    """
    class_dim = 1000
    IMG_NAME = 'img'
    LABEL_NAME = 'label'
    #模型路径
    pretrained_model = "models/resnet_50/115"
    with_memory_optimization = 1
    image_shape = [3,224,224]

    image = fluid.layers.data(name=IMG_NAME, shape=image_shape, dtype='float32')
    # gradient should flow
    image.stop_gradient = False
    label = fluid.layers.data(name=LABEL_NAME, shape=[1], dtype='int64')

    # model definition

    model = resnet.ResNet50()

    out = model.net(input=image, class_dim=class_dim)

    #test_program = fluid.default_main_program().clone(for_test=True)

    if with_memory_optimization:
        fluid.memory_optimize(fluid.default_main_program())

    # 根据配置选择使用CPU资源还是GPU资源
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    #加载模型参数
    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)


    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)


    # advbox demo
    m = PaddleModel(
        fluid.default_main_program(),
        IMG_NAME,
        LABEL_NAME,
        out.name,
        avg_cost.name, (-1, 1),
        channel_axis=3)
    attack = FGSM(m)

    attack_config = {"epsilons": 0.3}

    test_data = get_image("cat.jpg")
    # 猫对应的标签
    test_label = 285

    adversary = Adversary(test_data, test_label)

    # FGSM non-targeted attack
    adversary = attack(adversary, **attack_config)

    if adversary.is_successful():

        print(
                'attack success, original_label=%d, adversarial_label=%d'
                % (test_label, adversary.adversarial_label))

    else:
        print('attack failed, original_label=%d, ' % (test_label))

    print("fgsm attack done")


if __name__ == '__main__':
    main(use_cuda=with_gpu)
