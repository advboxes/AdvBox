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
攻击对象caffe下训练的lenet，下载地址为：
https://github.com/ethereon/caffe-tensorflow
使用转换工具caffe2fluid将caffe下的基于mnist训练的模型lenet转换成模型文件lenet.py 参数文件lenet.npy
保存在tutorials/fluid/lenet目录下
https://github.com/PaddlePaddle/models/blob/e7684f07505c172beb4c4d9febb4a48f9fa83b68/fluid/image_classification/caffe2fluid/README.md
其中caffe2fluid自动生成了lenet的代码文件lenet.py
数据集mnist
攻击算法FGSM
set env variable before using converted model if used custom_layers:
export CAFFE2FLUID_CUSTOM_LAYERS=/mnt/advbox-demo/caffe2fluid/kaffe


"""
import sys
import os
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid
import paddle.v2 as paddle

from advbox.adversary import Adversary
from advbox.attacks.gradient_method import FGSM
from advbox.attacks.gradient_method import FGSMT
from advbox.models.paddle import PaddleModel
from image_classification.lenet import LeNet

#通过设置环境变量WITH_GPU 来动态设置是否使用GPU资源 特别适合在mac上开发但是在GPU服务器上运行的情况
#比如在mac上不设置该环境变量，在GPU服务器上设置 export WITH_GPU=1
with_gpu = os.getenv('WITH_GPU', '0') != '0'


def main(use_cuda):
    """
    Advbox demo which demonstrate how to use advbox.
    """
    TOTAL_NUM = 500
    IMG_NAME = 'image'
    LABEL_NAME = 'label'

    weight_file="fluid/lenet/lenet.npy"

    #1, define network topology
    images = fluid.layers.data(name=IMG_NAME, shape=[1, 28, 28], dtype='float32')
    # gradient should flow
    images.stop_gradient = False
    label = fluid.layers.data(name=LABEL_NAME, shape=[1], dtype='int64')

    net = LeNet({'data': images})
    prediction = net.layers['prob']

    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)


    #根据配置选择使用CPU资源还是GPU资源
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)
    #这句很关键 没有的话会报错
    # AttributeError: 'NoneType' object has no attribute 'get_tensor'
    exe.run(fluid.default_startup_program())

    #加载参数
    net.load(data_path=weight_file, exe=exe, place=place)

    BATCH_SIZE = 1


    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.test(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)


    # advbox demo
    m = PaddleModel(
        fluid.default_main_program(),
        IMG_NAME,
        LABEL_NAME,
        prediction.name,
        avg_cost.name, (-1, 1),
        channel_axis=1)
    attack = FGSM(m)
    # attack = FGSMT(m)
    attack_config = {"epsilons": 0.3}

    # use test data to generate adversarial examples
    total_count = 0
    fooling_count = 0
    for data in test_reader():
        total_count += 1
        adversary = Adversary(data[0][0], data[0][1])

        # FGSM non-targeted attack
        adversary = attack(adversary, **attack_config)

        if adversary.is_successful():
            fooling_count += 1
            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                % (data[0][1], adversary.adversarial_label, total_count))

        else:
            print('attack failed, original_label=%d, count=%d' %
                  (data[0][1], total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break
    print("fgsm attack done")


if __name__ == '__main__':
    main(use_cuda=with_gpu)
