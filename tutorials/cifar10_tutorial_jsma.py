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

#使用JSMA攻击deepfool 数据集为cifar10

from __future__ import print_function
from __future__ import absolute_import
import sys
import os

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid
try:
    import paddle.v2 as paddle
except ModuleNotFoundError as e:
    import paddle

from adversarialbox.adversary import Adversary

from adversarialbox.attacks.saliency import JSMA
from adversarialbox.models.paddle import PaddleModel

from image_classification.resnet import resnet_cifar10

#通过设置环境变量WITH_GPU 来动态设置是否使用GPU资源 特别适合在mac上开发但是在GPU服务器上运行的情况
#比如在mac上不设置该环境变量，在GPU服务器上设置 export WITH_GPU=1
with_gpu = os.getenv('WITH_GPU', '0') != '0'


def main(use_cuda):
    """
    Advbox demo which demonstrate how to use advbox.
    """
    TOTAL_NUM = 500
    IMG_NAME = 'img'
    LABEL_NAME = 'label'

    img = fluid.layers.data(name=IMG_NAME, shape=[3, 32, 32], dtype='float32')
    # gradient should flow
    img.stop_gradient = False
    label = fluid.layers.data(name=LABEL_NAME, shape=[1], dtype='int64')

    # logits = mnist_cnn_model(img)
    # logits = vgg_bn_drop(img)
    logits = resnet_cifar10(img, 32)

    cost = fluid.layers.cross_entropy(input=logits, label=label)
    avg_cost = fluid.layers.mean(x=cost)


    #根据配置选择使用CPU资源还是GPU资源
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    BATCH_SIZE = 1
    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)

    fluid.io.load_params(
        exe, "cifar10/resnet", main_program=fluid.default_main_program())

    # advbox demo
    m = PaddleModel(
        fluid.default_main_program(),
        IMG_NAME,
        LABEL_NAME,
        logits.name,
        avg_cost.name, (-1, 1),
        channel_axis=1)
    # attack = FGSM(m)
    attack = JSMA(m)
    # attack = FGSMT(m)
    # attack_config = {"epsilons": 0.3}
    attack_config = {
        "max_iter": 2000,
        "theta": 0.1,
        "max_perturbations_per_pixel": 7
    }
    # use test data to generate adversarial examples
    total_count = 0
    fooling_count = 0
    for data in test_reader():
        total_count += 1
        adversary = Adversary(data[0][0], data[0][1])

        # FGSM non-targeted attack
        adversary = attack(adversary, **attack_config)

        # FGSMT targeted attack
        # tlabel = 0
        # adversary.set_target(is_targeted_attack=True, target_label=tlabel)
        # adversary = attack(adversary, **attack_config)

        if adversary.is_successful():
            fooling_count += 1
            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                % (data[0][1], adversary.adversarial_label, total_count))
            # plt.imshow(adversary.target, cmap='Greys_r')
            # plt.show()
            # np.save('adv_img', adversary.target)
        else:
            print('attack failed, original_label=%d, count=%d' %
                  (data[0][1], total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break
    # print("fgsm attack done")
    print("jsma attack done")


if __name__ == '__main__':
    main(use_cuda=with_gpu)
