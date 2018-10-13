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

import sys
sys.path.append("..")

import logging
logging.basicConfig(level=logging.INFO,format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger=logging.getLogger(__name__)





from mxnet import gluon
import mxnet as mx
from mxnet.gluon import nn
from mxnet import ndarray as nd
import matplotlib.pyplot as plt
import cv2
from mxnet import image
from mxnet import autograd



from advbox.adversary import Adversary
from advbox.attacks.gradient_method import FGSMT
from advbox.attacks.gradient_method import FGSM
from advbox.models.mxnet import MxNetModel



import numpy as np
import cv2


def main(image_path):

    alexnet = mx.gluon.model_zoo.vision.alexnet(pretrained=True)

    # print(alexnet)

    orig = cv2.imread(image_path)[..., ::-1]
    orig = cv2.resize(orig, (224, 224))
    img = orig.copy().astype(np.float32)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, axis=0)

    #array = mx.nd.array(img)

    # advbox demo
    m = MxNetModel(
        alexnet, None,(-1, 1),
        channel_axis=1)
    attack = FGSMT(m)
    #attack = FGSM(m)

    # 静态epsilons
    attack_config = {"epsilons": 0.2, "epsilon_steps": 1, "steps": 100}

    inputs=img
    #labels=388
    labels = None

    print(inputs.shape)

    adversary = Adversary(inputs, labels)
    #adversary = Adversary(inputs, 388)

    tlabel = 538
    adversary.set_target(is_targeted_attack=True, target_label=tlabel)


    adversary = attack(adversary, **attack_config)


    if adversary.is_successful():
        print(
            'attack success, adversarial_label=%d'
            % (adversary.adversarial_label))

        adv=adversary.adversarial_example[0]
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = adv[..., ::-1]  # RGB to BGR
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        cv2.imwrite('img_adv.png', adv)

    else:
        print('attack failed')


    print("fgsm attack done")


if __name__ == '__main__':
    main("cropped_panda.jpg")
