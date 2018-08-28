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
This module provide the attack method for SinglePixelAttack & LocalSearchAttack's implement.
"""
from __future__ import division

import logging
from collections import Iterable

import numpy as np


from .base import Attack

__all__ = [
    'SinglePixelAttack','LocalSearchAttack'
]


class SinglePixelAttack(Attack):


    def __init__(self, model, support_targeted=True):

        self.support_targeted = support_targeted

    def _apply(self,
               adversary):

        if not self.support_targeted:
            if adversary.is_targeted_attack:
                raise ValueError(
                    "This attack method doesn't support targeted attack!")

        pre_label = adversary.original_label
        min_, max_ = self.model.bounds()

        assert self.model.channel_axis() == adversary.original.ndim
        assert (self.model.channel_axis() == 1 or
                self.model.channel_axis() == adversary.original.shape[0] or
                self.model.channel_axis() == adversary.original.shape[-1])

        # 强制拷贝 避免针对adv_img的修改也影响adversary.original
        adv_img = np.copy(adversary.original)

        '''
        adversary.original  原始数据
        adversary.original_label  原始数据的标签
        adversary.target_label       定向攻击的目标值
        adversary.__adversarial_example 保存生成的对抗样本
        adversary.adversarial_label  对抗样本的标签
        '''

        axes = [i for i in range(adversary.original.ndim) if i != self.model.channel_axis()]
        assert len(axes) == 2
        h = adv_img.shape[axes[0]]
        w = adv_img.shape[axes[1]]

        #攻击点的最多个数 目前先硬编码
        max_pixels = 28*28

        pixels = np.random.permutation(h * w)
        pixels = pixels[:max_pixels]

        for i, pixel in enumerate(pixels):
            x = pixel % w
            y = pixel // w

            location = [x, y]
            location.insert(self.model.channel_axis(), slice(None))
            location = tuple(location)

            for value in [min_, max_]:
                perturbed = np.copy(adv_img)
                perturbed[location] = value

                f = self.model.predict(perturbed)
                adv_label = np.argmax(f)

                if adversary.try_accept_the_example(adv_img, adv_label):
                    return adversary




        return adversary


