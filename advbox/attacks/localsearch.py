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
logger=logging.getLogger(__name__)

import numpy as np


from .base import Attack

__all__ = [
    'SinglePixelAttack','LocalSearchAttack'
]

#Simple Black-Box Adversarial Perturbations for Deep Networks
#随机在图像中选择max_pixels个点 在多个信道中同时进行修改，修改范围通常为0-255
class SinglePixelAttack(Attack):

    def __init__(self, model, support_targeted=True):
        super(SinglePixelAttack, self).__init__(model)
        self.support_targeted = support_targeted

    #如果输入的原始数据，isPreprocessed为False，如果驶入的图像数据被归一化了，设置为True
    def _apply(self,adversary,max_pixels=1000,isPreprocessed=False):

        if not self.support_targeted:
            if adversary.is_targeted_attack:
                raise ValueError(
                    "This attack method doesn't support targeted attack!")

        min_, max_ = self.model.bounds()


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

        #输入的图像必须具有长和宽属性
        assert len(axes) == 2

        h = adv_img.shape[axes[0]]
        w = adv_img.shape[axes[1]]

        #print("w={0},h={1}".format(w,h))

        #max_pixel为攻击点的最多个数 从原始图像中随机选择max_pixel个进行攻击

        pixels = np.random.permutation(h * w)
        pixels = pixels[:max_pixels]

        for i, pixel in enumerate(pixels):
            x = pixel % w
            y = pixel // w

            location = [x, y]

            logging.info("Attack x={0} y={1}".format(x,y))

            location.insert(self.model.channel_axis(), slice(None))
            location = tuple(location)


            if not isPreprocessed:
                #logger.info("value in [min_={0}, max_={1}]".format(min_, max_))
                #图像没有经过预处理 取值为整数 范围为0-255
                for value in [min_, max_]:
                    perturbed = np.copy(adv_img)
                    #针对图像的每个信道的点[x,y]同时进行修改
                    perturbed[location] = value

                    f = self.model.predict(perturbed)
                    adv_label = np.argmax(f)

                    if adversary.try_accept_the_example(adv_img, adv_label):
                        return adversary
            else:
                # 图像经过预处理 取值为整数 通常范围为0-1
                for value in np.linspace(min_, max_, num=256):
                    #logger.info("value in [min_={0}, max_={1},step num=256]".format(min_, max_))
                    perturbed = np.copy(adv_img)
                    #针对图像的每个信道的点[x,y]同时进行修改
                    perturbed[location] = value

                    f = self.model.predict(perturbed)
                    adv_label = np.argmax(f)

                    if adversary.try_accept_the_example(adv_img, adv_label):
                        return adversary

        return adversary
