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
        super(SinglePixelAttack, self).__init__(model)
        self.support_targeted = support_targeted

    def _apply(self,adversary,max_pixels=1000):

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

        if len(axes) == 2:
            h = adv_img.shape[axes[0]]
            w = adv_img.shape[axes[1]]
        else:
            h=adv_img.shape[-1]
            w=1

        #print("w={0},h={1}".format(w,h))

        #max_pixel为攻击点的最多个数 从原始图像中随机选择max_pixel个进行攻击

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
                #print("adv_label={0}".format(adv_label))
                if adversary.try_accept_the_example(adv_img, adv_label):
                    return adversary


        return adversary


class LocalSearchAttack(Attack):

    def __init__(self, model, support_targeted=True):
        super(LocalSearchAttack, self).__init__(model)
        self.support_targeted = support_targeted

    def _apply(self,adversary,r=1.5, p=10., d=5, t=5, R=150):

        if not self.support_targeted:
            if adversary.is_targeted_attack:
                raise ValueError(
                    "This attack method doesn't support targeted attack!")

        #r的范围
        assert 0 <= r <= 2


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

        if len(axes) == 2:
            h = adv_img.shape[axes[0]]
            w = adv_img.shape[axes[1]]
        else:
            h=adv_img.shape[-1]
            w=1

        #print("w={0},h={1}".format(w,h))

        def normalize(im):
            min_, max_ = self.model.bounds()

            im = im - (min_ + max_) / 2
            im = im / (max_ - min_)

            LB = -1 / 2
            UB = 1 / 2
            return im, LB, UB

        def unnormalize(im):
            min_, max_ = self.model.bounds()

            im = im * (max_ - min_)
            im = im + (min_ + max_) / 2
            return im

        #归一化
        adv_img, LB, UB = normalize(adv_img)
        channels = adv_img.shape[self.model.channel_axis()]

        def random_locations():
            n = int(0.1 * h * w)
            n = min(n, 128)
            locations = np.random.permutation(h * w)[:n]
            p_x = locations % w
            p_y = locations // w
            pxy = list(zip(p_x, p_y))
            pxy = np.array(pxy)
            return pxy

        def pert(Ii, p, x, y):
            Im = Ii.copy()
            location = [x, y]
            location.insert(self.model.channel_axis(), slice(None))
            location = tuple(location)
            Im[location] = p * np.sign(Im[location])
            return Im

        def cyclic(r, Ibxy):
            result = r * Ibxy
            if result < LB:
                result = result + (UB - LB)
            elif result > UB:
                result = result - (UB - LB)
            assert LB <= result <= UB
            return result

        Ii = adv_img
        PxPy = random_locations()

        for _ in range(R):
            PxPy = PxPy[np.random.permutation(len(PxPy))[:128]]
            L = [pert(Ii, p, x, y) for x, y in PxPy]

            def score(Its):
                Its = np.stack(Its)
                Its = unnormalize(Its)
                batch_logits, _ = a.batch_predictions(Its, strict=False)
                scores = [softmax(logits)[cI] for logits in batch_logits]
                return scores

            scores = score(L)

            indices = np.argsort(scores)[:t]

            PxPy_star = PxPy[indices]

            for x, y in PxPy_star:
                for b in range(channels):
                    location = [x, y]
                    location.insert(self.model.channel_axis(), b)
                    location = tuple(location)
                    Ii[location] = cyclic(r, Ii[location])

            f = self.model.predict(unnormalize(Ii))
            adv_label = np.argmax(f)
            # print("adv_label={0}".format(adv_label))
            if adversary.try_accept_the_example(adv_img, adv_label):
                return adversary

            PxPy = [
                (x, y)
                for _a, _b in PxPy_star
                for x in range(_a - d, _a + d + 1)
                for y in range(_b - d, _b + d + 1)]
            PxPy = [(x, y) for x, y in PxPy if 0 <= x < w and 0 <= y < h]
            PxPy = list(set(PxPy))
            PxPy = np.array(PxPy)

        return adversary