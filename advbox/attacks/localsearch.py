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


#这部分代码参考了foolbox的实现 移植到了paddle平台，原地址为：
#https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/localsearch.py

#Simple Black-Box Adversarial Perturbations for Deep Networks 函数命名也完全和论文一致
# perturbation factor p 扰动系数
# two perturbation parameters p ∈ R and r ∈ [0,2],
# a budget U ∈ N on the number of trials
# the half side length of the neighborhood square d ∈ N,
# the number of pixels perturbed at each round t ∈ N,
# and an upper bound on the number of rounds R ∈ N.
#
class LocalSearchAttack(Attack):

    def __init__(self, model, support_targeted=True):
        super(LocalSearchAttack, self).__init__(model)
        self.support_targeted = support_targeted


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _apply(self,adversary,r=1.5, p=10., d=5, t=5, R=150):

        if not self.support_targeted:
            if adversary.is_targeted_attack:
                raise ValueError(
                    "This attack method doesn't support targeted attack!")

        #r的范围
        assert 0 <= r <= 2

        min_, max_ = self.model.bounds()

        logger.info("LocalSearchAttack parameter:min_={}, max_={} ".format(min_, max_ ))

        # 强制拷贝 避免针对adv_img的修改也影响adversary.original
        adv_img = np.copy(adversary.original)

        logger.info("adv_img:{}".format(adv_img))

        original_label=adversary.original_label

        logger.info("LocalSearchAttack parameter:r={0}, p={1}, d={2}, t={3}, R={4}".format(r, p, d, t, R))

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

        #正则化到[-0.5,0.5]区间内
        def normalize(im):

            im = im -(min_ + max_) / 2
            im = im / (max_ - min_)

            LB = -1 / 2
            UB = 1 / 2
            return im, LB, UB

        def unnormalize(im):

            im = im * (max_ - min_)
            im = im + (min_ + max_) / 2
            return im

        #归一化
        adv_img, LB, UB = normalize(adv_img)

        logger.info("normalize adv_img:{}".format(adv_img))


        channels = adv_img.shape[self.model.channel_axis()]

        #随机选择一部分像素点 总数不超过全部的10% 最大为128个点
        def random_locations():
            n = int(0.1 * h * w)
            n = min(n, 128)
            locations = np.random.permutation(h * w)[:n]
            p_x = locations % w
            p_y = locations // w
            pxy = list(zip(p_x, p_y))
            pxy = np.array(pxy)
            return pxy

        #针对图像的每个信道的点[x,y]同时进行修改 修改的值为p * np.sign(Im[location]) 类似FGSM的一次迭代
        #不修改Ii的图像 返回修改后的图像
        def pert(Ii, p, x, y):
            Im = Ii.copy()
            location = [x, y]
            location.insert(self.model.channel_axis(), slice(None))
            location = tuple(location)
            Im[location] = p * np.sign(Im[location])
            return Im

        #截断 确保assert LB <= r * Ibxy <= UB 但是也有可能阶段失败退出 因此可以适当扩大配置的原始数据范围
        # 这块的实现没有完全参考论文
        def cyclic(r, Ibxy):

            #logger.info("cyclic Ibxy:{}".format(Ibxy))
            result = r * Ibxy
            #logger.info("cyclic result:{}".format(result))

            """
            foolbox的实现 存在极端情况  本来只是有一个元素超过UB，结果一减 都完蛋了
            if result.any() < LB:
                #result = result/r + (UB - LB)
                logger.info("cyclic result:{}".format(result))
                result = result + (UB - LB)
                logger.info("cyclic result:{}".format(result))
                #result=LB
            elif result.any()  > UB:
                #result = result/r - (UB - LB)
                logger.info("cyclic result:{}".format(result))
                result = result - (UB - LB)
                logger.info("cyclic result:{}".format(result))
                #result=UB
            """

            if result.any() < LB:
                result = result + (UB - LB)
            elif result.any() > UB:
                result = result - (UB - LB)



            result=result.clip(LB,UB)

            #logger.info("cyclic result:{}".format(result))

            #assert LB <= np.all(result) <= UB

            return result

        Ii = adv_img
        PxPy = random_locations()

        #循环攻击轮
        for try_time in range(R):
            #重新排序 随机选择不不超过128个点
            PxPy = PxPy[np.random.permutation(len(PxPy))[:128]]
            L = [pert(Ii, p, x, y) for x, y in PxPy]

            #批量返回预测结果 获取原始图像标签的概率
            def score(Its):
                Its = np.stack(Its)
                Its = unnormalize(Its)
                """
                original_label为原始图像的标签
                """
                scores=[ self.model.predict(unnormalize(Ii))[original_label] for It in Its ]

                return scores


            #选择影响力最大的t个点进行扰动 抓主要矛盾 这里需要注意 np.argsort是升序 所以要取倒数的几个
            scores = score(L)

            indices = np.argsort(scores)[:-t]
            logger.info("try {0} times  selected pixel indices:{1}".format(try_time,str(indices)))

            PxPy_star = PxPy[indices]

            for x, y in PxPy_star:
                #每个颜色通道的[x，y]点进行扰动并截断 扰动算法即放大r倍
                for b in range(channels):
                    location = [x, y]
                    location.insert(self.model.channel_axis(), b)
                    location = tuple(location)
                    Ii[location] = cyclic(r, Ii[location])

            f = self.model.predict(unnormalize(Ii))
            adv_label = np.argmax(f)
            adv_label_pro=self.softmax(f)[adv_label]
            logger.info("adv_label={0} adv_label_pro={1}".format(adv_label,adv_label_pro))
            # print("adv_label={0}".format(adv_label))
            if adversary.try_accept_the_example(unnormalize(Ii), adv_label):

                return adversary

            #扩大搜索范围，把原有点周围2d乘以2d范围内的点都拉进来 去掉超过【w，h】的点
            #"{Update a neighborhood of pixel locations for the next round}"

            PxPy = [
                (x, y)
                for _a, _b in PxPy_star
                for x in range(_a - d, _a + d + 1)
                for y in range(_b - d, _b + d + 1)]
            PxPy = [(x, y) for x, y in PxPy if 0 <= x < w and 0 <= y < h]
            PxPy = list(set(PxPy))
            PxPy = np.array(PxPy)

        return adversary