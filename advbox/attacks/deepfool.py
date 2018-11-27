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
This module provide the attack method for deepfool. Deepfool is a simple and
accurate adversarial attack.
"""
from __future__ import division

import logging

import numpy as np

from .base import Attack

__all__ = ['DeepFoolAttack']


class DeepFoolAttack(Attack):
    """
    DeepFool: a simple and accurate method to fool deep neural networks",
    Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
    https://arxiv.org/abs/1511.04599
    """

    def _apply(self, adversary, iterations=100, overshoot=0.02):
        """
          Apply the deep fool attack.

          Args:
              adversary(Adversary): The Adversary object.
              iterations(int): The iterations.
              overshoot(float): We add (1+overshoot)*pert every iteration.
          Return:
              adversary: The Adversary object.
          """
        assert adversary is not None

        pre_label = adversary.original_label
        min_, max_ = self.model.bounds()

        logging.info('min_={0}, max_={1}'.format(min_,max_))

        f = self.model.predict(adversary.original)

        if adversary.is_targeted_attack:
            labels = [adversary.target_label]
        else:
            max_class_count = 10
            class_count = self.model.num_classes()
            if class_count > max_class_count:
                #labels = np.argsort(f)[-(max_class_count + 1):-1]
                logging.info('selec top-{0} class'.format(max_class_count))
                labels= np.argsort(f)[::-1]
                labels = labels[:max_class_count]
            else:
                labels = np.arange(class_count)

        t= [ str(i) for i in labels]

        logging.info('select label:'+" ".join(t))

        gradient = self.model.gradient(adversary.original, pre_label)
        x = np.copy(adversary.original)

        for iteration in xrange(iterations):
            w = np.inf
            w_norm = np.inf
            pert = np.inf
            for k in labels:
                if k == pre_label:
                    continue
                #标签k对应的梯度
                gradient_k = self.model.gradient(x, k)
                #标签k对应的梯度和现有便签的梯度的差
                w_k = gradient_k - gradient
                #标签k和现有标签在logit层面的差距
                f_k = f[k] - f[pre_label]
                #保证非0 避免计算时除以0 梯度差距除以l2范式值
                w_k_norm = np.linalg.norm(w_k.flatten()) + 1e-8
                #标签k和现有标签在logit层面的差距除以l2范式的值
                pert_k = (np.abs(f_k) + 1e-8) / w_k_norm

                #对应论文中的选择最小的l
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
                    w_norm = w_k_norm
                    #logging.info("k={0} pert={1} w_norm={2}".format(k,pert, w_norm))

            #论文中为+  advbox老版本实现成了- l2实现
            r_i = w * pert / w_norm

            #logging.info(r_i)

            # 放大系数 在原论文上的创新 提高攻击速度 论文中相当于overshoot=0
            x = x + (1 + overshoot) *r_i

            x = np.clip(x, min_, max_)

            f = self.model.predict(x)
            gradient = self.model.gradient(x, pre_label)
            adv_label = np.argmax(f)
            logging.info('iteration={}, f[pre_label]={}, f[target_label]={}'
                         ', f[adv_label]={}, pre_label={}, adv_label={}'
                         ''.format(iteration, f[pre_label], (
                             f[adversary.target_label]
                             if adversary.is_targeted_attack else 'NaN'), f[
                                 adv_label], pre_label, adv_label))
            if adversary.try_accept_the_example(x, adv_label):
                return adversary

        return adversary
