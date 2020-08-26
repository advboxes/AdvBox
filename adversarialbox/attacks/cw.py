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
This module provide the attack method of "CW".
L2 distance metrics especially
"""
from __future__ import division
from __future__ import print_function

from builtins import range
import logging
import numpy as np

import paddle.fluid as fluid
from .base import Attack

__all__ = ['CW_L2_Attack', 'CW_L2']


class CW_L2_Attack(Attack):
    """
    Uses Adam to minimize the CW L2 objective function

    Paper link: https://arxiv.org/abs/1608.04644
    """
    def __init__(self, model, learning_rate, attack_model, with_gpu=False,
                 shape=[1, 28, 28], dim=10, confidence_level=0, multi_clip=False):
        super(CW_L2_Attack, self).__init__(model)
        self._predicts_normalized = None
        self._adversary = None  # type: Adversary
        #########################################
        # build cw attack computation graph
        # use CPU or GPU
        self.place  = fluid.CUDAPlace(0) if with_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)
        # clone the prebuilt program that has cnn to attack
        self.attack_main_program = fluid.Program()
        self.attack_startup_program = fluid.Program()
        # init CW Graph parameters
        self._attack_model = attack_model
        self._shape = shape
        self._dim = dim
        self.confidence = confidence_level
        # compute clip min and max
        if multi_clip:
            self.clip_shape = [3, 1, 1]
            img_mean = np.array([0.485, 0.456, 0.406]).reshape(self.clip_shape)
            img_std = np.array([0.229, 0.224, 0.225]).reshape(self.clip_shape)
            self.pa_clip_min = (np.zeros(self.clip_shape) - img_mean) / img_std
            self.pa_clip_max = (np.ones(self.clip_shape) - img_mean) / img_std
        else:
            clip_min, clip_max = self.model._bounds
            self.clip_shape = [1, 1, 1]
            self.pa_clip_min = np.array(clip_min).reshape(self.clip_shape)
            self.pa_clip_max = np.array(clip_max).reshape(self.clip_shape)

        # build cw attack compute graph within attack programs
        with fluid.program_guard(main_program=self.attack_main_program, startup_program=self.attack_startup_program):
            img_0_1_placehold = fluid.layers.data(name='img_data_scaled', shape=self._shape, dtype="float32")
            target_placehold = fluid.layers.data(name='target', shape=[self._dim], dtype="float32")
            c_placehold = fluid.layers.data(name='c', shape=[1], dtype="float32")
            # add this perturbation
            self.ad_perturbation = fluid.layers.create_parameter(name='parameter',
                                                                shape=self._shape,
                                                                dtype='float32',
                                                                is_bias=False)
            # add clip_min and clip_max for normalization
            self.clip_min = fluid.layers.create_parameter(name='clip_min',
                                                                shape=self.clip_shape,
                                                                dtype='float32',
                                                                is_bias=False)
            self.clip_max = fluid.layers.create_parameter(name='clip_max',
                                                                shape=self.clip_shape,
                                                                dtype='float32',
                                                                is_bias=False)

            # construct graph with perturbation and cnn model
            constrained, dis_L2 = self._constrain_cwb(img_0_1_placehold)
            loss, _, _ = self._loss_cwb(target_placehold, constrained, dis_L2, c_placehold)

            # Adam optimizer as suggested in paper
            optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
            optimizer.minimize(loss, parameter_list=['parameter'])

        # initial variables and parameters every time before attack
        self.exe.run(self.attack_startup_program)

        ad_min = fluid.global_scope().find_var("clip_min").get_tensor()
        ad_min.set(self.pa_clip_min.astype('float32'), self.place)
        ad_max = fluid.global_scope().find_var("clip_max").get_tensor()
        ad_max.set(self.pa_clip_max.astype('float32'), self.place)
        #########################################

    def _apply(self,
               adversary,
               attack_iterations=100,
               c_search_step = 20,
               c_range=(0.01,100),
               c_start=10,
               c_accuracy=0.1,
               targeted=True):
        """
        put adversary instance inside of the attack instance so all other function within can access
        """
        if not adversary.is_targeted_attack:
            raise ValueError("This attack method only support targeted attack!")

        self._adversary = adversary
        img = self._adversary.original  # original image to be attacked
        # print original predict result
        print('guess img before preprocess:{}  expect:{}'\
                    .format(self._adversary.original_label, self._adversary.target_label))

        # binary search for smallest c and smallest l2
        logging.info('searching for the smallest c that makes attack possible within ({},{})'\
                     .format(c_range[0],c_range[1]))
        # init ad perturbation with (-0.001, 0.001)
        ad_perturbation = fluid.global_scope().find_var("parameter").get_tensor()
        ad_perturbation.set(0.002 * np.random.random_sample(self._shape).astype('float32') - 0.001, self.place)

        self.l2 = None
        self.img_adv = None
        c_half = c_start
        c_low = c_range[0]
        c_high = c_range[1]
        for i in range(c_search_step):
            l2, img_adv = self._cwb(img, c_half, attack_steps=attack_iterations)
            # update smallest l2 and img_adv
            if l2 != None:
                # found smallest l2 by current c
                if self.l2 == None or self.l2 > l2:
                    self.l2 = l2
                    self.img_adv = img_adv
                c_high = c_half
            else:
                c_low = c_half

            new_half = (c_low + c_high) / 2
            # end of reaching maximum accuracy
            if abs(new_half - c_half) <= c_accuracy:
                break
            c_half = new_half
        # if each c is unsuccessful , use img as the end
        if self.l2 == None:
            self.img_adv = img
        adv_label, adv_score = self._predict_adv(self.img_adv)
        print(('predict label:', adv_label, 'softmax:', adv_score))
        # check adversary target if success
        self.img_adv = np.squeeze(self.img_adv)
        self.img_adv = self.img_adv.reshape(img.shape)

        self._adversary.try_accept_the_example(self.img_adv, adv_label)

        return adversary

    def _cwb(self, img, c, attack_steps):
        """
        use CW attack on an original image for a
        limited number of iterations
        :return l2, img_adv
        """
        smallest_l2 = None
        corresponding_constrained = None
        # inital data
        screen_nontarget_logit = np.zeros(shape=[self._dim], dtype="float32")
        screen_nontarget_logit[self._adversary.target_label] = 1

        feeder = fluid.DataFeeder(
            feed_list=['img_data_scaled',
                       "target",
                       "c"],
            place=self.place,
            program=self.attack_main_program)

        # img normalization
        img_0_1 = self._process_input(img)
        # calculate adv and l2
        for i in range(attack_steps):
            result = self.exe.run(self.attack_main_program,
                                  feed=feeder.feed([(img_0_1,
                                                     screen_nontarget_logit,
                                                     c)]),

                                  fetch_list=[self.constrained,
                                              self.distance_L2,
                                              self.loss,
                                              self.logits,
                                              self.softmax])

            pred = result[3][0]
            pre_index = np.argmax(pred)
            softmax = result[4][0]
            l2 = result[1][0]

            #logging.info("distance_L2:{} pred:{} softmax:{}".format(l2,pre_index,softmax[pre_index]))
            #Validation using inference models
            adv_label, adv_score = self._predict_adv(result[0])
            logging.info("distance_L2:{} pred:{} softmax:{}".format(l2, adv_label, adv_score))

            if adv_label == self._adversary.target_label \
                    and adv_score > self.confidence:
                if smallest_l2 == None or l2 < smallest_l2:
                    smallest_l2 = l2
                    corresponding_constrained = result[0]
            ######
        # output result for this c
        if smallest_l2 != None:
            adv_label, adv_score = self._predict_adv(corresponding_constrained)
            print('Checking if {0:f} is a successful c.'.format(c))
            print('label:{} softmax:{} L2:{}'.format(adv_label, adv_score, smallest_l2))
        else:
            print('Checking if {0:f} is a unsuccessful c.'.format(c))

        return smallest_l2, corresponding_constrained

    # this build up the CW attack computation graph in Paddle
    def _constrain_cwb(self, img_0_1):
        """
        create constrained and distance with img_0_1(0,1)
        """
        # img to (-1, 1)
        self.y = 2 * img_0_1 - 1
        # compute arctan for y to get w
        self.xplus1 = 1 + self.y
        self.xminus1 = 1 - self.y
        self.ln = fluid.layers.log(self.xplus1 / self.xminus1)
        self.w = fluid.layers.scale(x=self.ln, scale=0.5)
        self.w_ad = self.w + self.ad_perturbation

        self.tanh_constrained = (fluid.layers.tanh(self.w_ad) + 1) * 0.5
        self.tanh_original = (fluid.layers.tanh(self.w) + 1) * 0.5
        # restore the range as original img
        self.constrained = self.reconstruct(self.tanh_constrained)
        self.original = self.reconstruct(self.tanh_original)
        # L2
        self.sub = fluid.layers.elementwise_sub(self.constrained, self.original)
        self.squared = fluid.layers.elementwise_mul(self.sub, self.sub)
        self.distance_L2 = fluid.layers.reduce_sum(self.squared)

        return self.constrained, self.distance_L2

    def _loss_cwb(self, target, constrained, distance_L2, c):
        """
        loss function for cw
        The components are L2 and f6
        """
        self.logits = self._attack_model(constrained)
        self.softmax = fluid.layers.softmax(self.logits)

        self.negetive_screen_nontarget_logit = fluid.layers.scale(target, scale=-1.0)
        self.screen_target_logit = self.negetive_screen_nontarget_logit.__add__(
            fluid.layers.ones(shape=[self._dim], dtype="float32"))

        self.logits_i_not_t = fluid.layers.elementwise_mul(self.screen_target_logit, self.logits)
        self.logit_target = fluid.layers.elementwise_mul(target, self.logits)

        self.maxlogit_i_not_t = fluid.layers.reduce_max(self.logits_i_not_t)
        self.maxlogit_target = fluid.layers.reduce_sum(self.logit_target)

        self.softmax_target = fluid.layers.elementwise_mul(target, self.softmax)
        self.maxsoftmax_target = fluid.layers.reduce_sum(self.softmax_target)

        self.difference_between_two_logits = self.maxlogit_i_not_t - self.maxlogit_target

        self.soft_diff_two_logits = self.maxlogit_i_not_t * (self.confidence - self.maxsoftmax_target)
        self.f6 = fluid.layers.relu(self.difference_between_two_logits + self.soft_diff_two_logits)

        self.loss = c * self.f6 + distance_L2

        return self.loss, self.logits, self.softmax

    def reconstruct(self, corresponding_constrained):
        """
        restore the img from corresponding_constrained float32 ===> (clip_min, clip_max)
        :return: numpy.ndarray
        """
        return corresponding_constrained * (self.clip_max - self.clip_min) + self.clip_min

    def _process_input(self, input_):
        """
        format img form (clip_min, clip_max) to (0, 1)
        """
        res = None
        sub = self.pa_clip_min
        div = self.pa_clip_max - self.pa_clip_min

        if np.any(sub != 0):
            res = input_ - sub
        if not np.all(sub == 1):
            if res is None:  # "res = input_ - sub" is not executed!
                res = input_ / (div)
            else:
                res /= div
        if res is None:  # "res = (input_ - sub)/ div" is not executed!
            return input_

        res = np.where(res == 0, 0.00001, res)
        res = np.where(res == 1, 0.99999, res)  # no 0 or 1

        return res

    def _predict_adv(self, img_constrained):
        """
        model predict
        the image already restore(clip_min, clip_max)
        """
        adv_logits = self.model.predict(img_constrained)
        adv_label = np.argmax(adv_logits)

        return adv_label, adv_logits[adv_label] # adv_lab, adv_score


CW_L2 = CW_L2_Attack

