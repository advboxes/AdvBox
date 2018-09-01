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
This module provide the attack method of "CW".

L2 distance metrics especially
"""
from __future__ import division

import logging

import numpy as np

from tutorials.mnist_model import mnist_cnn_model
import paddle.fluid as fluid
import paddle.v2 as paddle
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.param_attr import ParamAttr
from .base import Attack
import pdb

__all__ = ['CW_L2_Attack', 'CW_L2']


# In[5]:


class CW_L2_Attack(Attack):
    """
    Uses Adam to minimize the CW L2 objective function

    Paper link: https://arxiv.org/abs/1608.04644
    """

    def __init__(self, model, learning_rate):

        super(CW_L2_Attack, self).__init__(model)
        self._predicts_normalized = None
        self._adversary = None  # type: Adversary
        #########################################
        # build cw attack computation graph
        self._place = self.model._place
        self._exe = self.model._exe

        # clone the prebuilt program that has cnn to attack
        self.attack_main_program = fluid.Program()  # prebuilt_program.clone(for_test=False)
        # create an empty program for variable init
        self.attack_startup_program = fluid.Program()  # start_up_program.clone(for_test=False)

        # build cw attack compute graph within attack programs
        with fluid.program_guard(main_program=self.attack_main_program, startup_program=self.attack_startup_program):
            img_0_1_placehold = fluid.layers.data(name='img_data_scaled', shape=[1, 28, 28], dtype="float32")
            target_placehold = fluid.layers.data(name='target', shape=[10], dtype="float32")
            shape_placehold = fluid.layers.data(name="shape", shape=[1], dtype="float32")
            # k_placehold = fluid.layers.data(name='k',shape=[1],dtype="float32")
            c_placehold = fluid.layers.data(name='c', shape=[1], dtype="float32")

            # get fluid.layer object from prebuilt program
            # img_placehold_from_prebuilt_program = attack_main_program.block(0).var(self.model._input_name)
            # softmax_from_prebuilt_program = attack_main_program.block(0).var(self.model._softmax_name)
            # logits_from_prebuilt_program = attack_main_program.block(0).var(self.model._predict_name)

            self.t0, self.t1, self.t2, self.t3, self.t4 = self._loss_cw(img_0_1_placehold,
                                                                        target_placehold,
                                                                        shape_placehold,
                                                                        c_placehold)  # ,
            # img_placehold_from_prebuilt_program,
            # softmax_from_prebuilt_program,
            # logits_from_prebuilt_program)
            
            # Init Adam optimizer as suggested in paper
            optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
            optimizer.minimize(self.t2, parameter_list=['parameter'])

            
        # initial variables and parameters every time before attack
        # print("Initial param!")
        self._exe.run(self.attack_startup_program)
        self.ret = fluid.global_scope().find_var("parameter").get_tensor()
        # print(np.array(ret))
        # self.ret.set((1/255) * np.random.random_sample((1, 28, 28)).astype('float32'), self._place)
        # print(np.array(ret))
        # print(attack_main_program.current_block()["parameter"])
        # pdb.set_trace()
        c1 = self.attack_main_program.block(0).var("conv2d_2.b_0")
        c2 = self.attack_main_program.block(0).var("conv2d_2.w_0")
        c3 = self.attack_main_program.block(0).var("conv2d_3.b_0")
        c4 = self.attack_main_program.block(0).var("conv2d_3.w_0")
        f1 = self.attack_main_program.block(0).var("fc_2.b_0")
        f2 = self.attack_main_program.block(0).var("fc_2.w_0")
        f3 = self.attack_main_program.block(0).var("fc_3.b_0")
        f4 = self.attack_main_program.block(0).var("fc_3.w_0")
        var_list = [c1, c2, c3, c4, f1, f2, f3, f4]

        fluid.io.load_vars(executor=self._exe, dirname="../advbox/attacks/mnist/", vars=var_list,
                           main_program=self.attack_main_program)  # ../advbox/attacks/mnist/
        #########################################

    def _apply(self,
               adversary,
               nb_classes=10,
               learning_rate=0.01,
               c_search_step=20,
               c_range=(0.01,100),
               attack_iterations=10000,
               multi_startpoints=10,
               targeted=True):

        # put adversary instance inside of the attack instance so all other function within can access
        self._adversary = adversary
        if not self._adversary.is_targeted_attack:
            raise ValueError("This attack method only support targeted attack!")
        
        print("Number of classes:",nb_classes,
              "Learning_rate:",learning_rate,
              "Attack_iterations:",attack_iterations,
              "c_range:",c_range,
              "Targeted:",targeted)
        
        img = self._adversary.original  # original image to be attacked
        # binary search for smallest c that makes f6<=0
        print('searching for the smallest c that makes attack possible within ({},{})'.format(c_range[0],c_range[1]))
        c_low = c_range[0]
        c_high = c_range[1]
        for i in range(c_search_step):
            logging.info('c_high={}, c_low={}, diff={}'.format(c_high, c_low, c_high - c_low))
            c_half = (c_low + c_high) / 2
            print('Checking if {0:f} is a successful c.'.format(c_half))
            # multi-point gradient descent
            for j in range(multi_startpoints):
                is_adversary, f6 = self._cwb(img,
                                             c_half,
                                             attack_steps=attack_iterations,
                                             learning_rate=learning_rate,
                                             nb_classes=nb_classes)
                if is_adversary: break
            # pdb.set_trace()
            is_f6_smaller_than_0 = f6 <= 0

            if is_adversary and is_f6_smaller_than_0:
                c_high = c_half
            else:
                c_low = c_half

        return adversary

    def _cwb(self, img, c, attack_steps, learning_rate, nb_classes):
        '''
        use CW attack on an original image for a
        limited number of iterations
        :return bool
        '''
        
        smallest_f6 = None
        corresponding_constrained = None

        # inital data
        # print("Initial parameter!")
        self.ret.set((1/255) * np.random.random_sample((1, 28, 28)).astype('float32'), self._place)
        screen_nontarget_logit = np.zeros(shape=[nb_classes], dtype="float32")
        screen_nontarget_logit[self._adversary.target_label] = 1

        feeder = fluid.DataFeeder(
            feed_list=["img_data_scaled",
                       "target",
                       "shape",
                       "c"],  # self.model._input_name,self.model._logits_name,
            place=self._place,
            program=self.attack_main_program)

        sub = -1
        div = 2

        img_0_1 = self._process_input(img, sub, div)
        # pdb.set_trace()
            
        for i in range(attack_steps):
            # print("steps:",i)
            result = self._exe.run(self.attack_main_program,
                                  feed=feeder.feed([(img_0_1,
                                                     screen_nontarget_logit,
                                                     np.zeros(shape=[1], dtype='float32'),
                                                     c)]),  # img_0_1,0,
                                  fetch_list=[self.maxlogit_i_not_t,
                                              self.maxlogit_target,
                                              self.loss,
                                              self.logits_i_not_t,
                                              self.constrained,
                                              self.softmax])
            '''
            print("maxlogit_i_not_t:",result[0],\
                  "maxlogit_target:",result[1],\
                  "loss:",result[2],
                  "logits_i_not_t:",result[3],\
                  "softmax:",result[5])
            '''
            f6 = result[0] - result[1]
            if i == 0:
                smallest_f6 = f6
                corresponding_constrained = result[4]
            if f6 < smallest_f6:
                smallest_f6 = f6
                corresponding_constrained = result[4]
                ######
        # pdb.set_trace()
        # print(corresponding_constrained)
        # recover image (-1,1) from corresponding_constrained which is within (0,1)
        img_ad = self.reconstruct(corresponding_constrained)
        # convert into img.shape
        img_ad = np.squeeze(img_ad)
        img_ad = img_ad.reshape(img.shape)
        # let model guess
        adv_label = np.argmax(self.model.predict(img_ad))  # img,img_ad
        '''
        print(self._adversary.original_label,self.model.predict(img))
        print(self._adversary.target_label,screen_nontarget_logit)
        print(adv_label,self.model.predict(img_ad))
        #pdb.set_trace()
        '''
        # try to accept new result, success or fail
        return self._adversary.try_accept_the_example(
            img_ad, adv_label), f6  # img,img_ad

    # this build up the CW attack computation graph in Paddle
    def _loss_cw(self, img_0_1, target, shape, c):  # ,img_input_entrance,softmax_entrance,logits_entrance

        ####
        # use layerhelper to init w
        self.helper = LayerHelper("Jay")
        # name a name for later to take it out
        self.param_attr = ParamAttr(name="parameter")

        # add this perturbation on w space, then, reconstruct as an image within (0,1)
        self.ad_perturbation = self.helper.create_parameter(attr=self.param_attr,
                                                            shape=[1, 28, 28],
                                                            dtype='float32',
                                                            is_bias=False)

        self.y = 2 * img_0_1 - 1
        # compute arctan for y to get w
        self.xplus1 = 1 + self.y
        self.xminus1 = 1 - self.y
        self.ln = fluid.layers.log(self.xplus1 / self.xminus1)
        self.w = fluid.layers.scale(x=self.ln, scale=0.5)
        self.w_ad = self.w + self.ad_perturbation
        self.tanh_w = fluid.layers.tanh(self.w_ad)
        self.constrained = 0.5 * (self.tanh_w + 1)

        self.softmax, self.logits = mnist_cnn_model(self.constrained)

        self.sub = fluid.layers.elementwise_sub(img_0_1, self.constrained)
        self.squared = fluid.layers.elementwise_mul(self.sub, self.sub)
        self.distance_L2 = fluid.layers.reduce_sum(self.squared)

        self.negetive_screen_nontarget_logit = fluid.layers.scale(target, scale=-1.0)
        self.screen_target_logit = self.negetive_screen_nontarget_logit.__add__(
            fluid.layers.ones(shape=[10], dtype="float32"))

        self.logits_i_not_t = fluid.layers.elementwise_mul(self.screen_target_logit, self.logits)
        self.logit_target = fluid.layers.elementwise_mul(target, self.logits)

        self.maxlogit_i_not_t = fluid.layers.reduce_max(self.logits_i_not_t)
        self.maxlogit_target = fluid.layers.reduce_sum(self.logit_target)

        self.difference_between_two_logits = self.maxlogit_i_not_t - self.maxlogit_target

        self.f6 = fluid.layers.relu(self.difference_between_two_logits)

        self.loss = c * self.f6 + self.distance_L2

        return self.maxlogit_i_not_t, self.maxlogit_target, self.loss, self.logits_i_not_t, self.constrained  # distance_L2

    # reconstruct corresponding_constrained to an image in MNIST format
    def reconstruct(self, corresponding_constrained):
        """
        Restore the img from corresponding_constrained float32
        :return: numpy.ndarray
        """
        return corresponding_constrained * 2 - 1  # mnist is belong to (-1,1)
    # unused
    def _f6(self, w):
        '''
        _f6 is the special f function CW chose as part of the
        objective function, this returns the values directly
        :return float32
        '''
        target = self._adversary.target_label
        img = (np.tanh(w) + 1) / 2
        Z_output = self._Z(img)
        f6 = max(max([Z for i, Z in enumerate(Z_output) if i != target]) - Z_output[target], 0)

        return f6
    # unused
    def _Z(self, img):
        """
        Get the Zx logits as a numpy array.
        :return: numpy.ndarray
        """
        return self.model.get_logits(img)

    def _process_input(self, input_, sub, div):
        res = None

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


CW_L2 = CW_L2_Attack


class NumpyInitializer(fluid.initializer.Initializer):

    def __init__(self, ndarray):
        super(NumpyInitializer, self).__init__()
        self._ndarray = ndarray.astype('float32')

    def __call__(self, var, block):
        values = [float(v) for v in self._ndarray.flat]
        op = block.append_op(
            type="assign_value",
            outputs={"Out": [var]},
            attrs={
                "shape": var.shape,
                "dtype": int(var.dtype),
                "fp32_values": values,
            })
        var.op = op
        return op