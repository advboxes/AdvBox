"""
This module provide the defence method for SpatialSmoothingDefence's implement.

Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks


"""
from __future__ import division
from builtins import range
from past.utils import old_div
import logging
logger=logging.getLogger(__name__)

import numpy as np
import tensorflow as tf

#tf的梯度知识：https://blog.csdn.net/wuguangbin1230/article/details/71169863

#FGSM 通过loss函数控制目标攻击或者无目标攻击
def fgsm(x,
        loss=None,
        eps=0.3,
        ord=np.inf,
        bounds=(0,1)):

    (clip_min, clip_max)=bounds

    grad, = tf.gradients(loss, x)

    if  ord == 1:
        red_ind = list(range(1, len(x.get_shape())))
        avoid_zero_div = 1e-8
        avoid_nan_norm = tf.maximum(avoid_zero_div,
                                    reduce_sum(tf.abs(grad),
                                               reduction_indices=red_ind,
                                               keepdims=True))
        normalized_grad = old_div(grad, avoid_nan_norm)
    elif ord == 2:
        red_ind = list(range(1, len(x.get_shape())))
        avoid_zero_div = 1e-8
        square = tf.maximum(avoid_zero_div,
                            reduce_sum(tf.square(grad),
                                       reduction_indices=red_ind,
                                       keepdims=True))
        normalized_grad = old_div(grad, tf.sqrt(square))
    else:
        normalized_grad = tf.sign(grad)
        normalized_grad = tf.stop_gradient(normalized_grad)

    scaled_grad = eps * normalized_grad

    #目标是让loss下降
    adv_x = x - scaled_grad

    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x

#DeepFool 仅实现了目标攻击
def deepfool(x,
        loss=None,
        bounds=(0,1)):

    (clip_min, clip_max)=bounds

    grad, = tf.gradients(loss, x)

    r=old_div(grad*loss,tf.reduce_sum(tf.square(grad)))

    #目标是让loss下降
    adv_x = x - r


    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x
