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
This module provide the defence method for FeatureFqueezingDefence's implement.

Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks


"""
import logging
logger=logging.getLogger(__name__)

import numpy as np

__all__ = [
    'FeatureFqueezingDefence'
]

#bit_depth为设置的像素深度 允许的范围为1-64 一般为8以内
#clip_values为最终处理后取值范围 可能包含负数  常见的为[0,1] [-1,1]

def FeatureFqueezingDefence(x, y=None, bit_depth=None, clip_values=(0.0, 1.0)):

    assert (type(bit_depth) is int ) and  ( bit_depth >= 1 ) and  ( bit_depth <= 64)

    (LB,UB)=clip_values

    #归一化到(0,1)
    x_ = x - LB
    x_ = x_ / (UB-LB)

    #转换到bit_depth整数
    max_value = np.rint(2 ** bit_depth - 1)
    res = np.rint(x_ * max_value)

    #还原到clip_values范围
    res= res/max_value* (UB - LB) + LB

    #确保万无一失 clip生效
    assert (res <= UB).all() and (res >= LB).all()

    return res


