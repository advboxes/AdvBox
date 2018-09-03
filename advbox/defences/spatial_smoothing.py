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
This module provide the defence method for SpatialSmoothingDefence's implement.

Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks


"""
import logging
logger=logging.getLogger(__name__)

import numpy as np

from scipy import ndimage

__all__ = [
    'SpatialSmoothingDefence'
]

#window_size 窗口大小 推荐[1，20] channel_index标记彩色信道  灰色图像设置为0
#中值滤波法是一种非线性平滑技术，它将每一像素点的灰度值设置为该点某邻域窗口内的所有像素点灰度值的中值.
#中值滤波是基于排序统计理论的一种能有效抑制噪声的非线性信号处理技术，中值滤波的基本原理是把数字图像或
#数字序列中一点的值用该点的一个邻域中各点值的中值代替，让周围的像素值接近的真实值，从而消除孤立的噪声点。

#Feature Squeezing:Detecting Adversarial Examples in Deep Neural Networks中window_size为2
def SpatialSmoothingDefence(x, y=None, window_size=2,channel_index=0):

    assert channel_index < len(x.shape)

    #在每个信道都进行中值滤波 窗口大小为window_size 彩色信道为1
    median_filte_size = [1] + [window_size] * (len(x.shape) - 1)
    median_filte_size[channel_index] = 1
    median_filte_size = tuple(median_filte_size)

    res = ndimage.filters.median_filter(x, size=median_filte_size, mode="reflect")

    return res


