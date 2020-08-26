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
Paddle model
"""
from __future__ import absolute_import

import sys
sys.path.append("..")

import numpy as np
import os
import paddle.fluid as fluid

from .base import Model
from .paddle  import PaddleModel
from advbox.defences.spatial_smoothing import SpatialSmoothingDefence

#通过设置环境变量WITH_GPU 来动态设置是否使用GPU资源 特别适合在mac上开发但是在GPU服务器上运行的情况
#比如在mac上不设置该环境变量，在GPU服务器上设置 export WITH_GPU=1
with_gpu = os.getenv('WITH_GPU', '0') != '0'

class PaddleSpatialSmoothingDefenceModel(PaddleModel):

    def __init__(self,
                 program,
                 input_name,
                 logits_name,
                 predict_name,
                 cost_name,
                 bounds,
                 channel_axis=3,
                 preprocess=None,
                 window_size=None,
                 channel_index=0):
        if preprocess is None:
            preprocess = (0, 1)

        super(PaddleSpatialSmoothingDefenceModel, self).__init__(
            program=program,input_name=input_name,logits_name=logits_name,predict_name=predict_name,
            cost_name=cost_name,
            bounds=bounds, channel_axis=channel_axis, preprocess=preprocess)


        self.__window_size=window_size
        self.__channel_index=channel_index


    def predict(self, data):

        x=SpatialSmoothingDefence(data.copy(),None,self.__window_size,self.__channel_index)
        predict = super(PaddleSpatialSmoothingDefenceModel, self).predict(x)
        return predict

