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
Tensorflow model
"""
from __future__ import absolute_import

import numpy as np
import os

from .base import Model

import logging
logger=logging.getLogger(__name__)


#直接加载pb文件
class MxNetModel(Model):


    def __init__(self,
                 model,
                 loss,
                 bounds,
                 channel_axis=3,
                 preprocess=None):

        import mxnet

        if preprocess is None:
            preprocess = (0, 1)

        super(MxNetModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocess=preprocess)


        self._model = model

        #暂时不支持自定义loss
        self._loss=loss

        logger.info("Finish MxNetModel init")

    #返回值为标量
    def predict(self, data):
        """
        Calculate the prediction of the data.
        Args:
            data(numpy.ndarray): input data with shape (size,
            height, width, channels).
        Return:
            numpy.ndarray: predictions of the data with shape (batch_size,
                num_of_classes).
        """

        import mxnet
        import mxnet as mx

        scaled_data = self._process_input(data)

        scaled_data=mx.nd.array(scaled_data)

        # Run prediction
        predict = self._model(scaled_data).asnumpy()
        predict = np.squeeze(predict, axis=0)

        #logging.info(predict)

        return predict

    def num_classes(self):
        """
            Calculate the number of classes of the output label.
        Return:
            int: the number of classes
        """

        return self._nb_classes

    def gradient(self, data, label):
        """
        Calculate the gradient of the cross-entropy loss w.r.t the image.
        Args:
            data(numpy.ndarray): input data with shape (size, height, width,
            channels).
            label(int): Label used to calculate the gradient.
        Return:
            numpy.ndarray: gradient of the cross-entropy loss w.r.t the image
                with the shape (height, width, channel).
        """

        import mxnet
        import mxnet as mx
        from mxnet import autograd, nd

        scaled_data = self._process_input(data)

        #logging.info(scaled_data)

        scaled_data = mx.nd.array(scaled_data)

        # 要求系统申请对应的空间存放梯度
        scaled_data.attach_grad()

        with autograd.record(train_mode=False):
            preds = self._model(scaled_data)
            class_slice = preds[:, label]

        class_slice.backward()
        grad = np.expand_dims(scaled_data.grad.asnumpy(), axis=1)


        return grad.reshape(scaled_data.shape)

    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        :return: string
        """
        return self._predict_program.block(0).var(self._predict_name).op.type
