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


class TensorflowModel(Model):
    """
    Create a TensorflowModel instance.
    When you need to generate a adversarial sample, you should construct an
    instance of PaddleModel.
    Args:
        program(paddle.fluid.framework.Program): The program of the model
            which generate the adversarial sample.
        input_name(string): The name of the input.
        logits_name(string): The name of the logits.
        predict_name(string): The name of the predict.
        cost_name(string): The name of the loss in the program.
    """

    def __init__(self,
                 session,
                 input,
                 loss,
                 logits,
                 bounds,
                 channel_axis=3,
                 preprocess=None):

        import tensorflow as tf


        if preprocess is None:
            preprocess = (0, 1)

        super(TensorflowModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocess=preprocess)



        self._session = session
        self._loss=loss
        self._logits=logits
        self._input=input
        self._input_shape = tuple(input.get_shape()[1:])
        self._nb_classes=int(logits.get_shape()[-1])

        logger.info('self._input_shape:'+str(self._input_shape))

        self._probs = tf.nn.softmax(logits)

        self._loss_grads = tf.gradients(self._loss, self._input)[0]

        self._class_grads = [ tf.gradients(tf.nn.softmax(self._logits)[:, label], self._input)[0] for label in range(self._nb_classes)]


    def predict(self, data, logits=False):
        """
        Calculate the prediction of the data.
        Args:
            data(numpy.ndarray): input data with shape (size,
            height, width, channels).
        Return:
            numpy.ndarray: predictions of the data with shape (batch_size,
                num_of_classes).
        """

        import tensorflow as tf

        #scaled_data = self._process_input(data)

        #fd = {self._input: scaled_data}
        fd = {self._input: data}

        # Run prediction
        if logits:
            predict = self._session.run(self._logits, feed_dict=fd)
        else:
            predict = self._session.run(self._probs, feed_dict=fd)

        return predict

    def num_classes(self):
        """
            Calculate the number of classes of the output label.
        Return:
            int: the number of classes
        """

        return int(self._logits.get_shape()[-1])

    def gradient(self, data, label, logits=False):
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
        #scaled_data = self._process_input(data)

        import tensorflow as tf

        if logits:
            grads = self._session.run(self._logit_class_grads[label], feed_dict={self._input: data})
        else:
            grads = self._session.run(self._class_grads[label], feed_dict={self._input: data})

        grads = grads[None, ...]
        grads = np.swapaxes(np.array(grads), 0, 1)
        assert grads.shape == (data.shape[0], 1) + self.input_shape

        #grad = self._apply_processing_gradient(grad)

        assert grads.shape == data.shape


        return grads.reshape(data.shape)

    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        :return: string
        """
        return self._predict_program.block(0).var(self._predict_name).op.type
