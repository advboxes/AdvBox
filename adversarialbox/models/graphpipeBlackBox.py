"""
graphpipe BlackBox model
"""
from __future__ import absolute_import

import logging
logger=logging.getLogger(__name__)

import numpy as np
import os
import requests

from graphpipe import remote

from .base import Model

class graphpipeBlackBoxModel(Model):
    """
    Create a PaddleModel instance.
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
                 url,
                 bounds,
                 channel_axis=3,
                 preprocess=None):
        if preprocess is None:
            preprocess = (0, 1)

        super(graphpipeBlackBoxModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocess=preprocess)

        #用于预测
        self._remote_url = url




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
        scaled_data = self._process_input(data)

        predict = remote.execute(self._remote_url, scaled_data)

        predict = np.squeeze(predict, axis=0)

        return predict

    def num_classes(self):
        """
            Calculate the number of classes of the output label.
        Return:
            int: the number of classes
        """
        predict_var = self._predict_program.block(0).var(self._predict_name)
        assert len(predict_var.shape) == 2
        return predict_var.shape[1]


    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        :return: string
        """
        return self._predict_program.block(0).var(self._predict_name).op.type


    def gradient(self, data, label):

        return None

class graphpipeBlackBoxModel_onnx(Model):
    """
    Create a PaddleModel instance.
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
                 url,
                 bounds,
                 channel_axis=3,
                 preprocess=None):
        if preprocess is None:
            preprocess = (0, 1)

        super(graphpipeBlackBoxModel_onnx, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocess=preprocess)

        #用于预测
        self._remote_url = url


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
        scaled_data = self._process_input(data)

        predict = remote.execute(self._remote_url, scaled_data)

        predict = predict.reshape([1,np.max(predict.shape)])

        predict = np.squeeze(predict, axis=0)

        return predict

    def num_classes(self):
        """
            Calculate the number of classes of the output label.
        Return:
            int: the number of classes
        """
        predict_var = self._predict_program.block(0).var(self._predict_name)
        assert len(predict_var.shape) == 2
        return predict_var.shape[1]


    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        :return: string
        """
        return self._predict_program.block(0).var(self._predict_name).op.type


    def gradient(self, data, label):

        return None
