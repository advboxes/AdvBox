"""
Tensorflow model
"""
from __future__ import absolute_import

from builtins import str
from builtins import range
import numpy as np
import os

from .base import Model

import logging
logger=logging.getLogger(__name__)

#直接加载pb文件
class TensorflowModel(Model):


    def __init__(self,
                 session,
                 input,
                 label,
                 logits,
                 loss,
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


        self._label=label
        self._logits=logits
        self._input = input
        self._input_shape = tuple(self._input.get_shape()[1:])
        self._nb_classes=int(self._logits.get_shape()[-1])

        logger.info('self._input_shape:'+str(self._input_shape))

        logger.info("init grads[{0}]...".format(self._nb_classes))

        #self._grads= tf.gradients(self._loss, self._input)[0]
        #self._grads= tf.gradients(self._logits, self._input)[0]
        self._grads = [ None for _ in range(self._nb_classes) ]


        logger.info("Finish TensorflowPBModel init")


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

        import tensorflow as tf

        scaled_data = self._process_input(data)

        #print(scaled_data)

        fd = {self._input: scaled_data}

        # Run prediction
        predict = self._session.run(self._logits, feed_dict=fd)
        predict = np.squeeze(predict, axis=0)

        return predict

    def num_classes(self):
        """
            Calculate the number of classes of the output label.
        Return:
            int: the number of classes
        """

        return int(self._logits.get_shape()[-1])

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


        import tensorflow as tf

        scaled_data = self._process_input(data)

        if self._grads[label] is None:
            logging.info('Start to get _grads[{}]'.format(label))
            self._grads[label]=tf.gradients(self._logits[:,label], self._input)[0]
            logging.info('Finish to get _grads[{}]'.format(label))

        grads = self._session.run(self._grads[label], feed_dict={self._input: scaled_data})

        grads = grads[None, ...]
        grads = np.swapaxes(np.array(grads), 0, 1)

        return grads.reshape(data.shape)

    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        :return: string
        """
        return self._predict_program.block(0).var(self._predict_name).op.type
