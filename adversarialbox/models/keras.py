"""
keras model
"""
from __future__ import absolute_import

from builtins import str
import numpy as np
import os

import sys
sys.path.append("..")

from .base import Model
from adversarialbox.defences.feature_squeezing import FeatureFqueezingDefence

import logging
logger=logging.getLogger(__name__)

class KerasModel(Model):


    def __init__(self,
                 module,
                 input,
                 label,
                 logits,
                 loss,
                 bounds,
                 channel_axis=3,
                 preprocess=None,
                 featurefqueezing_bit_depth=None):

        import keras.backend as k

        if preprocess is None:
            preprocess = (0, 1)

        super(KerasModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocess=preprocess)


        self._module = module

        self._loss=loss


        self._label=label
        self._logits=logits
        self._input = input

        _, self._nb_classes = k.int_shape(self._logits)
        self._input_shape = k.int_shape(self._input)[1:]

        logger.info('self._input_shape:'+str(self._input_shape))

        logger.info("init grads[{0}]...".format(self._nb_classes))

        #定义预测函数
        preds= self._logits
        self._preds = k.function([self._input], [preds])

        self._featurefqueezing_bit_depth=featurefqueezing_bit_depth

        if self._featurefqueezing_bit_depth is not None:
            logging.info('use featurefqueezing_bit_depth={}'.format(self._featurefqueezing_bit_depth))


        logger.info("Finish KerasModel init")


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

        import keras.backend as k
        k.set_learning_phase(0)

        if self._featurefqueezing_bit_depth is not None:
            #logging.info(data)
            scaled_data=FeatureFqueezingDefence(data.copy(),None,self._featurefqueezing_bit_depth,self._bounds)
            #logging.info(scaled_data)


        scaled_data = self._process_input(scaled_data)

        #logging.info(scaled_data)

        # Run prediction
        predict = self._preds(inputs=[scaled_data])
        predict = np.squeeze(predict, axis=0)

        #print(predict)
        #print(np.argmax(predict))

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


        import keras.backend as k
        k.set_learning_phase(0)


        scaled_data = self._process_input(data.copy())

        grads_logits=k.gradients(self._logits[:, label], self._input)[0]
        self._grads = k.function([self._input], [grads_logits])

        grads = self._grads([scaled_data])

        grads = np.swapaxes(np.array(grads), 0, 1)

        return grads.reshape(data.shape)

    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        :return: string
        """
        return self._predict_program.block(0).var(self._predict_name).op.type
