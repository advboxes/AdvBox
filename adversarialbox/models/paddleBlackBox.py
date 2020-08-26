"""
Paddle BlackBox model
"""
from __future__ import absolute_import

import numpy as np
import os
import paddle.fluid as fluid

from .base import Model

#通过设置环境变量WITH_GPU 来动态设置是否使用GPU资源 特别适合在mac上开发但是在GPU服务器上运行的情况
#比如在mac上不设置该环境变量，在GPU服务器上设置 export WITH_GPU=1
with_gpu = os.getenv('WITH_GPU', '0') != '0'


class PaddleBlackBoxModel(Model):
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
                 _predict_program,
                 input_name,
                 logits_name,
                 predict_name,
                 bounds,
                 channel_axis=3,
                 preprocess=None):
        if preprocess is None:
            preprocess = (0, 1)

        super(PaddleBlackBoxModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocess=preprocess)

        #用于预测
        self._predict_program = _predict_program

        self._place = fluid.CUDAPlace(0) if with_gpu else fluid.CPUPlace()
        self._exe = fluid.Executor(self._place)

        self._input_name = input_name
        self._logits_name = logits_name
        self._predict_name = predict_name



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
        feeder = fluid.DataFeeder(
            feed_list=[self._input_name, self._logits_name],
            place=self._place,
            program=self._predict_program)
        predict_var = self._predict_program.block(0).var(self._predict_name)
        predict = self._exe.run(self._predict_program,
                                feed=feeder.feed([(scaled_data, 0)]),
                                fetch_list=[predict_var])
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
