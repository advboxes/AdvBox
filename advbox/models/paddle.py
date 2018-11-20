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

import numpy as np
import os
import paddle.fluid as fluid

from .base import Model

#通过设置环境变量WITH_GPU 来动态设置是否使用GPU资源 特别适合在mac上开发但是在GPU服务器上运行的情况
#比如在mac上不设置该环境变量，在GPU服务器上设置 export WITH_GPU=1
with_gpu = os.getenv('WITH_GPU', '0') != '0'


class PaddleModel(Model):
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
                 program,
                 input_name,
                 logits_name,
                 predict_name,
                 cost_name,
                 bounds,
                 channel_axis=3,
                 preprocess=None):
        if preprocess is None:
            preprocess = (0, 1)

        super(PaddleModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocess=preprocess)

        #用于计算梯度
        self._program = program
        #仅用于预测
        self._predict_program = program.clone(for_test=True)
        self._place = fluid.CUDAPlace(0) if with_gpu else fluid.CPUPlace()
        self._exe = fluid.Executor(self._place)

        self._input_name = input_name
        self._logits_name = logits_name
        self._predict_name = predict_name
        self._cost_name = cost_name

        #change all `is_test` attributes to True 使_program只计算梯度 不自动更新参数 单纯clone后不计算梯度的
        import six
        for i in six.moves.range(self._program.desc.num_blocks()):
            block = self._program.desc.block(i)
            for j in six.moves.range(block.op_size()):
                op = block.op(j)
                if op.has_attr('is_test') and op.type != 'batch_norm_grad':
                    # 兼容旧版本 paddle
                    if hasattr(op, 'set_attr'):
                        op.set_attr('is_test', True)
                    else:
                        op._set_attr('is_test', True)
        # gradient
        loss = self._program.block(0).var(self._cost_name)
        param_grads = fluid.backward.append_backward(
            loss, parameter_list=[self._input_name])
        self._gradient = filter(lambda p: p[0].name == self._input_name,
                                param_grads)[0][1]

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
        predict_var = self._program.block(0).var(self._predict_name)
        assert len(predict_var.shape) == 2
        return predict_var.shape[1]

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
        scaled_data = self._process_input(data)

        feeder = fluid.DataFeeder(
            feed_list=[self._input_name, self._logits_name],
            place=self._place,
            program=self._program)

        grad, = self._exe.run(self._program,
                              feed=feeder.feed([(scaled_data, label)]),
                              fetch_list=[self._gradient])
        return grad.reshape(data.shape)

    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        :return: string
        """
        return self._predict_program.block(0).var(self._predict_name).op.type
