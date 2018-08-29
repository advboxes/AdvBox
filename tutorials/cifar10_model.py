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

'''
ResNet on cifar10 data using fluid api of paddlepaddle
'''

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy
import sys
import os

from image_classification.resnet import resnet_cifar10

#通过设置环境变量WITH_GPU 来动态设置是否使用GPU资源 特别适合在mac上开发但是在GPU服务器上运行的情况
#比如在mac上不设置该环境变量，在GPU服务器上设置 export WITH_GPU=1
with_gpu = os.getenv('WITH_GPU', '0') != '0'

def inference_network():
    # The image is 32 * 32 with RGB representation.
    data_shape = [3, 32, 32]
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')

    #可选的resnet深度20, 32, 44, 56, 110, 1202
    '''
    实验数据
    深度为32
    Test with Pass 9, Loss 0.86, Acc 0.76
    深度为110
    Test with Pass 9, Loss 0.76, Acc 0.76
    '''
    predict = resnet_cifar10(images, 32)

    return predict


def train_network():
    predict = inference_network()
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy]


def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)


def train(use_cuda, train_program, params_dirname):
    #批处理大小
    BATCH_SIZE = 128
    #训练轮数
    EPOCH_NUM = 10

    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.cifar.train10(), buf_size=50000),
        batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            #每训练100个批次打印一次状态
            if event.step % 100 == 0:
                print("\nStep %d, Epoch %d, Cost %f, Acc %f" %
                      (event.step, event.epoch, event.metrics[0],
                       event.metrics[1]))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        if isinstance(event, fluid.EndEpochEvent):
            avg_cost, accuracy = trainer.test(
                reader=test_reader, feed_order=['pixel', 'label'])

            print('\nTest with Epoch {0}, Loss {1:2.2}, Acc {2:2.2}'.format(
                event.epoch, avg_cost, accuracy))
            if params_dirname is not None:
                trainer.save_params(params_dirname)

    #根据配置选择使用CPU资源还是GPU资源
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    trainer = fluid.Trainer(
        train_func=train_program, optimizer_func=optimizer_program, place=place)

    trainer.train(
        reader=train_reader,
        num_epochs=EPOCH_NUM,
        event_handler=event_handler,
        feed_order=['pixel', 'label'])



def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    #模型保存的路径
    save_path = "cifar10/resnet"

    train(
        use_cuda=use_cuda,
        train_program=train_network,
        params_dirname=save_path)


if __name__ == '__main__':

    main(use_cuda=with_gpu)