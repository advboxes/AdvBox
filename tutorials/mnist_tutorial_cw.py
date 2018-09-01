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
CW tutorial on mnist using advbox tool.
CW method only supports targeted attack.
"""
import sys

sys.path.append("..")

#import matplotlib.pyplot as plt
import paddle.fluid as fluid
import paddle.v2 as paddle
import os
from advbox.adversary import Adversary
from advbox.attacks.CW_L2 import CW_L2
from advbox.models.paddle import PaddleModel
from tutorials.mnist_model import mnist_cnn_model


#通过设置环境变量WITH_GPU 来动态设置是否使用GPU资源 特别适合在mac上开发但是在GPU服务器上运行的情况
#比如在mac上不设置该环境变量，在GPU服务器上设置 export WITH_GPU=1
with_gpu = os.getenv('WITH_GPU', '0') != '0'

def main(use_cuda):
    """
    Advbox demo which demonstrate how to use advbox.
    """
    TOTAL_NUM = 500
    IMG_NAME = 'img'
    LABEL_NAME = 'label'

    # create two empty program for training and variable init 
    cnn_main_program = fluid.Program()
    cnn_startup_program = fluid.Program()
    
    # 可以改成default_main_program()
    with fluid.program_guard(main_program=cnn_main_program, startup_program=cnn_startup_program):
        img = fluid.layers.data(name=IMG_NAME, shape=[1, 28, 28], dtype='float32')
        # gradient should flow
        img.stop_gradient = False
        label = fluid.layers.data(name=LABEL_NAME, shape=[1], dtype='int64')
        softmax, logits = mnist_cnn_model(img)
        cost = fluid.layers.cross_entropy(input=softmax, label=label)
        avg_cost = fluid.layers.mean(x=cost)

    #根据配置选择使用CPU资源还是GPU资源
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    BATCH_SIZE = 1

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.test(), buf_size=128 * 10),
        batch_size=BATCH_SIZE)

    fluid.io.load_params(
        exe, "./mnist/", main_program=cnn_main_program)

    # advbox demo
    m = PaddleModel(
        cnn_main_program,
        IMG_NAME,
        LABEL_NAME,
        
        softmax.name,
        logits.name,
        
        avg_cost.name, (-1, 1),
        channel_axis=1,
        preprocess = (-1, 2)) # x within(0,1) so we should do some transformation
    
    learning_rate = 0.05
    
    attack = CW_L2(m, learning_rate=learning_rate) # to init computation graph in attack.init(), we have to pass learning_rate here
    #######
    # change parameter later
    #######
    attack_config = {"nb_classes": 10,
                     "learning_rate": learning_rate, # learning_rate already passed in, this is only for printing
                     "attack_iterations": 100,
                     "epsilon": 0.2,
                     "targeted": True}

    # use test data to generate adversarial examples
    total_count = 0
    fooling_count = 0
    for data in test_reader():
        total_count += 1
        adversary = Adversary(data[0][0], data[0][1])

        # CW_L2 targeted attack
        tlabel = 0
        adversary.set_target(is_targeted_attack=True, target_label=tlabel)
        adversary = attack(adversary, **attack_config)

        if adversary.is_successful():
            fooling_count += 1
            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                % (data[0][1], adversary.adversarial_label, total_count))

        else:
            print('attack failed, original_label=%d, count=%d' %
                  (data[0][1], total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break
    print("CW attack done")


if __name__ == '__main__':
    main(with_gpu)
