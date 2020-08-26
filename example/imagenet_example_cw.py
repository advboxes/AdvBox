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
#attack resnet and alexnet model with CW, and the dataset is imagenet

from __future__ import print_function
import sys

sys.path.append("..")

import os
import numpy as np
import logging
import paddle.fluid as fluid
import paddle

#classification
import models
import reader
import argparse
import functools
from utility import add_arguments, print_arguments, generation_image

#attack
from adversarialbox.adversary import Adversary
from adversarialbox.attacks.cw import CW_L2
from adversarialbox.models.paddle import PaddleModel


#通过设置环境变量WITH_GPU 来动态设置是否使用GPU资源 特别适合在mac上开发但是在GPU服务器上运行的情况
#比如在mac上不设置该环境变量，在GPU服务器上设置 export WITH_GPU=1
with_gpu = os.getenv('WITH_GPU', '0') != '0'

# Test image
#  DATA_PATH is test image path
#  TEST_LIST is desc file, Support multiple files
TEST_LIST = './images/mytest_list.txt'
DATA_PATH = './images'

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,  256,                  "Minibatch size.")
add_arg('use_gpu',          bool, False,                 "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                 "Class number.")
add_arg('image_shape',      str,  "3,224,224",          "Input image size")
#add_arg('pretrained_model', str,  "./parameters/resnet_50/115",                 "Whether to use pretrained model.")
add_arg('pretrained_model', str,  "./parameters/alexnet/116",                 "Whether to use pretrained model.")
#add_arg('model',            str,  "ResNet50", "Set the network to use.")
add_arg('model',            str,  "AlexNet", "Set the network to use.")
add_arg('target',           int,  -1, "target class.")
add_arg('log_debug',        bool,  False, "Whether to open logging DEBUG.")
add_arg('inference',        bool,  False, "only inference,do not create adversarial example.")

model_list = [m for m in dir(models) if "__" not in m]
print(model_list)

def infer(infer_program, image, logits, place, exe):

    print("--------------------inference-------------------")

    test_batch_size = 1
    test_reader = paddle.batch(reader.test(TEST_LIST, DATA_PATH), batch_size=test_batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image])
    fetch_list = [logits.name]

    label_res = {}
    for batch_id, data in enumerate(test_reader()):
        data_img = data[0][0]
        filename = data[0][1]

        result = exe.run(infer_program,
                         fetch_list=fetch_list,
                         feed=feeder.feed([data_img]))
        #print(result)
        result = result[0][0]
        pred_label = np.argmax(result)
        print("Test-{0}-score: {1}, class {2}, name={3}"
              .format(batch_id, result[pred_label], pred_label, filename))
        label_res[filename] = pred_label
        sys.stdout.flush()

    return label_res


def main(use_cuda):
    """
    Advbox example which demonstrate how to use advbox.
    """
    # base marco
    TOTAL_NUM = 100
    IMG_NAME = 'image'
    LABEL_NAME = 'label'

    # parse args
    args = parser.parse_args()
    print_arguments(args)

    # parameters from arguments
    class_dim = args.class_dim
    model_name = args.model
    target_class = args.target
    pretrained_model = args.pretrained_model
    image_shape = [int(m) for m in args.image_shape.split(",")]
    if args.log_debug:
        logging.getLogger().setLevel(logging.INFO)

    assert model_name in model_list, "{} is not in lists: {}".format(args.model, model_list)

    # model definition
    model = models.__dict__[model_name]()

    # declare vars
    image = fluid.layers.data(name=IMG_NAME, shape=image_shape, dtype='float32')
    logits = model.net(input=image, class_dim=class_dim)

    # clone program and graph for inference
    infer_program = fluid.default_main_program().clone(for_test=True)

    image.stop_gradient = False
    label = fluid.layers.data(name=LABEL_NAME, shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=logits, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    BATCH_SIZE = 1
    test_reader = paddle.batch(
        reader.test(TEST_LIST, DATA_PATH), batch_size=BATCH_SIZE)
    # setup run environment
    enable_gpu = use_cuda and args.use_gpu
    place = fluid.CUDAPlace(0) if enable_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # advbox demo
    m = PaddleModel(
        fluid.default_main_program(),
        IMG_NAME,
        LABEL_NAME,
        logits.name,
        avg_cost.name,
        (0, 1),
        channel_axis=3)
    # Adversarial method: CW
    attack = CW_L2(m, learning_rate=0.1, attack_model=model.conv_net, with_gpu=enable_gpu,
                   shape=image_shape, dim=class_dim, confidence_level=0.9, multi_clip=True)
    attack_config = {"attack_iterations": 50,
                     "c_search_step": 10,
                     "c_range": (0.01,100),
                     "c_start": 10,
                     "targeted": True}

    # reload model vars
    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    # inference
    pred_label = infer(infer_program, image, logits, place, exe)
    # if only inference ,and exit
    if args.inference:
        exit(0)

    print("--------------------adversary-------------------")
    # use test data to generate adversarial examples
    total_count = 0
    fooling_count = 0
    for data in test_reader():
        total_count += 1
        data_img = [data[0][0]]
        filename = data[0][1]
        org_data = data_img[0][0]
        adversary = Adversary(org_data, pred_label[filename])
        #target attack
        if target_class != -1:
            tlabel = target_class
            adversary.set_target(is_targeted_attack=True, target_label=tlabel)

        adversary = attack(adversary, **attack_config)

        if adversary.is_successful():
            fooling_count += 1
            print(
                'attack success, original_label=%d, adversarial_label=%d, count=%d'
                % (pred_label[filename], adversary.adversarial_label, total_count))
            #output original image， adversarial image and difference image
            generation_image(total_count, org_data, pred_label[filename],
                        adversary.adversarial_example, adversary.adversarial_label, "CW")
        else:
            print('attack failed, original_label=%d, count=%d' %
                  (pred_label[filename], total_count))

        if total_count >= TOTAL_NUM:
            print(
                "[TEST_DATASET]: fooling_count=%d, total_count=%d, fooling_rate=%f"
                % (fooling_count, total_count,
                   float(fooling_count) / total_count))
            break

    print("cw attack done")

if __name__ == '__main__':
    main(use_cuda=with_gpu)
