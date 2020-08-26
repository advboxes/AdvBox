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

#使用LocalSearchAttack攻击AlexNet 数据集为imagenet2012

from __future__ import print_function

import sys


import logging
logging.basicConfig(level=logging.INFO,format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger=logging.getLogger(__name__)


sys.path.append("../../")

from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import requests

from graphpipe import remote

from adversarialbox.adversary import Adversary
from adversarialbox.attacks.localsearch import LocalSearchAttack

from adversarialbox.models.graphpipeBlackBox import graphpipeBlackBoxModel


'''
#服务器端启动方式为：

Tensorflow

cpu
docker run -it --rm \
      -e https_proxy=${https_proxy} \
      -p 9000:9000 \
      sleepsonthefloor/graphpipe-tf:cpu \
      --model=https://oracle.github.io/graphpipe/models/squeezenet.pb \
      --listen=0.0.0.0:9000

ONNX

docker run -it --rm \
      -e https_proxy=${https_proxy} \
      -p 9000:9000 \
      sleepsonthefloor/graphpipe-onnx:cpu \
      --value-inputs=https://oracle.github.io/graphpipe/models/squeezenet.value_inputs.json \
      --model=https://oracle.github.io/graphpipe/models/squeezenet.onnx \
      --listen=0.0.0.0:9000
'''

def main():

    m = graphpipeBlackBoxModel(
        "http://127.0.0.1:9000", (0, 255),
        channel_axis=0)

    #不定向攻击
    attack = LocalSearchAttack(m)

    # R 攻击次数
    # r p 绕定系数
    # t 每次攻击的点数
    # d 搜索半径
    attack_config = {"R": 200,"r":1.4,"p":0.3,"t":5}

    data = np.array(Image.open("mug227.png"))
    data = data.reshape([1] + list(data.shape))
    data = np.rollaxis(data, 3, 1).astype(np.float32)  # channels first
    print(data.shape)

    original_data=np.copy(data)
    # 猫对应的标签 imagenet 2012 对应链接https://blog.csdn.net/LegenDavid/article/details/73335578
    original_label = None
    adversary = Adversary(original_data, original_label)

    logger.info("Non-targeted Attack...")
    adversary = attack(adversary, **attack_config)

    if adversary.is_successful():

        print(
                'attack success, original_label=%d, adversarial_label=%d'
                % (adversary.original_label, adversary.adversarial_label))

        #对抗样本保存在adversary.adversarial_example
        adversary_image=np.copy(adversary.adversarial_example)

        adversary_image = np.array(adversary_image[0]).astype("uint8").transpose([1, 2, 0])

        im = Image.fromarray(adversary_image)
        im.save("adversary_image.jpg")


    else:
        print('attack failed, original_label=%d' % (adversary.original_label))

    logger.info("LocalSearchAttack attack done")



if __name__ == '__main__':
    main()
