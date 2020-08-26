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


from __future__ import print_function
from __future__ import division

from past.utils import old_div
import sys


import logging
logging.basicConfig(level=logging.INFO,format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
#logger=logging.getLogger(__name__)

#sys.path.append("../../")

from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import time
import requests
from graphpipe import remote
from adversarialbox.adversary import Adversary
from adversarialbox.attacks.localsearch import LocalSearchAttack
from adversarialbox.models.graphpipeBlackBox import graphpipeBlackBoxModel_onnx
from adversarialbox.models.graphpipeBlackBox import graphpipeBlackBoxModel

from optparse import OptionParser

#定义参数
"""

"""
parser = OptionParser(usage="%prog [options]")
parser.add_option("-u",
                  "--url",
                  default="http://127.0.0.1:9000",
                  type="string",
                  dest="url",
                  help="graphpipe url [default: %default]")

parser.add_option("-m",
                  "--model",
                  default="onnx",
                  type="string",
                  dest="m",
                  help="Deep learning frame [default: %default] ;must be in [onnx,tersorflow]")

parser.add_option("-R",
                  "--rounds",
                  default="200",
                  type="int",
                  dest="R",
                  help="An upper bound on the number of iterations [default: %default]")

parser.add_option("-p",
                  "--p-parameter",
                  default="0.3",
                  type="float",
                  dest="p",
                  help="Perturbation parameter that controls the pixel sensitivity estimation [default: %default]")


parser.add_option("-r",
                  "--r-parameter",
                  default="1.4",
                  type="float",
                  dest="r",
                  help="Perturbation parameter that controls the cyclic perturbation;must be in [0, 2]")

parser.add_option("-d",
                  "--d-parameter",
                  default="5",
                  type="int",
                  dest="d",
                  help="The half side length of the neighborhood square [default: %default]")

parser.add_option("-t",
                  "--t-parameter",
                  default="5",
                  type="int",
                  dest="t",
                  help="The number of pixels perturbed at each round [default: %default]")

parser.add_option("-i",
                  "--input-file",
                  default="mug227.png",
                  type="string",
                  dest="input_file",
                  help="Original image file [default: %default]")

parser.add_option("-o",
                  "--output-file",
                  default="adversary_image.jpg",
                  type="string",
                  dest="output_file",
                  help="Adversary image file [default: %default]")

parser.add_option("-c",
                  "--channel_axis",
                  default="0",
                  type="int",
                  dest="c",
                  help="Channel_axis [default: %default] ;must be in 0,1,2,3")

(options,args)=parser.parse_args()

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


docker run -it --rm \
      -e https_proxy=${https_proxy} \
      -v "$PWD:/models/"  \
      -p 9000:9000 \
      sleepsonthefloor/graphpipe-tf:cpu \
      --model=/models/squeezenet.pb \
      --listen=0.0.0.0:9000


ONNX
docker run -it --rm \
      -e https_proxy=${https_proxy} \
      -p 9000:9000 \
      sleepsonthefloor/graphpipe-onnx:cpu \
      --value-inputs=https://oracle.github.io/graphpipe/models/squeezenet.value_inputs.json \
      --model=https://oracle.github.io/graphpipe/models/squeezenet.onnx \
      --listen=0.0.0.0:9000

本地模式
docker run -it --rm \
        -v "$PWD:/models/"  \
        -p 9000:9000 \
        sleepsonthefloor/graphpipe-onnx:cpu \
        --value-inputs=https://oracle.github.io/graphpipe/models/squeezenet.value_inputs.json \
        --model=/models/squeezenet.onnx \
        --listen=0.0.0.0:9000


更多ONNX模型 请参考
https://github.com/onnx/models
更多tensorflow模型 请参考
https://github.com/tensorflow/models

'''


#绘图函数
import matplotlib
#服务器环境设置
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

#对比展现原始图片和对抗样本图片
def show_images_diff(original_img,adversarial_img):
    original_img = np.array(Image.open(original_img))
    adversarial_img = np.array(Image.open(adversarial_img))

    plt.figure()

    original_img=original_img/255.0
    adversarial_img=adversarial_img/255.0

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial Image')
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference')
    difference = adversarial_img - original_img
    #(-1,1)  -> (0,1)
    #灰色打底 容易看出区别
    difference=old_div(difference, abs(difference).max())/2.0+0.5
    #print(difference)
    plt.imshow(difference)
    plt.axis('off')

    plt.show()
    #plt.savefig('fig_cat.png')


def main():

    assert 0 <= options.r <= 2
    assert options.c in [0,1,2,3]
    assert options.m in ["onnx","tersorflow"]

    print("options:{}".format(options))

    if options.m == "onnx":
        m = graphpipeBlackBoxModel_onnx(
            options.url, (0, 255),
            channel_axis=options.c)

    else:
        m = graphpipeBlackBoxModel(
            options.url, (0, 255),
            channel_axis=options.c)

    start = time.time()

    # 不定向攻击
    attack = LocalSearchAttack(m)

    # R 攻击次数
    # r p 绕定系数
    # t 每次攻击的点数
    # d 搜索半径
    attack_config = {"R": options.R, "r": options.r, "p": options.p, "t": options.t,"d": options.d}

    data = np.array(Image.open(options.input_file))
    data = data.reshape([1] + list(data.shape))
    data = np.rollaxis(data, 3, 1).astype(np.float32)  # channels first
    print("Image shape :{}".format(data.shape))

    original_data = np.copy(data)
    # 猫对应的标签 imagenet 2012 对应链接https://blog.csdn.net/LegenDavid/article/details/73335578
    original_label = None
    adversary = Adversary(original_data, original_label)

    print("Non-targeted Attack...")
    adversary = attack(adversary, **attack_config)

    if adversary.is_successful():

        print(
            'attack success, original_label=%d, adversarial_label=%d'
            % (adversary.original_label, adversary.adversarial_label))

        # 对抗样本保存在adversary.adversarial_example
        adversary_image = np.copy(adversary.adversarial_example)

        adversary_image = np.array(adversary_image[0]).astype("uint8").transpose([1, 2, 0])

        im = Image.fromarray(adversary_image)
        im.save(options.output_file)

        print("Save file :{}".format(options.output_file))

        show_images_diff(options.input_file,options.output_file)


    else:
        print('attack failed, original_label=%d' % (adversary.original_label))

    end = time.time()
    print("LocalSearchAttack attack done. Cost time {}s".format(end-start))


if __name__ == '__main__':
    main()




