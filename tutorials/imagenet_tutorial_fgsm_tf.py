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
FGSM tutorial on mnist using advbox tool.
FGSM method is non-targeted attack while FGSMT is targeted attack.
"""
import sys
sys.path.append("..")
import logging
logging.basicConfig(level=logging.INFO,format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger=logging.getLogger(__name__)

#import matplotlib.pyplot as plt
import numpy as np

from advbox.adversary import Adversary
from advbox.attacks.gradient_method import FGSM
from advbox.models.tensorflowPB import TensorflowPBModel
from tutorials.mnist_model_tf import mnist_cnn_model
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main(dirname,imagename):
    """
    Advbox demo which demonstrate how to use advbox.
    """

    image_data = tf.gfile.FastGFile(imagename, 'rb').read()


    session=tf.Session()

    def create_graph(dirname):
        with tf.gfile.FastGFile(dirname, 'rb') as f:
            graph_def = session.graph_def
            graph_def.ParseFromString(f.read())

            _ = tf.import_graph_def(graph_def, name='')

    create_graph(dirname)

    # 初始化参数  非常重要


    session.run(tf.global_variables_initializer())

    #tensorlist=[n.name for n in session.graph_def.node]

    #logger.info(tensorlist)
    #获取softmax层而非logit层
    softmax = session.graph.get_tensor_by_name('softmax:0')

    #获取softmax/logits
    logits=session.graph.get_tensor_by_name('softmax/logits:0')

    x = session.graph.get_tensor_by_name('DecodeJpeg/contents:0')

    y = tf.placeholder(tf.int64, None, name='label')

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits)

    #tf.gradients(tf.nn.softmax(self._logits)[:, label], self._input)[0]





    print('!!!!!!!')
    #print(logits[:, 0])
    #print(tf.nn.softmax(logits[:, 0]) )
    #print(x)
    #print(cross_entropy)
    #print(g)
    #print(logits)
    #print(softmax)
    g = session.run(logits, feed_dict={x: image_data})
    print(g)

    g = session.run(softmax, feed_dict={x: image_data})
    print(g)
    #tf.gradients(tf.nn.softmax(self._logits)[:, label], self._input_ph)[0]
    #print(logits[:, 1])
    g = tf.gradients(logits, x)
    print(g)

    g = tf.gradients(softmax, x)
    print(g)

    z=tf.placeholder(tf.int64, None)
    z=2*y

    g = tf.gradients(z, y)
    print(g)





    #grads = session.run(g, feed_dict={x: image_data})

    #print(grads)




    # advbox demo
    m = TensorflowPBModel(
        session,
        x,
        y,
        softmax,
        cross_entropy,
        (0, 1),
        channel_axis=1)

    attack = FGSM(m)
    attack_config = {"epsilons": 0.3}


    #print(x.shape)
    #print(y.shape)

    adversary = Adversary(image_data,None)

    # FGSM non-targeted attack
    adversary = attack(adversary, **attack_config)


    if adversary.is_successful():
        print(
            'attack success, adversarial_label=%d'
            % (adversary.adversarial_label) )


    print("fgsm attack done")


if __name__ == '__main__':
    #从'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'下载并解压到当前路径
    #classify_image_graph_def.pb cropped_panda.jpg
    main(dirname="classify_image_graph_def.pb",imagename="cropped_panda.jpg")
