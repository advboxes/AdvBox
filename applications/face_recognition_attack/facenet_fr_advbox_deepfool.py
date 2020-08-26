# coding=utf-8

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

import tensorflow as tf
import numpy as np
from PIL import Image
import sys
from scipy import misc
import os
import time


sys.path.append("../../")

from  adversarialbox.attacks.tf.tools import  deepfool


sys.path.append("../thirdparty/facenet/src")
import facenet


FACENET_MODEL_CHECKPOINT = "20180402-114759.pb"


def get_pic_from_png(pic_path):
    img = misc.imread(os.path.expanduser(pic_path), mode='RGB')
    return regularize_pic(img)

def regularize_pic(img):
    return img * 2.0 / 255.0 - 1.0

def restore_pic(img):
    return (img + 1.0) * 255.0 / 2.0


def save_img2png(input_image, name):
    input_image = np.reshape(
        input_image, (input_image.shape[0], input_image.shape[1], input_image.shape[2]))
    img = np.round(restore_pic(input_image)).astype(np.uint8)
    img_ouput = Image.fromarray(img)
    filename = name + '.png'
    img_ouput.save(filename)


def Euclidian_distance(embeddings1, embeddings2):
    # Euclidian distance
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff))
    return np.sqrt(dist)


def cosine_distance(embeddings1, embeddings2):
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
    norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
    similarity = dot / norm
    dist = np.arccos(similarity) / math.pi
    return dist


def generate_inp2adv_name(input_pic, target_pic):
    (filepath, temp_input) = os.path.split(input_pic)
    (shotname_input, extension) = os.path.splitext(temp_input)

    (filepath, temp_target) = os.path.split(target_pic)
    (shotname_target, extension) = os.path.splitext(temp_target)

    return shotname_input + "_2_" + shotname_target



class FacenetFR():

    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(FACENET_MODEL_CHECKPOINT)

    def generate_embedding(self, pic):
        if type(pic) is str:
            pic = get_pic_from_png(pic)
        else:
            pic = regularize_pic(pic)

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [pic], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

    def compare(self, input_pic, target_pic):
        embedding1 = self.generate_embedding(input_pic)
        embedding2 = self.generate_embedding(target_pic)

        # if both pictures are same, return 0
        return Euclidian_distance(embedding1, embedding2)

    def generate_adv_whitebox(self, input_pic, target_pic):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        target_emb = self.generate_embedding(target_pic)

        #loss函数目的是将起下降
        loss = tf.sqrt(tf.reduce_sum(tf.square(embeddings - target_emb)))

        #定义deepfool迭代器
        adv_x=deepfool(images_placeholder,
                   loss=loss,
                    bounds=(-1.0,1.0))

        input_image = get_pic_from_png(input_pic)
        adv_image = np.reshape(
            input_image, (-1, input_image.shape[0], input_image.shape[1], input_image.shape[2]))


        #损失函数小于adv_loss_stop 认为满足需要了 退出
        adv_loss_stop=0.001
        #loss阈值 衡量前后两次loss差别过小 认为已经稳定了 收敛了 连续loss_cnt_threshold次小于loss_limit退出
        loss_limit = adv_loss_stop/10
        loss_cnt_threshold = 10
        #最大迭代次数
        num_iter = 2000

        last_adv_loss = 0
        cnt = 0
        flag = False

        for i in range(num_iter):
            feed_dict = {
                images_placeholder: adv_image,
                phase_train_placeholder: False
            }
            #调用迭代器
            adv_image, adv_loss = self.sess.run([adv_x, loss], feed_dict=feed_dict)

            print('[%d] Bias from original image: %2.6f Loss: %2.6f' % (i, np.sqrt(
                np.sum(np.square(adv_image[0, ...] - input_image))), adv_loss))

            if np.absolute(adv_loss - last_adv_loss) < loss_limit:
                if flag:
                    cnt += 1
                else:
                    flag = True
            else:
                flag = False
                cnt = 0

            if cnt == loss_cnt_threshold:
                print('always get too tiny loss...')
                break

            if adv_loss < adv_loss_stop:
                print('get final result...')
                break

            last_adv_loss = adv_loss


        if i == num_iter - 1:
            print('Out of maximum iterative number...')
        #filename = generate_inp2adv_name(input_pic, target_pic) + str(i)
        #调试阶段 文件名不随机
        filename = "output/"+generate_inp2adv_name(input_pic, target_pic)
        save_img2png(adv_image[0, ...], filename)

        feed_dict = {
            images_placeholder: adv_image,
            phase_train_placeholder: False
        }
        adv_embedding = self.sess.run(embeddings, feed_dict=feed_dict)[0]
        print('The distance between input embedding and target is %2.6f' %
              Euclidian_distance(adv_embedding, target_emb))


def batch_generate_adv_whitebox():
    import glob
    Bill_Gates_list=glob.glob("Bill_Gates/*.png")
    Michael_Jordan_list=glob.glob("Michael_Jordan/*.png")

    fr = FacenetFR()

    for a in Bill_Gates_list:
        for b in Michael_Jordan_list:
            fr.generate_adv_whitebox(a, b)


if __name__ == '__main__':
    #fr = FacenetFR()

    #input_pic = "Bill_Gates_0001.png"
    #target_pic = "Michael_Jordan_0002.png"
    # print fr.compare(input_pic,target_pic)

    #fr.generate_adv_whitebox(input_pic, target_pic)

    #批量生成
    batch_generate_adv_whitebox()
