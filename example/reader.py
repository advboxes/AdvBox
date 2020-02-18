from __future__ import division
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

from past.utils import old_div
import os
import random
import functools
import numpy as np
import paddle
from PIL import Image, ImageEnhance

random.seed(0)
DATA_DIM = 224
THREAD = 1
BUF_SIZE = 102400

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = old_div((width - size), 2)
        h_start = old_div((height - size), 2)
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def process_image(sample):
    img_path = sample[0]

    img = Image.open(img_path)
    if img.size[0] != img.size[1] or img.size[0] != DATA_DIM:
        img = resize_short(img, target_size=256)
        img = crop_image(img, target_size=DATA_DIM, center=True)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = old_div(np.array(img).astype('float32').transpose((2, 0, 1)), 255)
    img -= img_mean
    img /= img_std
    return [img], img_path


def _reader_creator(file_list,
                    data_path,
                    shuffle=False):
    def reader():
        with open(file_list) as flist:
            lines = [line.strip() for line in flist]
            if shuffle:
                random.shuffle(lines)
            for line in lines:
                img_path = os.path.join(data_path, line)
                yield [img_path]

    mapper = functools.partial(process_image)
    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def test(file_list, data_path):
    return _reader_creator(file_list, data_path, shuffle=False)
