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


import logging
logging.basicConfig(level=logging.INFO,format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger=logging.getLogger(__name__)


#pip install graphpipe
#pip install pillow # needed for image manipulation



'''
#服务器端启动方式为：
docker run -it --rm \
      -e https_proxy=${https_proxy} \
      -p 9000:9000 \
      sleepsonthefloor/graphpipe-onnx:cpu \
      --value-inputs=https://oracle.github.io/graphpipe/models/squeezenet.value_inputs.json \
      --model=https://oracle.github.io/graphpipe/models/squeezenet.onnx \
      --listen=0.0.0.0:9000


docker run -it --rm \
        -v "$PWD:/models/"  \
        -p 9000:9000 \
        sleepsonthefloor/graphpipe-onnx:cpu \
        --value-inputs=https://oracle.github.io/graphpipe/models/squeezenet.value_inputs.json \
        --model=/models/squeezenet.onnx \
        --listen=0.0.0.0:9000
'''

from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import requests

from graphpipe import remote

def main(image_path):

    print("image_path:{}".format(image_path))

    data = np.array(Image.open(image_path))
    data = data.reshape([1] + list(data.shape))
    data = np.rollaxis(data, 3, 1).astype(np.float32)  # channels first
    #print(data.shape)

    pred = remote.execute("http://127.0.0.1:9000", data)

    print(pred.shape)

    dims=pred.shape
    dim=np.max(dims)
    print(dim)

    pred=pred.reshape([1,dim])
    #pred = np.squeeze(pred)
    #print(pred)
    print(pred.shape)

    print("{}".format(np.argmax(pred, axis=1)))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])