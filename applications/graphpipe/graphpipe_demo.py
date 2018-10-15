#coding=utf-8


#graphpipe官网的一个例子
#https://oracle.github.io/graphpipe/#/guide/user-guide/quickstart

#pip install graphpipe
#pip install pillow # needed for image manipulation



'''
#服务器端启动方式为：
docker run -it --rm \
    -e https_proxy=${https_proxy} \
    -p 9000:9000 \
    sleepsonthefloor/graphpipe-tf:cpu \
    --model=https://oracle.github.io/graphpipe/models/squeezenet.pb \
    --listen=0.0.0.0:9000
'''

from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import requests

from graphpipe import remote

data = np.array(Image.open("mug227.png"))
data = data.reshape([1] + list(data.shape))
data = np.rollaxis(data, 3, 1).astype(np.float32)  # channels first
print(data.shape)

pred = remote.execute("http://127.0.0.1:9000", data)
print("Expected 504 (Coffee mug), got: {}".format(np.argmax(pred, axis=1)))