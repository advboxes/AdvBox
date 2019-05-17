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


#pip install qcloud_image

import ConfigParser

cf = ConfigParser.ConfigParser()
cf.read("../../conf/api.conf")


def test_t(a,b):
    from qcloud_image import Client
    from qcloud_image import CIUrl, CIFile, CIBuffer, CIUrls, CIFiles, CIBuffers
    appid = cf.get("qq", "appid")
    secret_id = cf.get("qq", "secret_id")
    secret_key = cf.get("qq", "secret_key")
    bucket = 'BUCKET'
    client = Client(appid, secret_id, secret_key, bucket)
    client.use_http()
    client.set_timeout(30)

    print("compare {} and {}".format(a,b))

    print(client.face_compare(CIFile(a), CIFile(b)))


def test_xf(a,b):
    import requests
    import time
    import json
    import hashlib
    import base64

    x_appid = cf.get("xf", "appid")
    api_key = cf.get("xf", "api_key")
    url = 'http://api.xfyun.cn/v1/service/v1/image_identify/face_verification'
    x_time = str(int(time.time()))
    param = {'auto_rotate': True}
    param = json.dumps(param)
    x_param = base64.b64encode(param.encode('utf-8'))
    m2 = hashlib.md5()
    #m2.update(str(api_key + x_time + str(x_param, 'utf-8')).encode('utf-8'))
    m2.update( str(api_key + x_time + str(x_param).encode('utf-8')))
    x_checksum = m2.hexdigest()
    x_header = {
        'X-Appid': x_appid,
        'X-CurTime': x_time,
        'X-CheckSum': x_checksum,
        'X-Param': x_param,
    }
    with open(a, 'rb') as f:
        f1 = f.read()
    with open(b, 'rb') as f:
        f2 = f.read()
    #f1_base64 = str(base64.b64encode(f1), 'utf-8')
    #f2_base64 = str(base64.b64encode(f2), 'utf-8')
    f1_base64 = str(base64.b64encode(f1))
    f2_base64 = str(base64.b64encode(f2))
    data = {
        'first_image': f1_base64,
        'second_image': f2_base64,
    }
    req = requests.post(url, data=data, headers=x_header)
    result = req.content
    print(result)

def batch_test_t():
    import glob
    import time

    Bill_Gates_list=glob.glob("Bill_Gates/*.png")
    Michael_Jordan_list=glob.glob("Michael_Jordan/*.png")
    adv_list=glob.glob("output/*.png")

    for a in Michael_Jordan_list:
        for b in adv_list:
            time.sleep(1)  # 休眠1秒
            test_t(a, b)


def batch_test_xf():
    import glob
    import time

    Bill_Gates_list=glob.glob("Bill_Gates/*.png")
    Michael_Jordan_list=glob.glob("Michael_Jordan/*.png")
    adv_list=glob.glob("output/*.png")

    for a in Michael_Jordan_list:
        for b in adv_list:
            time.sleep(1)  # 休眠1秒
            test_xf(a, b)


if __name__ == '__main__':
    #test_t("Bill_Gates_0001_2_Michael_Jordan_0002.png","Bill_Gates_0001.png")
    #test_t("Bill_Gates_0001_2_Michael_Jordan_0002.png", "Michael_Jordan_0002.png")

    #批量测试t
    #batch_test_t()

    #批量测试xf
    #batch_test_xf()
    import sys

    input_pic=sys.argv[1]
    target_pic=sys.argv[2]

    print("input_pic={} target_pic={}".format(input_pic,target_pic))

    print("Test QQ:")
    test_t(input_pic, target_pic)

    print("Test XF:")
    test_xf(input_pic, target_pic)