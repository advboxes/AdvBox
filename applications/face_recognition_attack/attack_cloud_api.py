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

import configparser

cf = configparser.ConfigParser()
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



if __name__ == '__main__':
    test_t("Bill_Gates_0001_2_Michael_Jordan_0002.png","Bill_Gates_0001.png")
    test_t("Bill_Gates_0001_2_Michael_Jordan_0002.png", "Michael_Jordan_0002.png")