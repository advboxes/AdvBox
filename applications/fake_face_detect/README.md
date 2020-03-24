Fake Face Detect
===
检测假脸的restful API，用于检测图片中人脸是否是假脸。目前包含两种假脸检测：     
1. deepfakes (AI换脸)  
2. face merge（人脸融合）     
    

对于视频中假脸，可以提取视频帧进行检测。例如使用ffmpeg进行提取。
```
ffmpeg -i /path/to/my/video.mp4 /path/to/output/video-frame-%d.png
```

### 请求示例

+ Deepfakes检测
```shell
curl "http://gwgp-h9xcyrmorux.n.bdcloudapi.com/rest/deepfakes/v1/detect?access_token=github"  -F "file=@fake_deepfakes.jpg" 
```

+ 人脸融合检测
```shell
curl "http://gwgp-h9xcyrmorux.n.bdcloudapi.com/rest/facemerge/v1/detect?access_token=github"  -F "file=@fake_merging.jpg" 
```

# 接口描述
----
## AI换脸检测
本接口用于检测Deepfakes生成的换脸图片。

### 接口请求
##### 请求说明
+ API服务地址： **http://gwgp-h9xcyrmorux.n.bdcloudapi.com/rest/deepfakes/v1/detect**  
+ 请求使用HTTP-POST协议，图片通过消息体传递，参数为“file”。
+ 图片格式：现支持PNG、JPG、JPEG、BMP，不支持GIF图片

### 接口返回

#### 返回值说明       

名称 | 必选 | 类型| 说明
------------ | ------------- | ------------| ------------
req_id | 是  | str| 请求id
err_code | 是  | int| 错误码
err_msg | 是  | str| 错误提示信息
face_num | 是  | int| 图片中的人脸数量
face_list | 是  | array| 人脸信息列表
+location | 是  | json| 人脸在图片中的位置
++left | 是  | float| 人脸区域离左边界的距离
++top | 是  | float| 人脸区域离上边界的距离
++width | 是  | float| 人脸区域的宽度
++height | 是  | float| 人脸区域的高度
+score | 是  | float| [0-1]的置信度，1表示不是假脸，0表示是假脸，越接近0表明是假脸概率越大

#### 返回示例        
```json
{"req_id": "1234", "err_code": 0, "face_num": 1, "face_list": [{"score": "0.704", "location": {"width": 134.661116912961, "height": 164.4973850734532, "left": 323.52674858272076, "top": 45.93883013725281}},{"score": "0.940", "location": {"width": 98.42351754754782, "height": 124.57625658810139, "left": 89.07363281399012, "top": 72.44846796244383}}], "err_msg": "success"}

```
#### 错误码说明       

错误码 | 描述 | 处理建议
------------ | ------------- | ------------
0 | 成功 | 无
100 | token失效 | 获取新token
101 | 请求未携带文件 | post带上文件

---
## 人脸融合图片检测
本接口用于检测百度人脸融合服务生成的假脸图片。     
API服务地址： **http://gwgp-h9xcyrmorux.n.bdcloudapi.com/rest/facemerge/v1/detect**   
接口请求与返回与上一个接口类似。

# 示例图片

我们从互联网上搜集了换脸的示例图片，在文件夹 [demo](demo)下

![换脸示例1](demo/deepfake01.png)
![换脸示例2](demo/deepfake02.png)
![换脸示例3](demo/deepfake03.png)

python调用示例文件为[api_demo.ipynb](api_demo.ipynb)
