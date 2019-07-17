
# coding: utf-8

# 演示如何批量调用API检测是否换脸

# In[1]:

import requests
import cv2
import os
import shutil
import glob
import argparse


# 解析命令行参数

parser = argparse.ArgumentParser(description = 'Benchmarking deapfake imgs')
parser.add_argument('--in_dir', default = 'IN',help="Raw imgs dir")
parser.add_argument('--out_dir', default = 'OUT',help="Where to save deepfake imgs")
#当得分低于threshold 将会视为换脸图片
parser.add_argument('--threshold', type = float, default = 0.2)

args = parser.parse_args()
print(args)

# 根据返回的假脸的坐标绘制位置

# In[2]:

def draw_face(path,face_list=[],p=0.2):
    
    deepfakes_num=0
    
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    for i in range(len(face_list)):
        score=face_list[i]["score"]
        
        #print(score)
            
        if float(score) <= p:
            left=int(face_list[i]["location"]["left"]) 
            top=int(face_list[i]["location"]["top"])
            width=int(face_list[i]["location"]["width"])
            height=int(face_list[i]["location"]["height"])
            
            cv2.rectangle(img, (left,top), (left+width,top+height), (0,255,0), 4)
            
            deepfakes_num+=1
            
    cv2.imwrite(path, img)
       
# In[3]:


def deepfakes_detect_by_img(path):
    url="http://gwgp-h9xcyrmorux.n.bdcloudapi.com/rest/deepfakes/v1/detect"
    
    files={"file":( path, open(path,"rb") ,"image/jpeg",{})}
     
    res=requests.request("POST",url, data={"type":1}, files=files)
    
    face_num=0
    face_list=[]
        
    try:
    
        face_num=res.json()["face_num"]

        if face_num > 0:
            face_list=res.json()["face_list"]
            #draw_face(path,face_list,p=0.2)
            
    except:
        print("Fail to detect!")
        face_num=0
        face_list=[]
    
    return face_num,face_list


# # 批量检测指定目录下的图片

# In[4]:



#原始爬取疑似换脸图片的目录
deepfakes_raw_dir=args.in_dir

if not os.path.exists(deepfakes_raw_dir):
    print("No files found!")
    


#保存换脸图片的目录
deepfakes_dir=args.out_dir

if not os.path.exists(deepfakes_dir):
    os.mkdir(deepfakes_dir)


    
for maindir, subdir, file_name_list in os.walk(deepfakes_raw_dir):

    for filename in file_name_list:
        filename = os.path.join(maindir, filename)#合并成一个完整路径
        
        face_num,face_list=deepfakes_detect_by_img(filename)

        deepfakes=0

        for i in range(face_num):
            score=face_list[i]["score"]

            #score小于0.2的认为是假脸
            if float(score) <= args.threshold:
                deepfakes+=1

                if deepfakes_dir is not None:
                    copy_filename="{}/{}".format(deepfakes_dir,os.path.basename(filename))
                    shutil.copyfile(filename,copy_filename)

        if deepfakes > 0:
            print("检测图片{}，其中检测到人脸{}个，疑似假脸{}个".format(filename,face_num,deepfakes))

