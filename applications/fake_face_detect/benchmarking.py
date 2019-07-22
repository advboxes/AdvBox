
# coding: utf-8

# 演示如何批量调用API检测是否换脸

# In[1]:

import requests
import cv2
import os
import shutil
import glob
import argparse
import time

# 解析命令行参数

parser = argparse.ArgumentParser(description = 'Benchmarking deapfake imgs')
parser.add_argument('--in_dir', default = 'IN',help="Raw imgs dir")
parser.add_argument('--out_dir', default = 'OUT',help="Where to save deepfake imgs")
#当得分低于threshold 将会视为换脸图片
parser.add_argument('--threshold', type = float, default = 0.2)
parser.add_argument('--draw', action = 'store_true',help="Draw fake face on img")
parser.add_argument('--debug', action = 'store_true',help="Debug model")
parser.add_argument('--nosleep', action = 'store_true',help="No sleep")

args = parser.parse_args()
print(args)

# 根据返回的假脸的坐标绘制位置

# In[2]:

#记录查询总数和失败个数
g_query_sum=0
g_query_fail=0
g_query_deepfakes=0

#记录开始时间
g_starttime = time.time()


def draw_face(path,face_list=[],p=0.2):
    
    deepfakes_num=0
    
    #img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.imread(path)
    
    for i in range(len(face_list)):
        score=face_list[i]["score"]
        
        #print(score)
            
        if float(score) <= p:
            left=int(face_list[i]["location"]["left"]) 
            top=int(face_list[i]["location"]["top"])
            width=int(face_list[i]["location"]["width"])
            height=int(face_list[i]["location"]["height"])
            
            cv2.rectangle(img, (left,top), (left+width,top+height), (0,255,0), 4)
            cv2.putText(img,"score={}".format(score),
                        ( int(left),int(top-height/5)),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2,1)
            
            deepfakes_num+=1
            
    cv2.imwrite(path, img)
       
# In[3]:


def deepfakes_detect_by_img(path):
    
    global g_query_fail
    
    url="http://gwgp-h9xcyrmorux.n.bdcloudapi.com/rest/deepfakes/v1/detect"
   
    files={"file":( path, open(path,"rb") ,"image/jpeg",{})}
    
    
    res=requests.request("POST",url, data={"type":1}, files=files)
    
    if args.debug:
        print(res)
        print(res.text)
        print(res.headers)
    
    face_num=0
    face_list=[]
        
    try:
    
        face_num=res.json()["face_num"]

        if face_num > 0:
            face_list=res.json()["face_list"]
            #draw_face(path,face_list,p=0.2)
            
    except:
        print("Fail to detect {}!".format(path))
        face_num=0
        face_list=[]
        g_query_fail+=1
        
    
    
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
        g_query_sum+=1
        filename = os.path.join(maindir, filename)#合并成一个完整路径
        
        if not  args.nosleep:
            time.sleep(1)
        
        starttime = time.time()
        face_num,face_list=deepfakes_detect_by_img(filename)
        endtime = time.time()
        dtime = endtime - starttime
        if args.debug:
            print("{}/(size={}K) cost {}s".format(os.path.basename(filename),os.path.getsize(filename)/1000,dtime))
        

        deepfakes=0

        for i in range(face_num):
            score=face_list[i]["score"]

            #score小于0.2的认为是假脸
            if float(score) <= args.threshold:
                deepfakes+=1

                if deepfakes_dir is not None:
                    copy_filename="{}/{}".format(deepfakes_dir,os.path.basename(filename))
                    shutil.copyfile(filename,copy_filename)
                    #画出假脸坐标
                    if args.draw:
                        draw_face(copy_filename,face_list,p=args.threshold)
                    
                    

        if deepfakes > 0:
            print("检测图片{}，其中检测到人脸{}个，疑似假脸{}个".format(filename,face_num,deepfakes))
            g_query_deepfakes+=1
#总结 

g_endtime = time.time()
g_costtime=g_endtime-g_starttime


if args.debug:
    print("查询总数{} 失败率为{}% 疑似具有假脸的图片个数{} 保存在目录{} 总耗时{}s 平均{}s".
          format(g_query_sum,100.0*g_query_fail/g_query_sum,g_query_deepfakes,deepfakes_dir,g_costtime,g_costtime/g_query_sum))
         

