#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import range
import os
import cv2
import time
import sys
import math
import numpy as np
import argparse
import functools

import paddle
import paddle.fluid as fluid

import six

from PIL import Image, ImageOps
#绘图函数
import matplotlib
#服务器环境设置
import matplotlib.pyplot as plt


#打印prog
def print_prog(prog):
    for name, value in sorted(six.iteritems(prog.block(0).vars)):
        print(value)
    for op in prog.block(0).ops:
        print("op type is {}".format(op.type))
        print("op inputs are {}".format(op.input_arg_names))
        print("op outputs are {}".format(op.output_arg_names))
        for key, value in sorted(six.iteritems(op.all_attrs())):
            if key not in ['op_callstack', 'op_role_var']:
                print(" [ attrs: {}:   {} ]".format(key, value))

#去除batch_norm的影响
def init_prog(prog):
    for op in prog.block(0).ops:
        #print("op type is {}".format(op.type))
        if op.type in ["batch_norm"]:
            # 兼容旧版本 paddle
            if hasattr(op, 'set_attr'):
                op.set_attr('is_test', False)
                op.set_attr('use_global_stats', True)
            else:
                op._set_attr('is_test', False)
                op._set_attr('use_global_stats', True)
                op.desc.check_attrs()

def img2tensor(img,image_shape=[3,224,224]):
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
      
    img = cv2.resize(img,(image_shape[1],image_shape[2]))

    #RGB img [224,224,3]->[3,224,224]
    img = img.astype('float32').transpose((2, 0, 1)) / 255
     
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    img=img.astype('float32')
    img=np.expand_dims(img, axis=0)
    
    return img

def process_img(img_path="",image_shape=[3,224,224]):
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
      
    img = cv2.imread(img_path)
    img = cv2.resize(img,(image_shape[1],image_shape[2]))
    
    #RBG img [224,224,3]->[3,224,224]
    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    img=img.astype('float32')
    img=np.expand_dims(img, axis=0)
    
    return img

def tensor2img(tensor):
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
    
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    
    img=tensor.copy()
      
    img *= img_std
    img += img_mean
    
    img = np.round(img*255) 
    img = np.clip(img,0,255)

    img=img[0].astype(np.uint8)
        
    img = img.transpose(1, 2, 0)
    
    return img

#对比展现原始图片和对抗样本图片
def show_images_diff(original_img,adversarial_img):
    #original_img = np.array(Image.open(original_img))
    #adversarial_img = np.array(Image.open(adversarial_img))
    original_img=cv2.resize(original_img.copy(),(224,224))
    adversarial_img=cv2.resize(adversarial_img.copy(),(224,224))

    plt.figure(figsize=(10,10))

    #original_img=original_img/255.0
    #adversarial_img=adversarial_img/255.0

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial Image')
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference')
    difference = 0.0+adversarial_img - original_img
        
    l0 = np.where(difference != 0)[0].shape[0]*100/(224*224*3)
    l2 = np.linalg.norm(difference)/(256*3)
    linf=np.linalg.norm(difference.copy().ravel(),ord=np.inf)
    # print(difference)
    print("l0={}% l2={} linf={}".format(l0, l2,linf))
    
    #(-1,1)  -> (0,1)
    #灰色打底 容易看出区别
    difference=difference/255.0
        
    difference=difference/2.0+0.5
   
    plt.imshow(difference)
    plt.axis('off')

    plt.show()
    

    #plt.savefig('fig_cat.png')
    
    
#实现linf约束 输入格式都是tensor 返回也是tensor [1,3,224,224]
def linf_img_tenosr(o,adv,epsilon=16.0/256):
    
    o_img=tensor2img(o)
    adv_img=tensor2img(adv)
    
    clip_max=np.clip(o_img*(1.0+epsilon),0,255)
    clip_min=np.clip(o_img*(1.0-epsilon),0,255)
    
    adv_img=np.clip(adv_img,clip_min,clip_max)
    
    adv_img=img2tensor(adv_img)
    
    return adv_img
"""
Explaining and Harnessing Adversarial Examples, I. Goodfellow et al., ICLR 2015
实现了FGSM 支持定向和非定向攻击的单步FGSM


input_layer:输入层
output_layer:输出层
step_size:攻击步长
adv_program：生成对抗样本的prog 
eval_program:预测用的prog
isTarget：是否定向攻击
target_label：定向攻击标签
epsilon:约束linf大小
o:原始数据
use_gpu：是否使用GPU

返回：
生成的对抗样本
"""
def FGSM(adv_program,eval_program,gradients,o,input_layer,output_layer,step_size=16.0/256,epsilon=16.0/256,isTarget=False,target_label=0,use_gpu=False):
    
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
   
    result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={ input_layer.name:o })
    result = result[0][0]
   
    o_label = np.argsort(result)[::-1][:1][0]
    
    if not isTarget:
        #无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label=o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label,o_label))
        
        
    target_label=np.array([target_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    
    #计算梯度
    g = exe.run(adv_program,
                     fetch_list=[gradients],
                     feed={ input_layer.name:o,'label': target_label  }
               )
    g = g[0][0]
    
    #print(g)
    
    if isTarget:
        adv=o-np.sign(g)*step_size
    else:
        adv=o+np.sign(g)*step_size
    
    #实施linf约束
    adv=linf_img_tenosr(o,adv,epsilon)
    
    return adv


"""
Towards deep learning models resistant to adversarial attacks, A. Madry, A. Makelov, L. Schmidt, D. Tsipras, 
and A. Vladu, ICLR 2018
实现了PGD 支持定向和非定向攻击的PGD


input_layer:输入层
output_layer:输出层
step_size:攻击步长
adv_program：生成对抗样本的prog 
eval_program:预测用的prog
isTarget：是否定向攻击
target_label：定向攻击标签
epsilon:约束linf大小
o:原始数据
use_gpu：是否使用GPU

返回：
生成的对抗样本
"""
def PGD(adv_program,eval_program,gradients,o,input_layer,output_layer,step_size=2.0/256,epsilon=16.0/256,iteration=20,isTarget=False,target_label=0,use_gpu=False):
    
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
   
    result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={ input_layer.name:o })
    result = result[0][0]
   
    o_label = np.argsort(result)[::-1][:1][0]
    
    if not isTarget:
        #无定向攻击 target_label的值自动设置为原标签的值
        print("Non-Targeted attack target_label=o_label={}".format(o_label))
        target_label=o_label
    else:
        print("Targeted attack target_label={} o_label={}".format(target_label,o_label))
        
        
    target_label=np.array([target_label]).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    
    adv=o.copy()
    
    for _ in range(iteration):
    
        #计算梯度
        g = exe.run(adv_program,
                         fetch_list=[gradients],
                         feed={ input_layer.name:adv,'label': target_label  }
                   )
        g = g[0][0]

        if isTarget:
            adv=adv-np.sign(g)*step_size
        else:
            adv=adv+np.sign(g)*step_size
    
    #实施linf约束
    adv=linf_img_tenosr(o,adv,epsilon)
    
    return adv