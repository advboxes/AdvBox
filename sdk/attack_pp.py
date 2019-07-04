#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

def process_img(img_path="",image_shape=[3,224,224]):
    
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225] 
      
    img = cv2.imread(img_path)
    img = cv2.resize(img,(image_shape[1],image_shape[2]))

    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    img=img.astype('float32')
    img=np.expand_dims(img, axis=0)
    
    return img


"""
Explaining and Harnessing Adversarial Examples, I. Goodfellow et al., ICLR 2015
实现了FGSM 支持定向和非定向攻击的单步FGSM


input_layer:输入层
output_layer:输出层
step_size:攻击步长
loss：损失函数 
isTarget：是否定向攻击
target_label：定向攻击标签
o:原始数据

返回：
生成的对抗样本
"""
def FGSM(o,input_layer,output_layer,step_size=16.0/256,loss="",isTarget=False,target_label=0,use_gpu=False):
    
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    #exe.run(fluid.default_startup_program())
    
    label = fluid.layers.data(name="label", shape=[1] ,dtype='int64')
    
    if not isTarget:
        #评估模式
        eval_program =   fluid.default_main_program().clone(for_test=True)
        
        result = exe.run(eval_program,
                     fetch_list=[output_layer],
                     feed={ input_layer.name:o })
        result = result[0][0]
        #无定向攻击 target_label的值自动设置为原标签的值
        target_label = np.argsort(result)[::-1][:1]
        
        
    target_label=np.array(target_label).astype('int64')
    target_label=np.expand_dims(target_label, axis=0)
    
    if loss == "":
        print("")
        loss = fluid.layers.cross_entropy(input=output_layer, label=label)

    #http://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/api_cn/backward_cn.html
    gradients = fluid.backward.gradients(targets=loss, inputs=[input_layer])[0]
    
    # 测试模式
    adv_program = fluid.default_main_program().clone(for_test=True)
    
    #设置特殊状态
    #init_prog(adv_program)
    for op in adv_program.block(0).ops:
        #print("op type is {}".format(op.type))
        if op.type in ["batch_norm"]:
            # 兼容旧版本 paddle
            if hasattr(op, 'set_attr'):
                op.set_attr('is_test', False)
                op.set_attr('use_global_stats', True)
            else:
                op._set_attr('is_test', False)
                op._set_attr('use_global_stats', True)

    #计算梯度
    g = exe.run(adv_program,
                     fetch_list=[gradients],
                     feed={ input_layer.name:o,
                            'label': target_label  })
    g = g[0][0]
    
    print(g)
    
    if isTarget:
        adv=o-np.sign(g)*step_size
    else:
        adv=o+np.sign(g)*step_size
    
    return adv