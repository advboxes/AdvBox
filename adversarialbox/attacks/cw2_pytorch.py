"""
This module provide the attack method of "CW".
L2 distance metrics especially
"""
from __future__ import division
from __future__ import print_function

from builtins import range
import logging
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.utils.data.dataloader as Data
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
from .base import Attack

__all__ = ['CW_L2_Attack', 'CW_L2']


class CW_L2_Attack(Attack):
    """
    Uses Adam to minimize the CW L2 objective function

    Paper link: https://arxiv.org/abs/1608.04644
    """
    def __init__(self,model):
        super(CW_L2_Attack, self).__init__(model)
        
        self._model=model._model
        mean, std = model._preprocess
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)
        

    def _apply(self,
               adversary,
               max_iterations=1000,
               learning_rate=0.01,
               initial_const=10.0,
               binary_search_steps=10,
               k=40,
               num_labels=10):
        

        self._adversary = adversary
        img = self._adversary.original.copy()
        pre_label = adversary.original_label
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        #定向
        if adversary.is_targeted_attack:
            #攻击目标标签 必须使用one hot编码
            target_label=adversary.target_label
            tlab=Variable(torch.from_numpy(np.eye(num_labels)[target_label]).to(device).float())
        #无定向
        else:
            #攻击目标标签 必须使用one hot编码
            tlab=Variable(torch.from_numpy(np.eye(num_labels)[pre_label]).to(device).float()) 
        boxmin, boxmax = self.model.bounds()   
        #print("boxmin={}, boxmax={}".format(boxmin, boxmax))   
        logging.info("boxmin={}, boxmax={}".format(boxmin, boxmax)) 
        #print(tlab)  
        shape=adversary.original.shape  
        #c的初始化边界
        lower_bound = 0
        confidence=initial_const
        upper_bound = 1e10
        # the best l2, score, and image attack
        o_bestl2 = 1e10
        o_bestscore = -1
        o_bestattack = [np.zeros(shape)]   
        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        boxmul = (boxmax - boxmin) / 2.
        boxplus = (boxmin + boxmax) / 2.
        
        for outer_step in range(binary_search_steps):
            #print("o_bestl2={} confidence={}".format(o_bestl2,confidence)  )
            logging.info("o_bestl2={} confidence={}".format(o_bestl2,confidence) )
            #把原始图像转换成图像数据和扰动的形态
            timg = Variable(torch.from_numpy(np.arctanh((img - boxplus) / boxmul * 0.999999)).to(device).float())
            modifier=Variable(torch.zeros_like(timg).to(device).float())
            #图像数据的扰动量梯度可以获取
            modifier.requires_grad = True
            #定义优化器 仅优化modifier
            optimizer = torch.optim.Adam([modifier],lr=learning_rate)
            
            for iteration in range(1,max_iterations+1):
                optimizer.zero_grad()
                #定义新输入
                newimg = torch.tanh(modifier + timg) * boxmul + boxplus
                output=self._model((newimg - self.mean) / self.std) 
                #定义cw中的损失函数
                loss2=torch.dist(newimg,(torch.tanh(timg) * boxmul + boxplus),p=2)
                real=torch.max(output*tlab)
                other=torch.max((1-tlab)*output)  
                if adversary.is_targeted_attack: 
                    loss1=other-real+k      
                else:
                    loss1=-other+real+k 
                    #loss1=other-real+k    
                loss1=torch.clamp(loss1,min=0)
                loss1=confidence*loss1
                loss=loss1+loss2
                loss.backward(retain_graph=True)
                optimizer.step()
                l2=loss2
                sc=output.data.cpu().numpy()
                #输出的是概率
                #pro=F.softmax(self._model(newimg),dim=1)[0].data.cpu().numpy()[target_label]
                pred=np.argmax(sc)
                if iteration%(max_iterations//10) == 0:
                    #print("iteration={} loss={} loss1={} loss2={} pred={}".format(iteration,loss,loss1,loss2,pred))  
                    logging.info("iteration={} loss={} loss1={} loss2={} pred={}".format(iteration,loss,loss1,loss2,pred))
                if adversary.is_targeted_attack:
                    if (l2 < o_bestl2) and (np.argmax(sc) == target_label ):
                        #print("attack success l2={} target_label={} pro={}".format(l2,target_label,pro))
                        print("attack success l2={} target_label={}".format(l2,target_label))
                        o_bestl2 = l2
                        o_bestscore = pred
                        o_bestattack = newimg.data.cpu().numpy()
                else:
                    if (l2 < o_bestl2) and (pred != pre_label ):
                        #print("attack success l2={} target_label={} pro={}".format(l2,target_label,pro))
                        #print("attack success l2={} label={}".format(l2,pred))
                        logging.info("attack success l2={} label={}".format(l2,pred))
                        o_bestl2 = l2
                        o_bestscore = pred
                        o_bestattack = newimg.data.cpu().numpy()

            confidence_old=-1        
            
            if adversary.is_targeted_attack:

                if (o_bestscore == target_label) and (o_bestscore != -1):
                    #攻击成功 减小c
                    upper_bound = min(upper_bound,confidence)
                    if upper_bound < 1e9:
                            print()
                            confidence_old=confidence
                            confidence = (lower_bound + upper_bound)/2
                else:
                    lower_bound = max(lower_bound,confidence)
                    confidence_old=confidence
                    if upper_bound < 1e9:
                            confidence = (lower_bound + upper_bound)/2
                    else:
                            confidence *= 10

            else:
                if (o_bestscore != pre_label) and (o_bestscore != -1):
                    #攻击成功 减小c
                    upper_bound = min(upper_bound,confidence)
                    if upper_bound < 1e9:
                            confidence_old=confidence
                            confidence = (lower_bound + upper_bound)/2
                else:
                    lower_bound = max(lower_bound,confidence)
                    confidence_old=confidence
                    if upper_bound < 1e9:
                            confidence = (lower_bound + upper_bound)/2
                    else:
                            confidence *= 10
            
            #print("outer_step={} confidence {}->{}".format(outer_step,confidence_old,confidence))   
            logging.info("outer_step={} confidence {}->{}".format(outer_step,confidence_old,confidence))
        #print(o_bestattack)
    
        if o_bestscore != -1:
            if adversary.try_accept_the_example(o_bestattack, o_bestscore):
                    return adversary

        return adversary

    

CW_L2 = CW_L2_Attack

