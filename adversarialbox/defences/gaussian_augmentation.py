"""
This module provide the defence method for GaussianAugmentationDefence's implement.

Efficient Defenses Against Adversarial Attacks


"""
import logging
logger=logging.getLogger(__name__)

import numpy as np


__all__ = [
    'GaussianAugmentationDefence'
]


#Efficient Defenses Against Adversarial Attacks
#std为高斯分布的标准差 r为增加的训练数据的比例 范围为[0,1]
def GaussianAugmentationDefence(x,y, std,r):

    #强制拷贝
    x_raw=x.copy()
    y_raw=y

    size = int(x_raw.shape[0] * r)
    #随机选择指定数目的原始数据
    indices = np.random.randint(0, x_raw.shape[0], size=size)

    #叠加高斯噪声
    x_gad = np.random.normal(x_raw[indices], scale=std, size=(size,) + x_raw[indices].shape[1:])
    x_gad = np.vstack((x_raw, x_gad))

    y_gad = np.concatenate((y_raw, y_raw[indices]))
    return x_gad, y_gad



