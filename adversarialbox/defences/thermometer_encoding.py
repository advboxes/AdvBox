"""
This module provide the defence method for THERMOMETER ENCODING's implement.

THERMOMETER ENCODING: ONE HOT WAY TO RESIST ADVERSARIAL EXAMPLES


"""
from builtins import range
import logging
logger=logging.getLogger(__name__)

import numpy as np
from keras.utils import to_categorical

__all__ = [
    'ThermometerEncodingDefence'
]

def _perchannel(x,num_space):    
    pos = np.zeros(shape=x.shape)
    for i in range(1, num_space):
        pos[x > float(i) / num_space] += 1

    onehot_rep = to_categorical(pos.reshape(-1), num_space)

    for i in reversed(list(range(1, num_space))):
        onehot_rep[:, i] += np.sum(onehot_rep[:, :i], axis=1)

            
    result = onehot_rep.reshape(list(x.shape) + [num_space])

    return result


#num_space=10为 一般为10
#clip_values为最终处理后取值范围 可能包含负数  常见的为[0,1] [-1,1]
# 支持的格式为[28,28,1]
def ThermometerEncodingDefence(x, y=None, num_space=10, clip_values=(0.0, 1.0)):
    result = []
    
    #for c in range(x.shape[-1]):
    #    result.append(_perchannel(x[:, :, :, c],num_space))
    for c in range(x.shape[1]):
        result.append(_perchannel(x[:, c, :, :],num_space))

    result = np.concatenate(result, axis=3)
    result = np.clip(result, clip_values[0], clip_values[1])


    return result


