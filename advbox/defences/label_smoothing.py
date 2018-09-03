#coding=utf-8

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
"""
This module provide the defence method for SpatialSmoothingDefence's implement.

Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks


"""
import logging
logger=logging.getLogger(__name__)

import numpy as np


__all__ = [
    'LabelSmoothingDefence'
]


#Perturbation, Optimization and Statistics
def LabelSmoothingDefence(y, max_value=0.9):

    min_value = (1 - max_value) / (y.shape[0] - 1)
    assert max_value >= min_value

    res = y.copy()
    res[res == 1.] = max_value
    res[res == 0.] = min_value

    return res


