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
def LabelSmoothingDefence(y, smoothing=0.9):

    assert ( smoothing > 0 ) and ( smoothing < 1)

    y -= smoothing * (y - 1. / y.shape[0])

    return y


