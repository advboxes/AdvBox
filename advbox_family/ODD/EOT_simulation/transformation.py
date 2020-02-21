from __future__ import division
from __future__ import print_function
from past.utils import old_div
import numpy as np
import math
import EOT_simulation.eulerangles as eu

import cv2
import xmltodict
import pdb

HALF_SZ = 224
# HALF_SZ = 1512
SCALE = 224.0 / 1512.0

sample_6para_expand = []
sample_6para = [ \
# Random Sample Scope Define.

# # Sample Translation for Original Photo
[0,0,0,0,0,0], \
[5,0,0,0,0,0], \
[30,0,0,0,0,0], \
[60, 0,0,0,0,0], \
[100, 0,0,0,0,0], \
[200, 0,0,0,0,0], \
[300, 0,0,0,0,0], \

[0, 30,0,0,0,0], \
[0, 60,0,0,0,0], \
[0, 100,0,0,0,0], \

[0, 0,300,0,0,0], \
[0, 0,600,0,0,0], \
[0, 0,900,0,0,0], \

# sample success: when sticker move some step along X, picture go right,
# when move some step along Y, picture go up
# when move some step along Z, picture shrink


# Sample Rotation for Original Photo
# 几何中心旋转

[0, 0,700,old_div(-math.pi,8),0,0], \
[0, 0,700,old_div(-math.pi,9),0,0], \
[0, 0,700,old_div(-math.pi,10),0,0], \
[0, 0,700,old_div(-math.pi,11),0,0], \
[0, 0,700,old_div(-math.pi,12),0,0], \

[0, 0,700,old_div(-math.pi,13),0,0], \
[0, 0,700,old_div(-math.pi,14),0,0], \
[0, 0,700,old_div(-math.pi,15),0,0], \
[0, 0,700,old_div(-math.pi,16),0,0], \
[0, 0,700,old_div(-math.pi,17),0,0], \
[0, 0,700,old_div(-math.pi,18),0,0], \
[0, 0,700,old_div(-math.pi,19),0,0], \
[0, 0,700,old_div(-math.pi,20),0,0], \
[0, 0,700,old_div(-math.pi,21),0,0], \
[0, 0,700,old_div(-math.pi,22),0,0], \
[0, 0,700,old_div(-math.pi,23),0,0], \
[0, 0,700,old_div(-math.pi,24),0,0], \
[0, 0,700,old_div(-math.pi,25),0,0], \
[0, 0,700,old_div(-math.pi,26),0,0], \
[0, 0,700,old_div(-math.pi,27),0,0], \
[0, 0,700,old_div(-math.pi,28),0,0], \
[0, 0,700,old_div(-math.pi,29),0,0], \
[0, 0,700,old_div(-math.pi,30),0,0], \
[0, 0,700,old_div(-math.pi,50),0,0], \
[0, 0,700,old_div(-math.pi,70),0,0], \
[0, 0,700,old_div(-math.pi,80),0,0], \
[0, 0,700,old_div(-math.pi,100),0,0], \
[0, 0,700,old_div(-math.pi,200),0,0], \
[0, 0,700,old_div(-math.pi,300),0,0], \

[0, 0,700,0,0,0], \
[0, 0,700,old_div(math.pi,8),0,0], \
[0, 0,700,old_div(math.pi,9),0,0], \
[0, 0,700,old_div(math.pi,10),0,0], \
[0, 0,700,old_div(math.pi,11),0,0], \
[0, 0,700,old_div(math.pi,12),0,0], \

[0, 0,700,old_div(math.pi,13),0,0], \
[0, 0,700,old_div(math.pi,14),0,0], \
[0, 0,700,old_div(math.pi,15),0,0], \
[0, 0,700,old_div(math.pi,16),0,0], \
[0, 0,700,old_div(math.pi,17),0,0], \
[0, 0,700,old_div(math.pi,18),0,0], \
[0, 0,700,old_div(math.pi,19),0,0], \
[0, 0,700,old_div(math.pi,20),0,0], \
[0, 0,700,old_div(math.pi,21),0,0], \
[0, 0,700,old_div(math.pi,22),0,0], \
[0, 0,700,old_div(math.pi,23),0,0], \
[0, 0,700,old_div(math.pi,24),0,0], \
[0, 0,700,old_div(math.pi,25),0,0], \
[0, 0,700,old_div(math.pi,26),0,0], \
[0, 0,700,old_div(math.pi,27),0,0], \
[0, 0,700,old_div(math.pi,28),0,0], \
[0, 0,700,old_div(math.pi,29),0,0], \
[0, 0,700,old_div(math.pi,30),0,0], \
[0, 0,700,old_div(math.pi,50),0,0], \
[0, 0,700,old_div(math.pi,70),0,0], \
[0, 0,700,old_div(math.pi,80),0,0], \
[0, 0,700,old_div(math.pi,100),0,0], \
[0, 0,700,old_div(math.pi,200),0,0], \
[0, 0,700,old_div(math.pi,300),0,0], \


# 水平角旋转
[0, 0,700,0,old_div(-math.pi,8),0], \
[0, 0,700,0,old_div(-math.pi,9),0], \
[0, 0,700,0,old_div(-math.pi,10),0], \
[0, 0,700,0,old_div(-math.pi,11),0], \
[0, 0,700,0,old_div(-math.pi,12),0], \

[0, 0,700,0,0,0], \
[0, 0,700,0,old_div(math.pi,8),0], \
[0, 0,700,0,old_div(math.pi,9),0], \
[0, 0,700,0,old_div(math.pi,10),0], \
[0, 0,700,0,old_div(math.pi,11),0], \
[0, 0,700,0,old_div(math.pi,12),0], \


# [0, 0,900,0,math.pi/2,0], \

# 俯仰角旋转
[0, 0,700,0,0,old_div(-math.pi,8)], \
[0, 0,700,0,0,old_div(-math.pi,9)], \
[0, 0,700,0,0,old_div(-math.pi,10)], \
[0, 0,700,0,0,old_div(-math.pi,11)], \
[0, 0,700,0,0,old_div(-math.pi,12)], \

[0, 0,700,0,0,0], \
[0, 0,700,0,0,old_div(math.pi,8)], \
[0, 0,700,0,0,old_div(math.pi,9)], \
[0, 0,700,0,0,old_div(math.pi,10)], \
[0, 0,700,0,0,old_div(math.pi,11)], \
[0, 0,700,0,0,old_div(math.pi,12)], \

# [0, 0,900,0,0,math.pi/2], \

# sample success: when sticker rotate positively alpha around z-axis, the sticker on the image turns around z-axis unclockwise.
# when sticker rotate positively beta aound y-axis, the sticker on the image turns around y-axis negatively(mirror)
# when sticker rotate positively gamma around x-axis, the sticker on the image turns around x-axis negatively(mirror)

# analysis: actually for the rotation of a plane, it doesn't matter it is rotated clockwisely or unclockwisely.
# [0, 5, 0, 0, 0, 0] \
]

# write the 6 num -> Mx as a function and test it.
def transform6para(V, transx = 0, transy = 0, transz = 0, rotz = 0, roty = 0, rotx = 0):
    Mt = [transx, transy, transz]
    Mr = eu.euler2mat(rotz, roty, rotx)
    return np.dot(Mr, V +  Mt)


def target_sample(display = False):
    sample_matrixes = []
    img = cv2.imread("calibration_file/calibration.JPG")
    calib = cv2.imread("calibration_file/calibration.jpg")
    
    height, width, channels = img.shape
    

    f = open("calibration_file/calibration.xml")
    dic = xmltodict.parse(f.read())

    xmin = int(dic['annotation']['object']['bndbox']['xmin'])
    ymin = int(dic['annotation']['object']['bndbox']['ymin'])
    xmax = int(dic['annotation']['object']['bndbox']['xmax'])
    ymax = int(dic['annotation']['object']['bndbox']['ymax'])    

    x_f0 = old_div(- (xmax - xmin), 2) * SCALE
    y_f0 = old_div(- (ymax - ymin), 2) * SCALE

    x_0f0_1 = HALF_SZ + x_f0
    y_0f0_1 = HALF_SZ + y_f0

    x_0f0_2 = HALF_SZ - x_f0
    y_0f0_2 = HALF_SZ + y_f0

    x_0f0_3 = HALF_SZ + x_f0
    y_0f0_3 = HALF_SZ - y_f0

    x_0f0_4 = HALF_SZ - x_f0
    y_0f0_4 = HALF_SZ - y_f0


    pts1 = np.float32([[x_0f0_1,y_0f0_1],[x_0f0_2,y_0f0_2],[x_0f0_3,y_0f0_3], [x_0f0_4,y_0f0_4]])

    # get camera focal length f (unit pixel) with A4 paper parameter.
    f = (-2000) / 92.3 * x_f0 * SCALE


    # 3 points of A4 paper in camera coordinate system.
    x1, y1, z1 = -92.3, 135.8, 2000
    x2, y2, z2 = 92.3, 135.8, 2000
    x3, y3, z3 = -92.3, -135.8, 2000
    x4, y4, z4 = 92.3, -135.8, 2000
    

    V1 = np.array([x1, y1, z1])
    V2 = np.array([x2, y2, z2])
    V3 = np.array([x3, y3, z3])
    V4 = np.array([x4, y4, z4])


    # sample x,y,z  a,b,g: 0~2*pi, -pi/2 ~ pi/2
    max_a = math.pi * 2
    max_b = old_div(math.pi, 2)
    max_g = old_div(math.pi, 2)
    max_distance = 2600
    distance_step = 200
    
    if display:
        print((height, width, channels))
        print (f"estimate focal length: , {f},  pixel")
        print((x1, y2, z3))
        print(V1[0])
        print((V1, V2, V3, V4))


    for item in sample_6para:
        x, y, z, a, b, g = item[0], item[1], item[2], item[3], item[4], item[5]


        '''rotate in self coordinate system'''
        V1_self = np.array([V1[0], V1[1], 0])
        V2_self = np.array([V2[0], V2[1], 0])
        V3_self = np.array([V3[0], V3[1], 0])
        V4_self = np.array([V4[0], V4[1], 0])

        V1_self_ = transform6para(V1_self, 0, 0, 0, a, b, g)
        V2_self_ = transform6para(V2_self, 0, 0, 0, a, b, g)
        V3_self_ = transform6para(V3_self, 0, 0, 0, a, b, g)
        V4_self_ = transform6para(V4_self, 0, 0, 0, a, b, g)
        
        V1_ = np.array([V1_self_[0], V1_self_[1], V1_self_[2] + V1[2]])
        V2_ = np.array([V2_self_[0], V2_self_[1], V2_self_[2] + V2[2]])
        V3_ = np.array([V3_self_[0], V3_self_[1], V3_self_[2] + V3[2]])
        V4_ = np.array([V4_self_[0], V4_self_[1], V4_self_[2] + V4[2]])

        '''transform in camera xyz coordinate system. '''
        V1_ = transform6para(V1_, x, y, z, 0, 0, 0)
        V2_ = transform6para(V2_, x, y, z, 0, 0, 0)
        V3_ = transform6para(V3_, x, y, z, 0, 0, 0)
        V4_ = transform6para(V4_, x, y, z, 0, 0, 0)

        x_f_1 = old_div(x_f0 * V1_[0], V1_[2]) * (-2000) / 92.3
        y_f_1 = old_div(y_f0 * V1_[1], V1_[2]) * (2000) / 135.8
        x_0f_1 = HALF_SZ + x_f_1
        y_0f_1 = HALF_SZ + y_f_1

        x_f_2 = old_div(x_f0 * V2_[0], V2_[2]) * (-2000) / 92.3
        y_f_2 = old_div(y_f0 * V2_[1], V2_[2]) * (2000) / 135.8
        x_0f_2 = HALF_SZ + x_f_2
        y_0f_2 = HALF_SZ + y_f_2

        x_f_3 = old_div(x_f0 * V3_[0], V3_[2]) * (-2000) / 92.3
        y_f_3 = old_div(y_f0 * V3_[1], V3_[2]) * (2000) / 135.8
        x_0f_3 = HALF_SZ + x_f_3
        y_0f_3 = HALF_SZ + y_f_3

        x_f_4 = old_div(x_f0 * V4_[0], V4_[2]) * (-2000) / 92.3
        y_f_4 = old_div(y_f0 * V4_[1], V4_[2]) * (2000) / 135.8
        x_0f_4 = HALF_SZ + x_f_4
        y_0f_4 = HALF_SZ + y_f_4

        pts2 = np.float32([[x_0f_1, y_0f_1],[x_0f_2, y_0f_2],[x_0f_3, y_0f_3], [x_0f_4, y_0f_4]])

        M0 = cv2.getPerspectiveTransform(pts1,pts2)
        M = cv2.getPerspectiveTransform(pts2,pts1)
        
        if display:
            print((x, y, z, a, b, g))
            # M is 3x3 matrix, take only the first 8
            print(("M element is ", [M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1]]))

        # choose use M or M0
        sample_matrixes.append([M0[0][0], M0[0][1], M0[0][2], M0[1][0], M0[1][1], M0[1][2], M0[2][0], M0[2][1]])
        # sample_matrixes.append([M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1]])

        # imshow if needed
        img_resized = cv2.resize(img, (448, 448))
        dst_resized = cv2.warpPerspective(img_resized, M0, (448, 448))

    return sample_matrixes


if __name__=='__main__':
    target_sample()
