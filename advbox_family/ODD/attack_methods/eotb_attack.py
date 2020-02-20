'''
This is an implementation for phycisal attack on test distribution.

'''
from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np
import cv2
import tensorflow as tf

import time
import os

from EOT_simulation import transformation
from attack_methods.base_logic import ODD_logic
import pdb


class EOTB_attack(ODD_logic):
    def __init__(self, model):
        super(EOTB_attack, self).__init__(model)
        # init global variable
        self.filewrite_img = False
        self.filewrite_txt = False
        self.tofile_img = os.path.join(self.path,'output.jpg')
        self.tofile_txt = os.path.join(self.path,'output.txt')
        
        self.imshow = False
        self.useEOT = True
        self.Do_you_want_ad_sticker = True
        self.weights_file = 'weights/YOLO_tiny.ckpt'
        
        # optimization settings
        self.learning_rate = 1e-2
        self.steps = 4
        self.alpha = 0.1
        self.threshold = 0.2
        self.iou_threshold = 0.5
        self.num_class = 20
        self.num_box = 2
        self.grid_size = 7
        self.classes =  ["aeroplane",
                         "bicycle", 
                         "bird", 
                         "boat", 
                         "bottle", 
                         "bus", 
                         "car", 
                         "cat", 
                         "chair", 
                         "cow", 
                         "diningtable", 
                         "dog", 
                         "horse", 
                         "motorbike", 
                         "person", 
                         "pottedplant", 
                         "sheep", 
                         "sofa", 
                         "train",
                         "tvmonitor"]

    def argv_parser(self, argvs):
        for i in range(1,len(argvs),2):
            # read picture file
            if argvs[i] == '-fromfile' : self.fromfile = argvs[i+1]
            if argvs[i] == '-fromfolder' : self.fromfolder = argvs[i+1]
            if argvs[i] == '-frommaskfile' : self.frommaskfile = argvs[i+1]
            if argvs[i] == '-fromlogofile' :
                self.fromlogofile = argvs[i+1]
            else:
                self.fromlogofile = None

            if argvs[i] == '-tofile_img' : self.tofile_img = argvs[i+1] ; self.filewrite_img = True
            if argvs[i] == '-tofile_txt' : self.tofile_txt = argvs[i+1] ; self.filewrite_txt = True
            
            if argvs[i] == '-imshow' :
                if argvs[i+1] == '1' :self.imshow = True
                else : self.imshow = False
                    
            if argvs[i] == '-useEOT' :
                if argvs[i+1] == '1' :self.useEOT = True
                else : self.useEOT = False
                    
            if argvs[i] == '-Do_you_want_ad_sticker' :
                if argvs[i+1] == '1' :self.Do_you_want_ad_sticker = True
                else : self.Do_you_want_ad_sticker = False
                    
            if argvs[i] == '-disp_console' :
                if argvs[i+1] == '1' :self.disp_console = True
                else : self.disp_console = False

    def build_model_attack_graph(self):
        if self.disp_console : print("Building attack graph...")
        # compute the EOT-transformed masked inter in a batch, 
        if self.useEOT == True:
            print("Building EOT Model graph!")
            self.EOT_transforms = transformation.target_sample()
            self.num_of_EOT_transforms = len(self.EOT_transforms)
            print(f'EOT transform number: {self.num_of_EOT_transforms}')


        # x is the image
        self.x = tf.placeholder('float32',[self.num_of_EOT_transforms,448,448,3])
        self.mask = tf.placeholder('float32',[1,448,448,3])

        self.punishment = tf.placeholder('float32',[1])
        self.smoothness_punishment=tf.placeholder('float32',[1])
        
        # original
        # init_inter = tf.constant_initializer(0.001*np.random.random([1,448,448,3]))
        
        # improved
        # think of it, we want ad sticker starts at somewhere easy
        init_inter = tf.constant_initializer(0.7*np.random.normal(scale=0.8,size=[1,448,448,3]))
        
        self.inter = tf.get_variable(name='inter',
                                     shape=[1,448,448,3],
                                     dtype=tf.float32,
                                     initializer=init_inter)

        # box constraints ensure self.x within(0,1)
        self.w = tf.atanh(self.x)
        # add mask
        self.masked_inter = tf.multiply(self.mask,self.inter)
        
        
        # compute the EOT-transformed masked inter in a batch, 
        if self.useEOT == True:
            # broadcast self.masked_inter [1,448,448,3] into [num_of_EOT_transforms, 448, 448, 3]
            self.masked_inter_batch = self.masked_inter
            for i in range(self.num_of_EOT_transforms):
                if i == self.num_of_EOT_transforms-1: break
                self.masked_inter_batch = tf.concat([self.masked_inter_batch,self.masked_inter],0)

            # interpolation choices "NEAREST", "BILINEAR"
            self.masked_inter_batch = tf.contrib.image.transform(self.masked_inter_batch,
                                                                 self.EOT_transforms,
                                                                 interpolation='BILINEAR')


        else:
            self.masked_inter_batch = self.masked_inter
            print("EOT mode disabled!")

        # tf.add making self.w [1,448,448,3] broadcast into [num_of_EOT_transforms, 448, 448, 3]
        self.shuru = tf.add(self.w,self.masked_inter_batch)
        self.constrained = tf.tanh(self.shuru)
        
        # create session
        self.sess = tf.Session() # config=tf.ConfigProto(log_device_placement=True)

        init_dict = {'yolo_model_input': self.constrained,
                     'yolo_mode': "init_model",
                     'yolo_disp_console': self.disp_console,
                     'session': self.sess}

        # init a model instance
        self.object_detector = self.model(init_dict)
        self.C_target = self.object_detector.get_output_tensor()

        MODEL_variables = self.object_detector.get_yolo_variables()
        # Alternatives:
        # leave out tf.inter variable which is not part of yolo model
        # MODEL_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[1:]
        # MODEL_variables = tf.contrib.framework.get_variables()[1:]

        # unused
        MODEL_variables_name = [variable.name for variable in MODEL_variables]


        # computer graph for norm 2 distance
        # init an ad example
        self.perturbation = self.x-self.constrained
        self.distance_L2 = tf.norm(self.perturbation, ord=2)
        self.punishment = tf.placeholder('float32',[1])

        # non-smoothness
        self.lala1 = self.masked_inter[0:-1,0:-1]
        self.lala2 = self.masked_inter[1:,1:]
        self.sub_lala1_2 = self.lala1-self.lala2
        self.non_smoothness = tf.norm(self.sub_lala1_2, ord=2)

        # loss is maxpooled sum confidence from batch dimension + distance_L2 + print smoothness
        self.loss = self.C_target+self.punishment*self.distance_L2+self.smoothness_punishment*self.non_smoothness

        # set optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)#GradientDescentOptimizerAdamOptimizer
        self.attackoperator = self.optimizer.minimize(self.loss,var_list=[self.inter])

        # init and load weights by variables
        self.sess.run(tf.global_variables_initializer())

        # restore model variable
        saver = tf.train.Saver(MODEL_variables)
        saver.restore(self.sess,self.weights_file)


        if self.disp_console : print("Loading complete!" + '\n')



    def attack_optimize(self, img_list, mask, logo_mask=None, resized_logo_mask=None):
        s = time.time()

        # set fetch list
        fetch_list = [self.object_detector.fc_19, # fetch_list[0] is for user interpretation
                      self.attackoperator,        # controlling attack loss
                      self.constrained,           # picture matrix to be reconstructed
                      self.C_target,              # for interpreting attack process
                      self.loss]                  # for interpreting attack process


        # attack
        print("Adversarial attack...")
        
        inputs = np.zeros((self.num_of_EOT_transforms,448,448,3),dtype='float32')   
        inputs_mask = np.zeros((1,448,448,3),dtype='float32')
        mask_resized = cv2.resize(mask, (448,448))
        inputs_mask[0] = mask_resized
        
        # hyperparameter to control two optimization objectives
        punishment = np.array([0.01])
        smoothness_punishment = np.array([0.5])
        
        # if you want
        Target_Loss = []
        Image_Loss = []
        # np.random.shuffle(img_list)
        for i in range(self.steps):
            # prepare attack batch
            for j in range(self.num_of_EOT_transforms):
                choose = np.random.randint(len(img_list))
                # print(img_list[choose][0])
                img_with_logo = self._init_sticker_area(img_list[choose][1], 
                                                       logo_mask, 
                                                       resized_logo_mask)

                img_resized = cv2.resize(img_with_logo, (448, 448))

                # just like np.newaxis, expanding a dimension at the front
                inputs[j] = (img_resized/255.0)*2.0-1.0


            # set original image and punishment
            in_dict = {self.x: inputs,
                       self.punishment: punishment,
                       self.mask: inputs_mask,
                       self.smoothness_punishment: smoothness_punishment}

            # fetch something in self(tf.Variable)
            net_output = self.sess.run(fetch_list, feed_dict=in_dict)
            print(f"step: {i}, Target Loss: {net_output[3]}, Image Loss: {net_output[4][0]-net_output[3]}")
            Target_Loss.append(net_output[3])
            Image_Loss.append(net_output[4][0]-net_output[3])
            
            ##tmp code
            # adsticker = self._save_np_as_jpg(str(i)+'_whole_pic.jpg', net_output[2][0])
            ##

        strtime = str(time.time()-s)
        if self.disp_console : print('Elapsed time : ' + strtime + ' secs' + '\n')
        print("Attack finished!")

        result = self._interpret_output(net_output[0][0])
        self.show_results(img_with_logo, result)
        
        return net_output[2][0]


    # add logo on input
    def _init_sticker_area(self, pic_in_numpy_0_255, logo_mask=None, resized_logo_mask=None):
        is_saved = None

        # copy a new array out
        pic_in_numpy_0_255_copy = np.array(pic_in_numpy_0_255)
        _object = self.mask_list[0]
        xmin = int(_object['bndbox']['xmin'])
        ymin = int(_object['bndbox']['ymin'])
        xmax = int(_object['bndbox']['xmax'])
        ymax = int(_object['bndbox']['ymax'])

        if logo_mask is not None and resized_logo_mask is not None:
            ad_area_center_x = old_div((xmin+xmax),2)
            ad_area_center_y = old_div((ymin+ymax),2)

            # cv2.resize only eats integer
            resized_width = resized_logo_mask.shape[1]
            resized_height = resized_logo_mask.shape[0]

            paste_xmin = int(ad_area_center_x - old_div(resized_width,2))
            paste_ymin = int(ad_area_center_y - old_div(resized_height,2))
            paste_xmax = paste_xmin + resized_width
            paste_ymax = paste_ymin + resized_height

            # can also write as np.where(cond, v1, v2)
            for i in range(paste_xmin,paste_xmax):
                for j in range(paste_ymin,paste_ymax):
                    if resized_logo_mask[j-paste_ymin,i-paste_xmin,0]==self.very_small:
                        # plot logo
                        pic_in_numpy_0_255_copy[j,i] = 255

        # uniform base color of the ad_sticker area
        # balance between printability and constrast
        # rand = (np.random.normal(loc=0.5, scale=0.1, size=pic_in_numpy_0_255_copy[ymin:ymax, xmin:xmax].shape)/2)*255
        # rand_int = rand.astype('int')
        # pic_in_numpy_0_255_copy[ymin:ymax, xmin:xmax] = rand_int

        pic_in_numpy_0_255_copy[ymin:ymax, xmin:xmax] = [164, 83, 57] # a rgb point inside CMYK color space
        
        return pic_in_numpy_0_255_copy

    def _iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return old_div(intersection, (box1[2]*box1[3] + box2[2]*box2[3] - intersection))
    
    def _interpret_output(self, output):
        probs = np.zeros((7,7,2,20))
        class_probs = np.reshape(output[0:980],(7,7,20))
        scales = np.reshape(output[980:1078],(7,7,2))
        boxes = np.reshape(output[1078:],(7,7,2,4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])

        boxes[:,:,:,0] *= self.w_img
        boxes[:,:,:,1] *= self.h_img
        boxes[:,:,:,2] *= self.w_img
        boxes[:,:,:,3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

        filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)

        boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]

        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0 : continue
            for j in range(i+1,len(boxes_filtered)):
                if self._iou(boxes_filtered[i],boxes_filtered[j]) > self.iou_threshold : 
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]

        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]],
                           boxes_filtered[i][0],
                           boxes_filtered[i][1],
                           boxes_filtered[i][2],
                           boxes_filtered[i][3],
                           probs_filtered[i]])

        return result

    def show_results(self, img, results):
        img_cp = img.copy()
        if self.filewrite_txt :
            ftxt = open(self.tofile_txt,'w')
        class_results_set = set()
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3])//2
            h = int(results[i][4])//2
            class_results_set.add(results[i][0])
            if self.disp_console : print('    class : ' + 
                                         results[i][0] + ' , [x,y,w,h]=[' + 
                                         str(x) + ',' + str(y) + ',' + 
                                         str(int(results[i][3])) + ',' + 
                                         str(int(results[i][4]))+'], Confidence = ' + 
                                         str(results[i][5]))

            if self.filewrite_img or self.imshow:
                cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
                cv2.putText(img_cp,
                            results[i][0] + ' : %.2f' % results[i][5],(x-w+5,y-h-7),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

            if self.filewrite_txt :                
                ftxt.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h)+',' + str(results[i][5]) + '\n')
        # suppose we want to know how attack performs on class "person"
        if "person" not in class_results_set:
            self.success+=1
            print("Attack succeeded!")
        else:
            print("Attack failed!")

        if self.filewrite_img : 
            if self.disp_console : print('image file writed : ' + self.tofile_img)
            cv2.imwrite(self.tofile_img, img_cp)

        if self.filewrite_txt : 
            if self.disp_console : print('txt file writed : ' + self.tofile_txt)
            ftxt.close()

        if self.imshow :
            cv2.imshow('detection display', img_cp)
            cv2.waitKey(1)
            


    

