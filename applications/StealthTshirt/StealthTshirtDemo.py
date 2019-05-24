import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import os
import pdb

class Faster_YOLO:
    fromstream = None
    disp_console = False
    
    weights_file = 'weights/YOLO_tiny.ckpt'
    alpha = 0.1
    threshold = 0.25
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

    # temporary global variables to store pic properties within YOLO_TF
    img_path = None
    img = None
    w_img = None
    h_img = None
    result = None

    def __init__(self,argvs = []):
        self.detected = 0
        self.overall_pics = 0
        self.argv_parser(argvs)
        self.build_networks()
        
    def argv_parser(self,argvs):
        for i in range(1,len(argvs),2):
            if argvs[i] == '-fromstream' : self.fromstream = argvs[i+1]

            if argvs[i] == '-disp_console' :
                if argvs[i+1] == '1' :self.disp_console = True
                else : self.disp_console = False
        
    # build detection network and load weights globally
    def build_networks(self):
        if self.disp_console : print("Building YOLO_tiny graph...")
        self.x = tf.placeholder('float32',[None,448,448,3])
        self.conv_1 = self.conv_layer(1,self.x,16,3,1)
        self.pool_2 = self.pooling_layer(2,self.conv_1,2,2)
        self.conv_3 = self.conv_layer(3,self.pool_2,32,3,1)
        self.pool_4 = self.pooling_layer(4,self.conv_3,2,2)
        self.conv_5 = self.conv_layer(5,self.pool_4,64,3,1)
        self.pool_6 = self.pooling_layer(6,self.conv_5,2,2)
        self.conv_7 = self.conv_layer(7,self.pool_6,128,3,1)
        self.pool_8 = self.pooling_layer(8,self.conv_7,2,2)
        self.conv_9 = self.conv_layer(9,self.pool_8,256,3,1)
        self.pool_10 = self.pooling_layer(10,self.conv_9,2,2)
        self.conv_11 = self.conv_layer(11,self.pool_10,512,3,1)
        self.pool_12 = self.pooling_layer(12,self.conv_11,2,2)
        self.conv_13 = self.conv_layer(13,self.pool_12,1024,3,1)
        self.conv_14 = self.conv_layer(14,self.conv_13,1024,3,1)
        self.conv_15 = self.conv_layer(15,self.conv_14,1024,3,1)
        self.fc_16 = self.fc_layer(16,self.conv_15,256,flat=True,linear=False)
        self.fc_17 = self.fc_layer(17,self.fc_16,4096,flat=False,linear=False)
        #skip dropout_18
        self.fc_19 = self.fc_layer(19,self.fc_17,1470,flat=False,linear=True)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

        self.saver.restore(self.sess,self.weights_file)

        if self.disp_console : print("Loading complete!" + '\n')

    def conv_layer(self,idx,inputs,filters,size,stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size//2
        pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        inputs_pad = tf.pad(inputs,pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')    
        conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')    
        if self.disp_console : print('    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels)))

        return tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

    def pooling_layer(self,idx,inputs,size,stride):
        if self.disp_console : print('    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride))

        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

    def fc_layer(self,idx,inputs,hiddens,flat = False,linear = False):
        input_shape = inputs.get_shape().as_list()        

        if flat:
            dim = input_shape[1]*input_shape[2]*input_shape[3]
            inputs_transposed = tf.transpose(inputs,(0,3,1,2))
            inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs

        weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))

        if self.disp_console : print('    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx,hiddens,int(dim),int(flat),1-int(linear)))

        if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
        
        ip = tf.add(tf.matmul(inputs_processed,weight),biases)

        return tf.maximum(self.alpha*ip,ip,name=str(idx)+'_fc')

    # comput iou value
    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

    # inference result from a picture
    def interpret_output(self,output):
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
                if self.iou(boxes_filtered[i],boxes_filtered[j]) > self.iou_threshold : 
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return result

    # show results on picture and optional choices for saving
    def show_results(self, results):
        img_cp = self.img.copy()
            
        class_results_set = set()
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3])//2
            h = int(results[i][4])//2
            class_results_set.add(results[i][0])
            if self.disp_console : print('    class : ' + results[i][0] + 
                                         ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + 
                                         str(int(results[i][3])) + ',' + str(int(results[i][4])) + 
                                         '], Confidence = ' + str(results[i][5]))
                
            # draw bbox
            if self.fromstream:
                cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
                
                # modify showing text 
                cv2.putText(img_cp,results[i][0],  #  + ' : %.2f' % results[i][5]
                            (x-w+5,y-h-7),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(0,0,0),1)

        return img_cp

    # detect from single picture file
    def detect(self, imgpath):
        time0 = time.time()
        if isinstance(imgpath,type(np.zeros(1))):
            self.imgpath = 'time-'+str(time0)
            self.img = imgpath
            self.h_img,self.w_img,_ = self.img.shape

        else:
            if self.disp_console : print('Detect from ' + imgpath)
            self.imgpath = imgpath
            self.img = cv2.imread(self.imgpath)
            self.h_img,self.w_img,_ = self.img.shape
        
        # consider cv2 imread failure
        if self.img is not None:

            img_resized = cv2.resize(self.img, (448, 448))
            img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_resized_np = np.asarray(img_RGB)

            inputs = np.zeros((1,448,448,3),dtype='float32')
            inputs[0] = (img_resized_np/255.0)*2.0-1.0
            in_dict = {self.x: inputs}

            net_output = self.sess.run(self.fc_19, feed_dict=in_dict)

            self.result = self.interpret_output(net_output[0])

            resulted_frame = self.show_results(self.result)

            strtime = str(time.time()-time0)
            if self.disp_console : print('Elapsed time : ' + strtime + ' secs' + '\n')

        return resulted_frame

    def run(self):
        if self.fromstream is not None:
            cap = cv2.VideoCapture(0)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_width = int(cap.get(3))
            video_height = int(cap.get(4))
            video_fps = cap.get(5)
            video_frame_num = cap.get(7)
            imshow_scale = 1.5

            outname = 'invisible-cloak.mp4'
            out = cv2.VideoWriter(outname, fourcc, video_fps, (int(video_width*imshow_scale), int(video_height*imshow_scale)))

            pdb.set_trace()
            while(True):
                # get a frame
                ret, frame = cap.read()
                resulted_frame = self.detect(frame)
                h_imshow, w_imshow, _ = resulted_frame.shape

                drawed_frame = cv2.resize(resulted_frame,
                                                 (int(w_imshow*imshow_scale), int(h_imshow*imshow_scale)))
                # show a frame
                cv2.imshow("capture", drawed_frame)

                out.write(drawed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            
        return None


def main(argvs):
    # init detection process
    object_detector = Faster_YOLO(argvs)
    
    # detect and save
    object_detector.run()
    

if __name__=='__main__':    
    main(sys.argv)
