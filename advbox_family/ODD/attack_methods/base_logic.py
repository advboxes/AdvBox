from __future__ import division
from __future__ import print_function
from builtins import input
from builtins import range
from builtins import object
from past.utils import old_div
from abc import ABCMeta
from abc import abstractmethod

import numpy as np
import cv2
import os
import xmltodict
import pdb
from future.utils import with_metaclass


class ODD_logic(with_metaclass(ABCMeta, object)):

    def __init__(self, model):
        """
        Abstract base class for ODD. With some visualization tools inside.
        """
        # default value
        self.disp_console = True
        self.model = model
        self.success = 0
        self.overall_pics = 0
        self.path = './result/'
        self.very_small = 0.000001
        self.mask_list = None
        
        # global variable to be parsed in argv_parser()
        self.h_img = None
        self.w_img = None
        self.d_img = None
        self.fromfile = None 
        self.frommaskfile = None
        self.fromlogofile = None
        self.fromfolder = None


    def __call__(self, argvs = []):
        self.argv_parser(argvs)
        self.build_model_attack_graph()
        
        self._attack()
        
        return None

    @abstractmethod
    def argv_parser(self, argvs, **kwargs):
        """
        Parse given command information as attack setting.

        Args:
        argvs: The adversary object.
        **kwargs: Other named arguments.
        """
        raise NotImplementedError


    @abstractmethod
    def build_model_attack_graph(self, **kwargs):
        """
        Prepare an end-to-end differentiable model.

        Args:
        **kwargs: Other named arguments.
        """
        raise NotImplementedError


    @abstractmethod
    def attack_optimize(self, img_list, mask, logo_mask=None, resized_logo_mask=None, **kwargs):
        """
        Customize optimization process. The goal is to make the most robust(generalize) ad sticker on real inputs.
        As attack objective varies, the cost function can be tailered to it.
        
        Args:
        img_list: The adversary object.
        mask: The numpy array with shape equals to a sample.
        logo_mask: The numpy array mask for perturbed area.
        resized_logo_mask: The logo_mask that resized.
        **kwargs: Other named arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def interpret_output(self, output):
        """
        Comprehend inferenced tensor from model for user interface.
        
        Args:
        output: Inferenced tensor from model.
        """
        raise NotImplementedError
    

    def _attack(self, **kwargs):
        """
        Implement task oriented attack. Organise file level attack logic.
        
        Args:
        **kwargs: Other named arguments.
        """
        assert self.fromfile is not None or self.fromfolder is not None
        
        if self.fromfile is not None and self.frommaskfile is not None:
            adversarial_output = self._attack_from_file(self.fromfile, self.frommaskfile, self.fromlogofile)


        if self.fromfolder is not None:
            filename_list = os.listdir(self.fromfolder)
            # take pics name out and construct xml filename to read from
            for filename in filename_list:
                pic_name = re.match(r'\d+.JPG', filename)

                if pic_name is not None:
                    self.overall_pics+=1
                    print(("Pics number:",self.overall_pics,"The",pic_name[0], "!"))

                    pic_mask_name = pic_name[0][:-3]+"xml"
                    fromfile = self.fromfolder+"/"+pic_name[0]
                    frommask = self.fromfolder+"/"+pic_mask_name

                    self._attack_from_file(fromfile, frommask)

            print(("Attack success rate:", old_div(self.success,self.overall_pics)))


    def _attack_from_file(self, sample_folder, maskfilename, logo_filename=None, **kwargs):
        """
        Search an adversarial example from given information with attack graph.
        
        Args:
        sample_folder: The complete path for base information.
        maskfilename: The complete path for mask coordination.
        **kwargs: Other named arguments.
        """
        if self.disp_console : print('Generating from ' + sample_folder + '...')

        sample_list = self._get_sample_list(sample_folder)
        # record scale of the orginal images
        self.h_img, self.w_img, self.d_img = sample_list[0][1].shape
        mask = self.very_small * np.ones(shape = sample_list[0][1].shape)

        # if there is logo file, prepare logo_mask
        logo_mask = None
        if logo_filename is not None:
            logopic = cv2.imread(logo_filename)
            # flag indicates where the pixels' value will spared from ad perturbation
            flag = 100
            logo_mask = self._generate_logomask(logopic,
                                                flag)
        
        print("Generating Mask...")
        self.mask_list = self._parse_mask(maskfilename)
        resized_logo_mask_list = []
        for _object in self.mask_list:
            xmin, ymin, xmax, ymax = self._get_mask_coordination(_object)
            mask = self._generate_MaskArea(mask, xmin, ymin, xmax, ymax)
            
            
            # if there is logo file, draw logo where there is flags
            if logo_filename is not None:
                mask, resized_logo_mask = self._add_logomask(mask, logo_mask, xmin, ymin, xmax, ymax)
                resized_logo_mask_list.append(resized_logo_mask)

                
        # usually the first area is what we need, the other one(for confusing tool-box users) is too small
        if logo_mask is not None:
            adversarial_output = self.attack_optimize(sample_list, mask, logo_mask, resized_logo_mask_list[0])
            # reconstruct image from perturbation
            adversarial_sticker = self._save_np_as_jpg('final_adsticker.jpg',
                                                       adversarial_output,
                                                       logo_mask, 
                                                       resized_logo_mask_list[0])

        else:
            adversarial_output = self.attack_optimize(sample_list, mask)
            # reconstruct image from perturbation
            adversarial_sticker = self._save_np_as_jpg('final_adsticker.jpg', adversarial_output)

        return adversarial_sticker

    def _get_sample_list(self, sample_folder):
        """
        Prepare samples as a list.
        
        Args:
        sample_folder: complete path for samples' folder.
        return: a list of samples with dir.
        """
        sample_list = []
        filenames = os.listdir(sample_folder)
        for filename in filenames:
            file_path = os.path.join(sample_folder, filename)
            img = cv2.imread(file_path)
            sample_list.append((file_path, img))
         
        return sample_list

    def _parse_mask(self, maskfilename):
        """
        Use to parse mask coordination in xml file as readable variables.
        
        Args:
        maskfilename: Path for the xml file containing mask coordination.
        **kwargs: Other named arguments.
        return: a structure with mask coordination.
        """
        f = open(maskfilename)
        dic = xmltodict.parse(f.read())

        return dic['annotation']['object']
    
    def _get_mask_coordination(self, _object):
        """
        Place mask coordination in variables.
        
        Args:
        maskfilename: Path for the xml file containing mask coordination.
        **kwargs: Other named arguments.
        """
        xmin = int(_object['bndbox']['xmin'])
        ymin = int(_object['bndbox']['ymin'])
        xmax = int(_object['bndbox']['xmax'])
        ymax = int(_object['bndbox']['ymax'])
        
        return xmin,ymin,xmax,ymax
        
    def _save_np_as_jpg(self, save_name, x, logo_mask=None, resized_logo_mask=None):
        '''
        Save numpy pic as a jpg
        
        Args:
        x: Numpy array between (-1,1)
        return: reconstruct_img_np_squeezed is numpy array between (0,1)
        '''
        # reconstruct image from perturbation
        ad_x=x
        ad_x_01=(ad_x/2.0)+0.5
        
        # bx.imshow only take value between 0 and 1
        squeezed=np.squeeze(ad_x_01)

        ad_x_squeezed=np.squeeze(ad_x)
        reconstruct_img_resized_np=(ad_x_squeezed+1.0)/2.0*255.0
        print(("min and max in img(numpy form):",reconstruct_img_resized_np.min(),reconstruct_img_resized_np.max()))

        reconstruct_img_np=cv2.resize(reconstruct_img_resized_np,(self.w_img,self.h_img))#reconstruct_img_BGR
        reconstruct_img_np_squeezed=np.squeeze(reconstruct_img_np)
        
        # choose to generate invisible clothe
        user_input = "Yes"
        save_name = 'HD_sticker.jpg'
        while user_input!="No" and self.Do_you_want_ad_sticker is True:
            user_input = eval(input("Do you want an invisible clothe? Yes/No:"))
            if user_input=="Yes":
                print("Ok!")
                if logo_mask is not None and resized_logo_mask is not None:
                    self._generate_sticker(save_name, reconstruct_img_np_squeezed, logo_mask, resized_logo_mask)
                else:
                    self._generate_sticker(save_name, reconstruct_img_np_squeezed)
                break
            elif user_input=="No":
                print("Bye-Bye!")
                break
            else:
                print("Wrong command!")
                user_input = eval(input("Do you want an invisible clothe? Yes/No:"))

        return reconstruct_img_np_squeezed
    
    # generate_sticker saved under result folder
    def _generate_sticker(self, save_name, pic_in_numpy_0_255, logo_mask=None, resized_logo_mask=None):
        """
        Process perturbed result and save.
        
        Args:
        save_name: sticker saving name.
        pic_in_numpy_0_255: numpy array with value within (0,255) for saving.
        """
        is_saved = None
        
        _object = self.mask_list[0]
        xmin, ymin, xmax, ymax = self._get_mask_coordination(_object)
        print((xmin,ymin,xmax,ymax))
        
        sticker_in_numpy_0_255_original = pic_in_numpy_0_255[ymin:ymax, xmin:xmax]

        if logo_mask is not None and resized_logo_mask is not None:
            resize_ratio = old_div(logo_mask.shape[0],resized_logo_mask.shape[0])
        else:
            resize_ratio = 6 # empirically...
        
        new_sticker_width = int(sticker_in_numpy_0_255_original.shape[1] * resize_ratio)
        new_sticker_height = int(sticker_in_numpy_0_255_original.shape[0] * resize_ratio)
        new_sticker = cv2.resize(sticker_in_numpy_0_255_original,(new_sticker_width,new_sticker_height))
        
        if logo_mask is not None and resized_logo_mask is not None:
            ad_area_center_x = old_div(new_sticker_width,2)
            ad_area_center_y = old_div(new_sticker_height,2)

            # cv2.resize only eats integer
            resized_height = logo_mask.shape[0]
            resized_width = logo_mask.shape[1]

            paste_xmin = int(ad_area_center_x - old_div(resized_width,2))
            paste_ymin = int(ad_area_center_y - old_div(resized_height,2))
            paste_xmax = paste_xmin + resized_width
            paste_ymax = paste_ymin + resized_height
            
            for i in range(paste_xmin,paste_xmax):
                for j in range(paste_ymin,paste_ymax):
                    if logo_mask[j-paste_ymin,i-paste_xmin,0]==self.very_small:
                        new_sticker[j,i] = 255
        
        assert new_sticker is not None
        is_saved=cv2.imwrite(self.path+save_name, new_sticker)
        if is_saved:
            print(("Sticker saved under:", self.path+save_name))
        else:
            print("Sticker saving error")

        return is_saved
    
    # generate mask
    def _generate_MaskArea(self,
                           mask,
                           xmin,
                           ymin,
                           xmax,
                           ymax):
        """
        make a mask where adversarial perturbed area
        """
        pdb.set_trace()
        for i in range(xmin,xmax):
            for j in range(ymin,ymax):
                mask[j][i] = 1

        return mask
    
    def _generate_logomask(self,
                          logopic,
                          flag):
        """
        turn logopic into binary matrix logo_mask.
        """
        logo_mask = np.where(logopic>flag, self.very_small, 1)

        return logo_mask
    
    
    def _add_logomask(self,
                      mask,
                      logo_mask,
                      xmin,
                      ymin,
                      xmax,
                      ymax):
        """
        Put logo_mask onto the center of the ready-to-perturbed area.
        
        Return: mask for img, resized logo mask for ad area
        """
        ad_width = xmax - xmin
        ad_height = ymax - ymin
        
        ad_area_center_x = old_div((xmin+xmax),2)
        ad_area_center_y = old_div((ymin+ymax),2)
        
        ad_ratio = old_div(ad_width,ad_height)

        logo_height = logo_mask.shape[0]
        logo_width = logo_mask.shape[1]
        logo_ratio = old_div(logo_width,logo_height)
        
        # make sure logo is contained by ad area
        if ad_ratio > logo_ratio:
            resize_ratio = old_div(ad_width,logo_height)
        else:
            resize_ratio = old_div(ad_height,logo_width)

        # cv2.resize only eats integer
        resized_height = int(logo_height*resize_ratio)
        resized_width = int(logo_width*resize_ratio)
        
        resized_logo_mask = None
        # skip cases where resize area is too small
        if resized_width!=0 and resized_height!=0:
            resized_logo_mask = cv2.resize(logo_mask, (resized_width,resized_height))
            
            paste_xmin = int(ad_area_center_x - old_div(resized_width,2))
            paste_ymin = int(ad_area_center_y - old_div(resized_height,2))
            paste_xmax = paste_xmin + resized_width
            paste_ymax = paste_ymin + resized_height
            
            if (xmin+resized_height)<mask.shape[0] and (ymin+resized_width)<mask.shape[1]:
                mask[paste_ymin:paste_ymax,paste_xmin:paste_xmax] = resized_logo_mask
                # np.zeros([resized_height,resized_width,3])
            else:
                pass
        else:
            pass

        return mask, resized_logo_mask
        