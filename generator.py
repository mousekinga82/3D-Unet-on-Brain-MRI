# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:09:31 2020

@author: mousekinga82
"""
from tensorflow import keras
import numpy as np
import os
from utils import standardized, load_case, get_sub_volume
import h5py
import json

class VolumeDataGenerator(keras.utils.Sequence):

    def __init__(self,
                 sample_fold, #train or val fold
                 base_dir,  # date directory      
                 batch_size = 1, #batch size
                 shuffle = True, #if using shuffle
                 input_dim = (240, 240, 155),  #input dim
                 output_dim = (160,160, 16),    #output dim
                 num_channel = 4, #num of input channel
                 num_class = 4,   #num of output class
                 seed = 0,  
                 max_tries = 1000, #maximum tries for searching sub image
                 bk_thres = 0.95,  #background threshold (maximum)
                 return_last = True, # if return the last search of sub image (use False to skip)
                 return_file_list = False, # if return input file list
                 pre_process = True, # if using preprocess
                 norm_mode = 2, # norm mode 1 or 2 , see standardized (valid only if pre_process = True)
                 sample_mean_std_json = None, # path of json file with sample mean and std info
                 from_patches = False, # Load the data from pathches file
                 exclude_background = False): #if exclude the background class
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_channel = num_channel
        self.num_class = num_class
        self.sample_fold = sample_fold  
        self.max_tries = max_tries
        self.bk_thres = bk_thres
        self.on_epoch_end()
        self.seed = seed
        self.return_last = return_last
        self.return_file_list = return_file_list
        self.set_seed()
        self.pre_process = pre_process
        self.norm_mode = norm_mode
        self.sample_mean_std_json = sample_mean_std_json
        self.from_patches = from_patches
        self.exclude_background = exclude_background
        
    def set_seed(self):
        np.random.seed(self.seed)
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sample_fold))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.floor(len(self.sample_fold) / self.batch_size))

    def __data_generation(self, fold_tmp):
        if self.exclude_background & self.from_patches:
            y = np.zeros((self.batch_size, self.num_class - 1, *self.output_dim), dtype=np.float64)
        else:
            y = np.zeros((self.batch_size, self.num_class, *self.output_dim), dtype=np.float64)
        x = np.zeros((self.batch_size, self.num_channel, *self.output_dim), dtype=np.float64)
        file_list = []
        if self.from_patches == False:
            if self.pre_process & (self.norm_mode == 2):
                with open(self.sample_mean_std_json, "r") as f:
                    info_dict = json.load(f)
            for i, tmp_dict in enumerate(fold_tmp):
                #Store sample
                _, img_dir, file_name  = tmp_dict['image'].split('/')
                _, lab_dir, file_name  = tmp_dict['label'].split('/')
                img_path = os.path.join(self.base_dir, img_dir, file_name)
                lab_path = os.path.join(self.base_dir, lab_dir, file_name)
                print("Loading Sample on: %s" %img_path )
                img, lab = load_case(img_path, lab_path)
                x[i], y[i], flag = get_sub_volume(img, lab, num_cls = self.num_class, origin_dim = self.input_dim,
                                                  output_dim = self.output_dim,  max_tries = self.max_tries, bk_thres = self.bk_thres)
                if (flag == False) & (self.return_last == False):
                    print(f"Pass file {img_path} ...")
                    file_list.append(None)
                    continue
                else:
                    if (self.pre_process) & (self.norm_mode == 1): x[i] = standardized(x[i], self.norm_mode)
                    elif (self.pre_process) & (self.norm_mode == 2): x[i] = standardized(x[i], self.norm_mode, info_dict[file_name])
                    file_list.append(file_name)
            return x, y, file_list
        
        else:
            if self.pre_process & (self.norm_mode == 2):
                with open(self.sample_mean_std_json, "r") as f:
                    info_dict = json.load(f)
            for i, tmp_file_name in enumerate(fold_tmp):
                #In this case, fold_tmp is the list of patches' name, base_dir is the patches location
                tmp_patch_path = os.path.join(self.base_dir, tmp_file_name)
                hf = h5py.File(tmp_patch_path, 'r')
                x[i] = hf.get('x')
                if self.exclude_background:
                    y[i] = hf.get('y')[1:]
                else:
                    y[i] = hf.get('y')
                hf.close()
                if (self.pre_process) & (self.norm_mode == 1): x[i] = standardized(x[i], self.norm_mode)
                elif (self.pre_process) & (self.norm_mode == 2): 
                    x[i] = standardized(x[i], self.norm_mode, info_dict[file_name[:file_name.find('_', 6)]])
                if self.exclude_background : x[i] = x[i][1]
                file_list.append(tmp_file_name)
            return x, y, file_list
                
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[
                    index * self.batch_size: (index + 1) * self.batch_size]
        #Find list of IDs
        sample_fold_tmp = [self.sample_fold[k] for k in indexes]
        #Generate data
        x, y , file_list = self.__data_generation(sample_fold_tmp)
        if self.return_file_list:
            return x, y, file_list
        else:
            return x, y