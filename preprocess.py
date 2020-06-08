# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:53:09 2020

@author: mousekinga82
"""
import h5py
import os
from utils import standardized
from generator import VolumeDataGenerator
from tqdm import tqdm

def create_h5py_pathes(data_dir, fold, fold_name, input_dim, input_sub_dim, bk_thres = 0.85, pre_dir = 'preprocess',
                       skip = True, pre_process = True, norm_mode = 2, sample_mean_std_json = "", max_tries = 1000, sample_batch_size = 30, sample_each_image = 3):
    '''
    Create pathches in h5py format
    ----------
    data_dir : str
        top data dir ex: 'Task01_BrainTumour'.
    fold : list
        file list for patch extracting with same format as dataset.json.
    fold_name : str 
        sub directory after pre_dir (using 'train' or 'validation').
    input_dim : tuple
        input image dimension.
    input_sub_dim : tuple
        output sub-image dimesion.
    bk_thres : int, optional
        background threshold. The default is 0.85.
    pre_dir : str, optional
        The dir name for preprocess data. The default is 'preprocess'.
    skip : bool, optional
        weather to skip the image that fails the threshold. The default is True.
    pre_process : bool, optional
        doing preprocess or not. The default is True.
    norm_mode : int, optional
        normalization mode. The default is 2.
    sample_mean_std_json : str, optional
        path for the json file which contains mean & std info. The default is "".
    max_tries : int, optional
        maximum tries for searching sub-image. The default is 1000.
    sample_batch_size : int, optional
        sample batch size. The default is 30.
    sample_each_image : TYPE, optional
        sample times for each image. The default is 3.

    Returns
    -------
    save_list : TYPE
        DESCRIPTION.

    '''
    gen = VolumeDataGenerator(fold, data_dir, pre_process = pre_process, norm_mode = norm_mode, sample_mean_std_json = sample_mean_std_json, input_dim = input_dim, output_dim = input_sub_dim, return_last = False, return_file_list = True, bk_thres = bk_thres, batch_size = sample_batch_size, max_tries = max_tries)
    full_dir = os.path.join(pre_dir, fold_name)
    save_list = []
    print(f'Creating {fold_name} data pathches ...')
    for i in range(sample_each_image):
        print(f'The {i} time searching starts ...')
        for j in tqdm(range(len(gen))):
            x, y, file_name = gen[j]
            for x_, y_, file_name_ in zip(x, y, file_name):
                if file_name_ == None: continue
                file_name_ = file_name_ + '_' + str(i)
                tmp_name = os.path.join(full_dir, file_name_)
                hf = h5py.File(tmp_name, 'w')
                hf.create_dataset('x', data = x_)
                hf.create_dataset('y', data = y_)
                hf.close()
                save_list.append(file_name_)
    return save_list