# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:11:38 2020

@author: mousekinga82
"""
import numpy as np
import nibabel as nib
from tensorflow import keras
import pandas as pd
import glob
import json
import os
from tqdm import tqdm

def load_case(image_nifty_file, label_nifty_file):
    # load the image and label file, get the image content and return a numpy array for each
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())
    return image, label

def get_sub_volume(image, label, origin_dim = (240, 240, 155), output_dim = (160, 160, 16), num_cls = 4, max_tries = 1000, bk_thres = 0.95, 
                   to_channel_first = True):
    """
    Get the sub volumn of the input image
    ----------
    image : np.array
        image with shape (H, W, D, C)
    label : np.array
        label with shape (H, W, D)
    orig_dim : tutple, optional
        original image dimension. The default is (240, 240, 155)
    output_dim : tuple, optional
        output image dimension. The default is (160, 160, 16)
    num_cls : int, optional
        number of labels. The default is 4.
    max_tries : int, optional
        maximum tries. The default is 1000.
    bk_thres : int, optional
        maximum background ratio. The default is 0.95.

    Returns
    -------
    x : np.array
        sub-image with shape (C, output_x, output_y, output_z).
    y : np.array
        sub-lable with shape (num_cls, output_x, output_y, output_z)

    """
    x = None
    y = None
    tries = 0
    flag = False
    while tries < max_tries:
        #Random choose the start index
        start_x = np.random.randint(origin_dim[0] - output_dim[0] + 1)
        start_y = np.random.randint(origin_dim[1] - output_dim[1] + 1)
        start_z = np.random.randint(origin_dim[2] - output_dim[2] + 1)
        #Extract the y lablel
        y = label[start_x: start_x + output_dim[0],
                  start_y: start_y + output_dim[1],
                  start_z: start_z + output_dim[2]]
        #one-hot
        y = keras.utils.to_categorical(y, num_classes = num_cls).astype(np.uint8)
        #calculate background ratio
        bk_ratio = np.sum(y[:,:,:,0] == 1) / (output_dim[0] * output_dim[1] * output_dim[2])
        tries += 1
        if bk_ratio < bk_thres:
            print(f"Accepted after {tries} times for finding sub-volume ...")
            flag = True
            break
    if flag == False: print(f"Tries > {max_tries}, return the latest result. ")
    #Extract imgage
    x = np.copy(image[start_x: start_x + output_dim[0],
                      start_y: start_y + output_dim[1],
                      start_z: start_z + output_dim[2], :])
    if to_channel_first == True:
        #change dimension from (H, W, D, C) to (C, H, W, D)
        x = np.moveaxis(x, -1, 0)
        #change dimension from (H, W, D, num_cls) to (num_cls, H, W, D)
        y = np.moveaxis(y, -1, 0)
        #y = y[1:,:,:,:]
    return x, y, flag
    
def standardized(image, norm_mode, ch_info = ""):
    """
    Standardize the input image
    ----------
    image : np.array
        image of shape (C, H, W, D)
    norm_mode : int
        way of normalization:
        1 : normalize each z slices for each channel
        2 : normalize the whole brain area (exclude background),
            load std & mean of samples from given value 
    ch_info: list
        list contains tuples (mean, std) for each channel, valid when norm mode = 2
    Returns
    -------
    standardized_image : np.array
        the normalized image with same shapes as input 
    """
    standardized_image = np.zeros_like(image)
    if norm_mode == 1:
        for c in range(image.shape[0]):
            for z in range(image.shape[3]):
                image_slice = image[c,:,:,z]
                centered = image_slice - np.mean(image_slice)
                centered_scaled = centered / np.std(image_slice) if np.std(image_slice > 0) else centered
                standardized_image[c,:,:,z] = centered_scaled
    elif norm_mode == 2:
        for c in range(image.shape[0]):
            image_slice = image[c,:,:,:].copy()
            image_slice = (image_slice - ch_info[c][0]) / ch_info[c][1]
            standardized_image[c,:,:,:][image[c,:,:,:] != 0] = image_slice[image[c,:,:,:] != 0]                                    
    else:
        print("Unknow norm mode...")
        exit()
    return standardized_image

def get_samples_brain_area_mean_std_json(img_dir, num_ch = 4, output_file_name = 'samples_mean_std.json'):
    """
    Get the sample's brain area mean and std and saved in json file.
        with format {file_name: [(mean_c1, std_c1), (mean_c2, std_c2), ... (mean_c4, std_c4)]}
    ----------
    img_dir : str
        the directory which contains images
    num_ch : int. optional
        the number of input channel. The default is 4.
    output_file_name : str, optional
        output file name. The default is 'samples_mean_std.json'.

    Returns
    -------
    None.

    """
    image_list = glob.glob(os.path.join(img_dir, '\*.nii.gz'))
    info_dict = {}
    for i in tqdm(range(len(image_list))):
        image = np.array(nib.load(image_list[i]).get_fdata())
        ch_info = []
        for j in range(num_ch):
            c_slice = image[:,:,:,j]
            non_bk = c_slice[c_slice != 0]
            mean = np.mean(non_bk)  
            std  = np.std(non_bk)
            ch_info.append((mean, std))
        info_dict[image_list[i].split(os.path.sep)[-1]] = ch_info.copy()
    ret = json.dumps(info_dict)
    with open(os.path.join(img_dir, output_file_name), 'w') as fp:
        fp.write(ret)


def get_train_val_split(list_from_json, val_split, seed=0):
    """
    Get the train, validation split list
    ----------
    list_from_jason : list
        list load from json file.
    val_split : float
        validation spilt.
    seed : int, optional
        random seed. The default is 0.

    Returns : tuple
        (devied list, available folds, number of samples in each folds)
    -------

    """
    #Random shuffle the train_val list
    train_val_list = list_from_json['training'].copy()
    np.random.seed(seed)
    np.random.shuffle(train_val_list)
    n = int(len(train_val_list) // ( 1 / val_split))
    folds = [ train_val_list[i:i+n] for i in range(0, len(train_val_list), n) ]
    return folds, int(len(train_val_list)/n), n

def set_train_val_split(folds, val_index):
    """
    Set the choosen folds for validation
    ----------
    folds : list
        devied folds.
    val_index : int
        which fold is for validation.

    Returns (train_fold, val_fold)
    -------
    train_fold : list
        folds for training.
    val_fold : list
        the fold for validation.
    """
    train_fold = np.array([])
    for i, f in enumerate(folds):
        if i == val_index : continue
        else:
            train_fold= np.concatenate((train_fold, f))
    return train_fold, folds[val_index]

def get_sen_spec_df(pred, true, label):
    """
    Get the metric for sensitivity and specificity
    ----------
    pred : np.array
        predicted label (one hot and channel last) 
    true : np.array
        true label (one hot and channel last)
    label : dict
        dict with label names.

    Returns
    -------
    patch_metrics : Pandas DataFrame

    """
    label_list = list(label.values())
    label_list.append("tumor")
    patch_metrics = pd.DataFrame(columns = label_list, 
                                 index = ['Sensitivity', 'Specificity'])
    for i, cls_name in enumerate(label_list):
        if i != (len(label_list) -1):
            label_pred = pred[...,i]
            label_true = true[...,i]
        else:
            label_pred = np.logical_or((pred[...,2] == True), (pred[...,3] == True))  
            label_true = np.logical_or((true[...,2] == True), (true[...,3] == True))
        tp = np.sum((label_pred == True) & (label_true == True))
        tn = np.sum((label_pred == False) & (label_true == False))
        fp = np.sum((label_pred == True) & (label_true == False))
        fn = np.sum((label_pred == False) & (label_true == True))
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        patch_metrics.loc['Sensitivity', cls_name] = sens
        patch_metrics.loc['Specificity', cls_name] = spec
    return patch_metrics