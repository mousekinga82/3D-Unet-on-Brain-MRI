# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:55:17 2020

@author: mousekinga82
"""

import json
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import os
from datetime import date
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from visualize import *
from utils import *
from model import *
from generator import *
from preprocess import create_h5py_pathes

K.set_image_data_format('channels_first')

#define global parameters
DATA_DIR = "Task01_BrainTumour"
IMG_DIR = os.path.join(DATA_DIR, "imagesTr")
LAB_DIR = os.path.join(DATA_DIR, "labelsTr")
NUM_CH = 4
CHANNEL = {
0: "FLAIR",                #"Fluid Attenuated Inversion Recovery" (FLAIR)
1: "T1-weighted",          #"T1-weighted"
2: "T1-weighted with Gd",  #"T1-weighted with gadolinium contrast enhancement" (T1-Gd)
3: "T2-weighted"           #"T2-weighted"
}
LABEL = {
0: "background",
1: "edema",
2: "non-enhancing tumor",
3: "enhancing tumor",
}
NUM_LAB = len(LABEL)
val_split = 0.2
input_dim = (160, 160, 155)
input_sub_dim = (120, 120, 32)

#Load DICOM data
img_path = os.path.join(IMG_DIR, "BRATS_003.nii.gz")
lab_path = os.path.join(LAB_DIR, "BRATS_003.nii.gz")
img, lab = load_case(img_path, lab_path)
#Show the data
labeled_img = get_labeled_image(img, lab, 2, NUM_CH)
draw_labeled_image(labeled_img, CHANNEL, LABEL, channel_num = 2, axis='z',index = 80)

#Extract & show sub-volume
sub_img, sub_lab, _ = get_sub_volume(img, lab, output_dim = input_sub_dim, bk_thres = 0.9, to_channel_first = True)
labeled_img_sub = get_labeled_image(sub_img, sub_lab, 2, NUM_CH, channel_first = True)
draw_labeled_image(labeled_img_sub, CHANNEL, LABEL, channel_num = 2, axis='z', index = 5)

#Read the json file with train_val, test info 
with open(DATA_DIR + "/dataset.json", "r") as f:
    config = json.load(f)
#Devide the train_val list to folds
folds, n_fold, n_sample = get_train_val_split(config, val_split)
print(f"Validation Splits: {val_split}")
print(f"Available folds for validation : {n_fold}")
print(f"Samples in each fold: {n_sample}")
train_fold, val_fold = set_train_val_split(folds, 0)

#Using generator to create smaples in advance (Optional)
pre_dir = 'preprocess'
#get sample mean & std json file for norm mode = 2
output_file_name = 'samples_mean_std.json'
get_samples_brain_area_mean_std_json(IMG_DIR, num_ch = 4, output_file_name = output_file_name)

#create patchces
json_file = os.path.join(IMG_DIR, output_file_name)
#train_patch_list = create_h5py_pathes(DATA_DIR, train_fold[:10], 'train', input_dim, input_sub_dim, bk_thres = 0.9, pre_dir = pre_dir, skip = True, 
#                                      pre_process = True, norm_mode = 1, sample_mean_std_json = json_file, max_tries = 100, sample_batch_size = 2, sample_each_image = 2)
#val_patch_list = create_h5py_pathes(DATA_DIR, val_fold[:10], 'validation', input_dim, input_sub_dim, bk_thres = 0.9, pre_dir = pre_dir, skip = True, 
#                                      pre_process = True, norm_mode = 1, sample_mean_std_json = json_file, max_tries = 100, sample_batch_size = 2, sample_each_image = 2)    

#Save patch list
#patch_list = {'train': train_patch_list, 'validation': val_patch_list}
#ret = json.dumps(patch_list)
#with open(os.path.join(pre_dir, 'path_list.json'), 'w') as fp:
#    fp.write(ret)

#Create generator for reading patches
with open(os.path.join(pre_dir, 'path_list.json'), 'r') as fp:
    config = json.load(fp)

train_gen = VolumeDataGenerator(config['train'], os.path.join(pre_dir, 'train'), batch_size = 4, output_dim = input_sub_dim, pre_process = False, from_patches = True, exclude_background = False)
#train_gen = VolumeDataGenerator(train_fold, DATA_DIR, batch_size = 4, output_dim = input_sub_dim, pre_process = True, from_patches = False, exclude_background = False, bk_thres=0.95, max_tries=100)
val_gen   = VolumeDataGenerator(config['validation'], os.path.join(pre_dir, 'validation'), batch_size = 4, output_dim = input_sub_dim, pre_process = False, from_patches = True, shuffle = False, exclude_background = False)

#Build the model
model = unet_model_3d(loss_function=soft_dice_loss, num_lab = NUM_LAB, batch_normalization = True,
                      activation_name = 'softmax', depth = 4, input_shape = (NUM_CH, *input_sub_dim))

#Define callbacks
day = str(date.today())
callbacks = [EarlyStopping(monitor='loss', patience=30, verbose=1),
             ModelCheckpoint(f'Model_weights_{day}.h5', save_best_only=True, save_weights_only=True),
             ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, min_lr=1e-8),
             CSVLogger(f'log_{day}.csv', separator=",", append=False)]

#Complie the model
model.compile(optimizer=Adam(lr = 1e-4), loss=soft_dice_loss, metrics=[dice_coefficient])

#Load previous saved weights
model.load_weights('Model_weights_2020-05-30_L4_softmax.h5')

"""
#Fit the model (training)
model.fit_generator(generator=train_gen,
        steps_per_epoch=len(train_gen),
        epochs= 200,
        callbacks = callbacks,
        use_multiprocessing=True)
"""

#draw training history
history = pd.read_csv('log_2020-05-30_L3_softmax2.csv')
fig, ax = plt.subplots(1, 2, figsize = (16, 5))
#soft dice loss
ax[0].plot(history['loss'])
ax[0].plot(history['val_loss'])
ax[0].legend(['loss', 'val_loss'])
ax[0].set_title('soft dice loss')
#dice coefficient
ax[1].plot(history['dice_coefficient'])
ax[1].plot(history['val_dice_coefficient'])
ax[1].legend(['dice_coefficient', 'val_dice_coefficient'])
ax[1].set_title('dice_coefficient')

#Draw picture
#pred = model.predict(val_gen[0])
#test
image  = val_gen[0][0][1]
true_lab = val_gen[0][1][1]
pred_lab = true_lab
lab_ = to_categorical(lab, num_classes = 4)
#compare_true_pred_image(image, true_lab, pred_lab, CHANNEL, LABEL)
compare_true_pred_image(img, lab_, lab_, CHANNEL, LABEL, channel_first = False, show_z = 100)

sen_spec_df =  get_sen_spec_df(lab_, lab_, LABEL)

class cat:
    aa =10
    def __init__(self):
        pass
    def add_cat(self):
       self.cat =5
    def cat_fun():
        print("HAA")
class dog(cat):
    aa= 20
    def __init__(self):
        self.head = 1
        self.add_leg()
        self.add_cat()
    def add_leg(self):
        self.leg = 10