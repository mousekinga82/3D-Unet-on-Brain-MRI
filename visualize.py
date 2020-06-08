# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:31:02 2020

@author: mousekinga82
"""

import cv2
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from IPython.display import Image
import imageio
from utils import standardized
import os

#For visualization
def get_labeled_image(image, label, channel, num_cls, channel_first = False):
    """
    Get labeled image
    ----------
    image : np.array 
        image with shape (H, W, D, C)
    label : np.array
        label with shape (H, W, D) or (num_cls, H, W, D)
    channel : int
        The intrested channel
    num_cls : int
        number of ouput labels
    channel_first : bool
        if the channel is first dimension    
    
    Returns
    -------
    labeled_image : np.array
        labeled image with shape (H, W, D, num_cls -1) 
    """
    if channel_first == True:
        image = np.moveaxis(image, 0, -1)
        label = np.moveaxis(label, 0, -1)
    if len(label.shape) == 3:
        label = to_categorical(label, num_classes = num_cls).astype(np.uint8)
    image = cv2.normalize(image[:,:,:,channel], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    labeled_image = np.zeros_like(label[:,:,:,1:])
    #Remove the tumor part from image
    labeled_image[:,:,:,0] = image * label[:,:,:,0] 
    labeled_image[:,:,:,1] = image * label[:,:,:,0]
    labeled_image[:,:,:,2] = image * label[:,:,:,0]
    #color labels
    labeled_image += label[:,:,:,1:] * 255
    return labeled_image

def draw_labeled_image(labeled_image, channel, label, channel_num = None, axis= 'z',index = 80):
    """
    Draw the labeled image
    ----------
    labeled_image : np.array
        Output of get_labeled_image with shape (H, W, D, num_cls -1)
    channel : dict
        dict with input channels
    label : dict
        dict with input labels
    channel_num : int
        Input channel index
    axis : char
        one of 'X', 'Y', 'Z'
    index : int, optional
        The index along the axis. The default is 80.

    Returns
    -------
    None.

    """
    if axis == 'z' or axis =='Z':
        axis = 'Z'; dirc = "Transversal" ; y = 0.7
    elif axis == 'x' or axis == 'X':
        labeled_image = np.transpose(labeled_image, [1, 2, 0, 3])
        labeled_image = np.rot90(labeled_image, 1); y = 0.67
        axis = 'X'; dirc = 'Saggittal'
    elif axis == 'y' or axis == 'Y':
        labeled_image = np.transpose(labeled_image, [0, 2, 1, 3])
        labeled_image = np.rot90(labeled_image, 1); y = 0.67
        axis = 'Y'; dirc = 'Coronal'
    else:
        print("choose one of the axis! (x, y ,z)")
        return None
    #Plot figure
    fig, ax = plt.subplots(1, len(label)-1, figsize = (13,13))
    fig.suptitle(f'Direction:{dirc}, channel:{channel[channel_num]}, {axis}:{index}', y=y, fontsize = 20, fontstyle = 'italic')
    fig.tight_layout()
    for i in range(len(label) -1):
        ax[i].imshow(labeled_image[:,:,index,i], cmap = 'gray')
        ax[i].set_title(label[i+1], fontsize = 18)
    plt.show()
    
def visualize_data_gif(data_):
    images = []
    for i in range(data_.shape[0]):
        x = data_[min(i, data_.shape[0] - 1), :, :]
        y = data_[:, min(i, data_.shape[1] - 1), :]
        z = data_[:, :, min(i, data_.shape[2] - 1)]
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)
    imageio.mimsave("./tmp/gif.gif", images, duration=0.01)
    return Image(filename="./tmp/gif.gif", format='png')

def visualize_comparison_gif(true, pred, axis, file_name = 'compare.gif'):
    images = []
    if (axis == 'x') or (axis == 'X'):
        for i in range(true.shape[0]):
            true_tmp = true[i, :, :]
            pred_tmp = pred[i, :, :]
            img = np.concatenate((true_tmp, pred_tmp), axis = 1)
            images.append(img)
    elif (axis == 'y') or (axis == 'Y'):
        for i in range(true.shape[1]):
            true_tmp = true[:, i, :]
            pred_tmp = pred[:, i, :]
            img = np.concatenate((true_tmp, pred_tmp), axis = 1)
            images.append(img)
    elif (axis == 'z') or (axis == 'Z'):
        for i in range(true.shape[2]):
            true_tmp = true[:, :, i]
            pred_tmp = pred[:, :, i]
            img = np.concatenate((true_tmp, pred_tmp), axis = 1)
            images.append(img)
    else:
        print("invalid axis, use one of 'x', 'y' ,'z' ")
    imageio.mimsave(os.path.join('tmp', file_name), images, duration = 0.012)
    return Image(filename=os.path.join('tmp', file_name), format='png')

def compare_true_pred_image(image, true_lab, pred_lab, channel, label, num_channel = 4, num_cls = 4, show_z = 0, figsize = (18,18), fontsize = 22):
    """
    Parameters
    ----------
    image : np.array 
        image array with shapes (num_channel, H, W, D)
    true_lab : np.array
        image true label with shapes (H, W, D)
    pred_lab : np.array
        image prediction output with shapes (H, W, D)
    channel: dict
        input channel names
    label: dict
        oupput label names
    num_channel : int
        The default is 4.
    num_cls : int, optional
        The default is 4.
    index : int, optional
        index for batches. The default is 0.
    show_z : int, optional
        index in z (D) direction. The default is 0.
    figsize : tuple, optional
        figure size. The default is (13,13).

    Returns
    -------
    None.

    """
    plt_dim = (3, max(num_channel, num_cls))
    fig, ax = plt.subplots(*plt_dim, figsize = figsize)
    image = np.moveaxis(image, -1, 0)
    true_lab = to_categorical(true_lab, num_classes = num_cls)
    pred_lab = to_categorical(pred_lab, num_classes = num_cls)
    true_lab = np.moveaxis(true_lab, -1, 0)
    pred_lab = np.moveaxis(pred_lab, -1, 0)
    for i in range(num_channel):
        ax[0][i].imshow(image[i,:,:,show_z], cmap='gray')
        ax[0][i].set_title(channel[i], fontsize = fontsize)
        if i == 0: ax[0][i].set_ylabel('Input channels', fontsize = fontsize)
    for i in range(num_cls):
        ax[1][i].imshow(true_lab[i,:,:,show_z], cmap='gray')
        ax[1][i].set_title(label[i], fontsize = fontsize)
        if i == 0: ax[1][i].set_ylabel('Ground Truth', fontsize = fontsize)
    #Choose maximum
    pred_ = np.argmax(pred_lab, axis = 0)
    pred_ = to_categorical(pred_, num_classes=4)
    pred_ = np.moveaxis(pred_, 0, -1)
    for i in range(num_cls):
        ax[2][i].imshow(pred_lab[i,:,:,show_z], cmap='gray')
        ax[2][i].set_title(label[i], fontsize = fontsize)
        if i == 0: ax[2][i].set_ylabel('Predicted', fontsize = fontsize)
    fig.tight_layout()
    plt.show()
    
def predict_and_viz(image, label, model, num_cls, ch_info, norm_mode = 2, num_ch = 4, channel = 0, loc = (100, 100, 80), input_dim = (240, 240, 155), input_sub_dim = (120, 120, 32), show_plot = True):
    """
    Parameters
    ----------
    image : np.array
        The raw image with shape (H, W, D, C).
    label : np.array
        The raw label with shape (H, W, D).
    model : keras model
        model for prediciton.
    num_cls : int
        number of output class.
    ch_info : list
        list contains tuples (mean, std) for each channel, valid when norm mode = 2
    norm_mode : int, optional 
        norm mode. The default is 2.
    num_ch : int, optional
        number of input channel. The default is 4.
    channel : int, optional
        The channel for GT plot. The default is 0.
    loc : tuple, optional
        slice location (X, Y, Z). The default is (100, 100, 80).
    input_dim : tuple, optional
        The raw input dimension. The default is (240, 240, 155).
    input_sub_dim : tuple, optional
        The sub-volumne dimension. The default is (120, 120, 32).
    show_plot : bool, optional
        Plot the figure or not. The default is True.

    Returns
    -------
    model_label : np.array
        predicted label with shape (num_cls, H, W, D).

    """
    image_labeled = get_labeled_image(image, label, channel, num_cls)
    #model_label  = np.zeros([num_cls, input_dim[0]*2, input_dim[1]*2, int(input_sub_dim[2]*np.ceil(input_dim[2]/input_sub_dim[2]))])
    model_label  = np.zeros([num_cls, input_dim[0], input_dim[1], input_dim[2]])
    image = np.moveaxis(image, -1, 0)
    image_ = standardized(image, 2, ch_info)
    for x in range(0, image.shape[0], input_sub_dim[0]):
        for y in range(0, image.shape[1], input_sub_dim[1]):
            for z in range(0, image.shape[2], input_sub_dim[2]):
                patch = np.zeros((num_ch, *input_sub_dim))
                p = image_[:, x: x+input_sub_dim[0], y: y+input_sub_dim[1], z: z+input_sub_dim[2]]
                patch[:, 0:p.shape[1], 0:p.shape[2], 0:p.shape[3]] = p
                pred = model.predict(np.expand_dims(patch, 0))
                model_label[:, x: x+p.shape[1], y: y+p.shape[2], z: z+p.shape[3]] = pred[0][:, :p.shape[1], :p.shape[2], :p.shape[3]]
    model_label = np.moveaxis(model_label[:, 0:input_dim[0], 0:input_dim[1], 0:input_dim[2]], 0, -1)
    model_label = np.argmax(model_label, axis = -1)
    image = np.moveaxis(image, 0, -1)
    model_labeled_image = get_labeled_image(image, model_label, channel, num_cls)
    fig, ax = plt.subplots(2, 3, figsize=[10, 7])
    # plane values
    if show_plot == True:
        x, y, z = loc
        ax[0][0].imshow(np.rot90(image_labeled[x, :, :, :]))
        ax[0][0].set_ylabel('Ground Truth', fontsize=15)
        ax[0][0].set_xlabel('Sagital', fontsize=15)
        
        ax[0][1].imshow(np.rot90(image_labeled[:, y, :, :]))
        ax[0][1].set_xlabel('Coronal', fontsize=15)
        
        ax[0][2].imshow(np.squeeze(image_labeled[:, :, z, :]))
        ax[0][2].set_xlabel('Transversal', fontsize=15)
        
        ax[1][0].imshow(np.rot90(model_labeled_image[x, :, :, :]))
        ax[1][0].set_ylabel('Prediction', fontsize=15)
    
        ax[1][1].imshow(np.rot90(model_labeled_image[:, y, :, :]))
        ax[1][2].imshow(model_labeled_image[:, :, z, :])
        
        fig.subplots_adjust(wspace=0, hspace=.12)

        for i in range(2):
            for j in range(3):
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])

    return model_label