# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:12:50 2020

@author: mousekinga82
"""
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import(
    Activation,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    UpSampling3D,
    Input,
    BatchNormalization)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

#Custom Metrics for training
def dice_coefficient(y_true, y_pred, axis=(2,3,4), epsilon=1e-5):
    """
    Parameters
    ----------
    y_true : Tensor
        true label with shape (None, num_lab, H, W, D).
    y_pred : Tensor
        predicted label with shape (None, num_lab, H, W, D).
    axis : tuple, optional
        sum axis. The default is (2,3,4) which is (H, W, D).
    epsilon : float, optional
        eposilon. The default is 1e-5.

    Returns
    -------
    dice_coeff : Tensor
        dice coefficient tensor.

    """
    dice_numerator = 2 * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_demoninator = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + epsilon
    dice_coeff = K.mean(dice_numerator / dice_demoninator)

    return dice_coeff

#Custom loss for training
def soft_dice_loss(y_true, y_pred, axis=(2,3,4), epsilon=1e-5):
    """
    Parameters
    ----------
    y_true : Tensor
        true label with shape (None, num_lab, H, W, D).
    y_pred : Tensor
        predicted label with shape (None, num_lab, H, W, D).
    axis : tuple, optional
        sum axis. The default is (2,3,4) which is (H, W, D).
    epsilon : float, optional
        eposilon. The default is 1e-5.

    Returns
    -------
    dice_loss : Tensor
        dice loss scalar tensor.

    """
    dice_numerator = 2 * K.sum(y_true * y_pred, axis = axis) + epsilon
    dice_demoninator = K.sum(y_true * y_true, axis = axis) + K.sum(y_pred * y_pred, axis = axis) + epsilon
    dice_loss = 1 - K.mean(dice_numerator / dice_demoninator)
    return dice_loss

def create_convolution_block(input_layer, n_filters, batch_normalization = False, kernel=(3,3,3), 
                             activation = True, padding = 'same', strides = (1,1,1)):
    layer = Conv3D(n_filters, kernel, strides, padding)(input_layer)
    if batch_normalization == True:
        layer = BatchNormalization(axis=1)(layer)
    if activation == True:
        return Activation('relu')(layer)
    else:
        return layer
    
def get_up_convolution(n_filters, pool_size, kernel_size=(2,2,2), strides=(2,2,2), is_deconv = False):
    if is_deconv:
        return Conv3DTranspose(filters=n_filters, kernel_size=kernel_size, strides=strides)
    else:
        return UpSampling3D(size = pool_size)

def unet_model_3d(loss_function, input_shape = (4, 160, 160, 16), pool_size = (2,2,2), num_lab = 4,
                   deconvolution = False, depth = 4, n_base_filter = 32,
                   batch_normalization = False, activation_name = 'softmax'):
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()
    #add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filter * (2 ** layer_depth), batch_normalization = batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filter * (2 ** layer_depth) * 2, batch_normalization = batch_normalization)
        if layer_depth < depth -1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2]) 
    #add levels with up-conv or up-sampling
    for layer_depth in range(depth -2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, is_deconv = deconvolution, n_filters = K.int_shape(current_layer)[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(concat       , n_filters=K.int_shape(levels[layer_depth][1])[1], batch_normalization=batch_normalization)
        current_layer = create_convolution_block(current_layer, n_filters=K.int_shape(levels[layer_depth][1])[1], batch_normalization=batch_normalization)
    
    final_convolution = Conv3D(num_lab, (1,1,1))(current_layer)
    if activation_name == 'softmax':
        act = tf.keras.activations.softmax(final_convolution, axis = 1)
    else:
        act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)
    return model