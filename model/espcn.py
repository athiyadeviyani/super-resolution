import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
# from custom_image_dataset_from_directory import image_dataset_from_directory
import matplotlib.pyplot as plt
import os
from PIL import Image
import utils
from model.attention import ChannelAttention, SpatialAttention

upscale_factor = 4
batch_size = 32

# Size for the training images
lr_image_size = (64,64) 
hr_image_size = (lr_image_size[0] * upscale_factor, lr_image_size[1] * upscale_factor)

# Stride for the cropping images
lr_stride = (lr_image_size[0] * 3 // 4, lr_image_size[1] * 3 // 4) 
hr_stride = (lr_stride[0] * upscale_factor, lr_stride[1] * upscale_factor)

model_dir = 'models'

def espcn(attention=False):
    """Return the espcn model"""
    input = tf.keras.Input(shape=(None, None, 3))
    layer1 = layers.Conv2D(64, (5,5), padding="same")(input)
    layer1_relu = layers.ReLU()(layer1)
    if attention:
        layer1_relu = ChannelAttention(64, 8)(layer1_relu)
        layer1_relu = SpatialAttention(7)(layer1_relu)
    layer2 = layers.Conv2D(32, (3,3), padding="same")(layer1_relu)
    layer2_relu = layers.ReLU()(layer2)
    if attention:
        layer2_relu = ChannelAttention(32, 8)(layer2_relu)
        layer2_relu = SpatialAttention(7)(layer2_relu)
    layer3 = layers.Conv2D(3 * upscale_factor * upscale_factor, (3,3), padding="same")(layer2_relu)
    if attention:
        layer3 = ChannelAttention(3 * upscale_factor * upscale_factor, 3)(layer3)
        layer3 = SpatialAttention(7)(layer2_relu)
    output = tf.nn.depth_to_space(layer3, upscale_factor)

    model = models.Model(inputs=input, outputs=output)
    return model


