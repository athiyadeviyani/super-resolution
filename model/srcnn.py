import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
# from custom_image_dataset_from_directory import image_dataset_from_directory
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

def srcnn():
    """Return the srcnn model"""
    model = tf.keras.Sequential([
        layers.Input(shape=(None, None, 3)),
        layers.Conv2D(64, (9,9), padding="same"),
        layers.ReLU(),
        layers.Conv2D(32, (1,1), padding="same"),
        layers.ReLU(),
        layers.Conv2D(3, (5,5), padding="same")
    ])

    return model