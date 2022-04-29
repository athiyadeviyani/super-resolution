import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
# from custom_image_dataset_from_directory import image_dataset_from_directory
from model.attention import ChannelAttention, SpatialAttention


def srcnn(attention=False):
    """Return the srcnn model"""

    input = tf.keras.Input(shape=(None, None, 3))
    layer1 = layers.Conv2D(64, (9,9), padding="same")(input)
    layer1_relu = layers.ReLU()(layer1)

    if attention:
        layer1_relu = ChannelAttention(64, 8)(layer1_relu)
        layer1_relu = SpatialAttention(7)(layer1_relu)

    layer2 = layers.Conv2D(32, (1,1), padding="same")(layer1_relu)
    layer2_relu = layers.ReLU()(layer2)

    if attention:
        layer2_relu = ChannelAttention(32, 8)(layer2_relu)
        layer2_relu = SpatialAttention(7)(layer2_relu)

    layer3 = layers.Conv2D(3, (5,5), padding="same")(layer2_relu)

    if attention:
        layer3 = ChannelAttention(3, 8)(layer3)
        layer3 = SpatialAttention(7)(layer3)

    output = layer3

    model = models.Model(inputs=input, outputs=output)
    print(model.shape)

    return model