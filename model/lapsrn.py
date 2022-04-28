import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from model.attention import ChannelAttention, SpatialAttention

def lapsrn(attention=False):
    """Return the espcn model"""
    def residual_network(x, d=3, name=None):
        # Convolution Layers Stacks
        for _ in range(d):
            x = layers.Conv2D(64, (3,3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2))(x)

            # add attention blocks
            if attention:
                x = ChannelAttention(64, 8)(x)
                x = SpatialAttention(7)(x)
                
        # Upscale
        x = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), name=name)(x)
        return x

    # Feature Extraction Network
    input_lr = tf.keras.Input(shape=(None, None, 3))
    feature_2x = residual_network(input_lr, d=3, name='feature_2x')
    feature_4x = residual_network(feature_2x, d=3, name='feature_4x')

    # Residual Image
    residual_2x = layers.Conv2D(3, (3,3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2), name='residual_2x')(feature_2x)
    residual_4x = layers.Conv2D(3, (3,3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.2), name='residual_4x')(feature_4x)
    
    # Reconstruction Network
    upscale_2x = layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same')(input_lr)
    reconstruction_2x = layers.Add(name='reconstruction_2x')([upscale_2x, residual_2x])
    upscale_4x = layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same')(reconstruction_2x)
    reconstruction_4x = layers.Add(name='reconstruction_4x')([upscale_4x, residual_4x])

    # model = models.Model(inputs=input_lr, outputs={'2x': reconstruction_2x, '4x': reconstruction_4x})
    model = models.Model(inputs=input_lr, outputs=reconstruction_4x)
    return model

# A wrapper function is required to pass additional arguments to custom loss functions, because keras only allows 2 arguments, y_true and y_pred.
def charbonnier(epsilon=0.001):
    def loss(y_true, y_pred):
        return tf.reduce_mean(tf.math.sqrt(tf.math.square(y_true-y_pred) + tf.math.square(epsilon)))
    return loss