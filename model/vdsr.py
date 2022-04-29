from keras.layers.convolutional import Conv2D
from keras.layers.merge import Add
from keras.layers import Input
from keras.models import Model
from model.attention import ChannelAttention, SpatialAttention

# input_shape = (600, 480, 3)  

def VDSR(attention=False):
    low_resolution_image = Input(shape=(None, None, 3))

    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(low_resolution_image)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    
    # add attention blocks
    if attention:
        processing = ChannelAttention(64, 8)(processing)
        processing = SpatialAttention(7)(processing)

    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    
    # add attention blocks
    if attention:
        processing = ChannelAttention(64, 8)(processing)
        processing = SpatialAttention(7)(processing)

    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    
    # add attention blocks
    if attention:
        processing = ChannelAttention(64, 8)(processing)
        processing = SpatialAttention(7)(processing)

    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(3 , (3, 3), padding='same', kernel_initializer='he_normal')(processing)
    
    # add attention blocks
    if attention:
        processing = ChannelAttention(64, 8)(processing)
        processing = SpatialAttention(7)(processing)
    
    Residual = processing

    high_resolution_image = Add()([low_resolution_image, Residual])
    
    model = Model(low_resolution_image, high_resolution_image)
    return model