from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model
from model.attention import Conv2D_attention, ChannelAttention, SpatialAttention

from model.common import normalize, denormalize, pixel_shuffle

def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None, attention=False):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    if attention:
        x = b = Conv2D(num_filters, 3, padding='same')(x)
        # x = b = ChannelAttention(num_filters, 8)(x)
        # x = b = SpatialAttention(7)(x)
    else:
        x = b = Conv2D(num_filters, 3, padding='same')(x)
    
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling, attention=attention)

    if attention:
        # b = Conv2D_attention(b, numFilters=num_filters, kernelSize=3)
        b = Conv2D(num_filters, 3, padding='same')(b)
        # b = ChannelAttention(num_filters, 8)(b)
        # b = SpatialAttention(7)(b)
# 
    else:
        b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    
    if attention:
        # x = Conv2D_attention(x, numFilters=3, kernelSize=3)
        x = Conv2D(3, 3, padding='same')(x)
        # x = ChannelAttention(3, 8)(x)
        # x = SpatialAttention(7)(x)
    else:
        x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling, attention=False):
    if attention:
        # x = Conv2D_attention(x_in, numFilters=filters, kernelSize=3, activation='relu')
        x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
        x = ChannelAttention(filters, 8)(x)
        x = SpatialAttention(7)(x)
    else:
        x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    
    if attention:
        # x = Conv2D_attention(x, numFilters=filters, kernelSize=3)
        x = Conv2D(filters, 3, padding='same')(x)
        x = ChannelAttention(filters, 8)(x)
        x = SpatialAttention(7)(x)
    else:
        x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters, attention=False):
    def upsample_1(x, factor, attention=False, **kwargs):
        if attention:
            # x = Conv2D_attention(x, numFilters=num_filters * (factor ** 2), kernelSize=3)
            x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
            # x = ChannelAttention(num_filters * (factor ** 2), 8)(x)
            # x = SpatialAttention(7)(x)
        else:
            x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, attention=attention, name='conv2d_1_scale_2', )
    elif scale == 3:
        x = upsample_1(x, 3, attention=attention, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, attention=attention, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, attention=attention, name='conv2d_2_scale_2')

    return x
