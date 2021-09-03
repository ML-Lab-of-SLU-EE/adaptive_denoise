import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, applications
from tensorflow import keras
import tensorflow_addons as tfa
from .mpr_networks import CAB, CBAM


'''UNet_Generator'''
class UNetDown(Model):
    def __init__(self, in_channel, out_channel, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.normalize = normalize
        self.dropout = dropout
        self.body = self.architecture_init()

    def architecture_init(self):
        body = [layers.Conv2D(filters=self.out_channel, kernel_size=4, strides=2, padding='same', use_bias=False)]
        if self.normalize:
            body.append(tfa.layers.InstanceNormalization())
        body.append(layers.LeakyReLU(alpha=0.2))
        if self.dropout:
            body.append(layers.Dropout(self.dropout))
        return Sequential(body)

    def call(self, inputs):
        return self.body(inputs)

class UNetUp(Model):
    def __init__(self, in_channel, out_channel, dropout=0.0):
        super(UNetUp, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dropout = dropout
        self.body = self.architecture_init()

    def architecture_init(self):
        body = [layers.Conv2DTranspose(filters=self.out_channel, kernel_size=4, strides=2, padding='same', use_bias=False),
                tfa.layers.InstanceNormalization(),
                layers.ReLU()]
        if self.dropout:
            body.append(layers.Dropout(self.dropout))
        return Sequential(body)

    def call(self, inputs, skip_input):
        x = self.body(inputs)
        x = tf.concat([x, skip_input], axis=-1)
        return x


class UNet_Generator_3_images(Model):
    def __init__(self, in_channel=3, out_channel=9, mask_num=1, name='UNet_Generator'):
        super(UNet_Generator_3_images, self).__init__(name=name)
        self.mask_num = mask_num

        self.down1 = UNetDown(in_channel, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = Sequential([layers.UpSampling2D(),
                                 layers.Conv2D(filters=out_channel, kernel_size=4, strides=1, padding='same')])

        self.haze_mask, self.rain_mask, self.dark_mask = self.architecture_init(self.mask_num)

    def architecture_init(self, mask_num):
        haze_mask = []
        rain_mask = []
        dark_mask = []

        for _ in range(mask_num):
            haze_mask += [layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                          tfa.layers.InstanceNormalization(),
                          layers.LeakyReLU()]

            rain_mask += [layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                          tfa.layers.InstanceNormalization(),
                          layers.LeakyReLU()]

            dark_mask += [layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same'),
                          tfa.layers.InstanceNormalization(),
                          layers.LeakyReLU()]
        return Sequential(haze_mask), Sequential(rain_mask), Sequential(dark_mask)


    def call(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        out = self.final(u7)


        haze = self.haze_mask(out[..., 0:3]) + x
        rain = self.rain_mask(out[..., 3:6]) + x
        dark = self.dark_mask(out[..., 6:9]) + x

        haze = tf.nn.tanh(haze)
        rain = tf.nn.tanh(rain)
        dark = tf.nn.tanh(dark)

        return [haze, rain, dark]


class UNet_Generator_single_image(Model):
    def __init__(self, in_channel=3, out_channel=3, mask_num=1, name='UNet_Generator'):
        super(UNet_Generator_single_image, self).__init__(name=name)
        self.mask_num = mask_num

        self.down1 = UNetDown(in_channel, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = Sequential([layers.UpSampling2D(),
                                 layers.Conv2D(filters=out_channel, kernel_size=4, strides=1, padding='same')])

    def call(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        out = self.final(u7)
        out += x
        out = tf.nn.tanh(out)

        return out


'''ResNet Generator'''
class new_ResNet_Generator_single_image(Model):
    def __init__(self, ngf=64, n_block=9, name='ResNet_Generator', use_mask=False, attention=None):
        super(new_ResNet_Generator_single_image, self).__init__(name=name)
        self.ngf = ngf
        self.n_block = n_block
        self.use_mask = use_mask
        self.attention = attention

        self.from_rgb = Sequential([
            new_Conv(filters=self.ngf, kernel_size=7, strides=1, pad=3, pad_type='reflect'),
            tfa.layers.InstanceNormalization(name='from_rgb_ins_norm'),
            layers.ReLU(name='from_rgb_relu')
        ])

        self.body1, self.body2, self.body3 = self.architecture_init()

        self.to_rgb = [new_Conv(filters=3, kernel_size=7, strides=1, pad=3, pad_type='reflect', use_bias=True)]
        if self.use_mask == True:
            self.to_rgb += [layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same', use_bias=False),
                            tfa.layers.InstanceNormalization(),
                            layers.ReLU(),
                            layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', use_bias=False),
                            tfa.layers.InstanceNormalization(),
                            layers.ReLU(),
                            layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same', use_bias=False),
                            tfa.layers.InstanceNormalization(),
                            layers.ReLU()]
        self.to_rgb = Sequential(self.to_rgb)
    def architecture_init(self):
        n_downsamplings = 2
        dim = self.ngf
        body1 = []
        body2 = []
        body3 = []
        for i in range(n_downsamplings):
            dim *= 2
            body1 += [layers.Conv2D(filters=dim, kernel_size=3, strides=2, padding='same', use_bias=False),
                     tfa.layers.InstanceNormalization(),
                     layers.ReLU()]
            # print('body1中当前的dim为：', dim)

        for i in range(self.n_block):
            body2 += [new_Resblock(dim=dim, attention=self.attention)]
            # print('body2中当前的dim为：', dim)

        for i in range(n_downsamplings):
            dim //= 2
            body3 += [layers.Conv2DTranspose(filters=dim, kernel_size=3, strides=2, padding='same', use_bias=False),
                      tfa.layers.InstanceNormalization(),
                      layers.ReLU()]

        return Sequential(body1), Sequential(body2), Sequential(body3)


    def call(self, inputs):
        x = self.from_rgb(inputs)
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x = self.to_rgb(x)
        if self.use_mask == True:
            x += inputs
        x = tf.nn.tanh(x)

        return x

class new_ResNet_Generator_single_image_with_tfl(Model):
    def __init__(self, ngf=64, n_block=9, name='ResNet_Generator', attention=None):
        super(new_ResNet_Generator_single_image_with_tfl, self).__init__(name=name)
        self.ngf = ngf
        self.n_block = n_block
        self.attention = attention

        self.from_rgb = Sequential([
            new_Conv(filters=self.ngf, kernel_size=7, strides=1, pad=3, pad_type='reflect'),
            tfa.layers.InstanceNormalization(name='from_rgb_ins_norm'),
            layers.ReLU(name='from_rgb_relu')
        ])

        self.body1, self.body2, self.body3 = self.architecture_init()

        self.to_rgb = [new_Conv(filters=3, kernel_size=7, strides=1, pad=3, pad_type='reflect', use_bias=True)]

        self.to_rgb = Sequential(self.to_rgb)
        self.tfl1 = Sequential([layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same', use_bias=False),
                                tfa.layers.InstanceNormalization(),
                                layers.LeakyReLU(),
                                layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', use_bias=False),
                                tfa.layers.InstanceNormalization(),
                                layers.LeakyReLU(),
                                layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same', use_bias=False),
                                tfa.layers.InstanceNormalization(),
                                layers.LeakyReLU()])

        self.tfl2 = Sequential([layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same', use_bias=False),
                                tfa.layers.InstanceNormalization(),
                                layers.LeakyReLU(),
                                layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', use_bias=False),
                                tfa.layers.InstanceNormalization(),
                                layers.LeakyReLU(),
                                layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same', use_bias=False),
                                tfa.layers.InstanceNormalization(),
                                layers.LeakyReLU()])
    def architecture_init(self):
        n_downsamplings = 2
        dim = self.ngf
        body1 = []
        body2 = []
        body3 = []
        for i in range(n_downsamplings):
            dim *= 2
            body1 += [layers.Conv2D(filters=dim, kernel_size=3, strides=2, padding='same', use_bias=False),
                     tfa.layers.InstanceNormalization(),
                     layers.ReLU()]
            # print('body1中当前的dim为：', dim)

        for i in range(self.n_block):
            body2 += [new_Resblock(dim=dim, attention=self.attention)]
            # print('body2中当前的dim为：', dim)

        for i in range(n_downsamplings):
            dim //= 2
            body3 += [layers.Conv2DTranspose(filters=dim, kernel_size=3, strides=2, padding='same', use_bias=False),
                      tfa.layers.InstanceNormalization(),
                      layers.ReLU()]

        return Sequential(body1), Sequential(body2), Sequential(body3)


    def call(self, inputs):
        x = self.from_rgb(inputs)
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x = self.to_rgb(x)  # get res image

        tfl1 = self.tfl1(x)
        tfl2 = self.tfl2(x)
        final = (tfl1 + inputs) * tfl2
        final = tf.nn.tanh(final)
        return final


class new_Conv(keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=3, strides=1, pad=0, pad_type='reflect', use_bias=False):
        super(new_Conv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad = pad
        self.pad_type = pad_type

        self.conv = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, use_bias=use_bias)

    def call(self, x):
        if self.pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]], mode='REFLECT')

        else:
            x = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])

        x = self.conv(x)

        return x

class new_Resblock(Model):
    def __init__(self, dim, pad_type='reflect', attention=None):
        super(new_Resblock, self).__init__()
        self.dim = dim
        self.pad_type = pad_type
        self.attention = attention

        self.body = self.architecture_init()


    def architecture_init(self):
        body = [new_Conv(filters=self.dim, kernel_size=3, pad=1, pad_type='reflect'),
                tfa.layers.InstanceNormalization(),
                layers.ReLU(),
                new_Conv(filters=self.dim, kernel_size=3, pad=1, pad_type='reflect'),
                tfa.layers.InstanceNormalization()]
        if self.attention == 'CBAM':
            body.append(CBAM(gate_channel=self.dim))

        elif self.attention == 'CAB':
            body.append(CAB(n_feat=self.dim, kernel_size=3, reduction=16, bias=False))
        return Sequential(body)

    def call(self, input):
        x = self.body(input)
        result = x + input
        return result


# 新实现的Dis参数相同
class new_Conv_Discriminator(Model):
    def __init__(self, img_size=256, input_nc=3, ndf=64, n_layers=3, name='Conv_Discriminator'):
        super(new_Conv_Discriminator, self).__init__(name=name)
        self.image_size = img_size
        self.input_nc = input_nc
        self.ndf = ndf
        self.n_layers = n_layers
        self.body1, self.body2, self.body3 = self.architecture_init()

    def architecture_init(self):
        body1 = []
        body2 = []
        body3 = []

        body1 += [layers.Conv2D(self.ndf, kernel_size=4, strides=2, padding='same'),
                 layers.LeakyReLU(alpha=0.2)]
        dim = self.ndf
        dim_ = self.ndf
        for i in range(self.n_layers-1):
            dim = min(dim * 2, dim_ * 8)
            body1 += [layers.Conv2D(filters=dim, kernel_size=4, strides=2, padding='same', use_bias=False),
                      tfa.layers.InstanceNormalization(),
                      layers.LeakyReLU(alpha=0.2)]

        dim = min(dim * 2, dim_ * 8)
        body2 += [layers.Conv2D(filters=dim, kernel_size=4, strides=1, padding='same', use_bias=False),
                  tfa.layers.InstanceNormalization(),
                  layers.LeakyReLU(alpha=0.2)]

        body3 += [layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same')]

        return Sequential(body1), Sequential(body2), Sequential(body3)


    def call(self, inputs):
        x = self.body1(inputs)
        x = self.body2(x)
        x = self.body3(x)
        return x


class SFT_layer(Model):
    def __init__(self):
        super(SFT_layer, self).__init__()
        condition_conv = [layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same'),
                          layers.LeakyReLU(alpha=0.2),
                          layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same'),
                          layers.LeakyReLU(alpha=0.2),
                          layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
                          layers.LeakyReLU(alpha=0.2)]

        scale_conv = [layers.Conv2D(filters=63, kernel_size=3, strides=1, padding='same'),
                      layers.LeakyReLU(alpha=0.2),
                      layers.Conv2D(filters=63, kernel_size=3, strides=1, padding='same'),
                      layers.LeakyReLU(alpha=0.2)]

        shift_conv = [layers.Conv2D(filters=63, kernel_size=3, strides=1, padding='same'),
                      layers.LeakyReLU(alpha=0.2),
                      layers.Conv2D(filters=63, kernel_size=3, strides=1, padding='same'),
                      layers.LeakyReLU(alpha=0.2)]

        self.condition_conv = Sequential(condition_conv)
        self.scale_conv = Sequential(scale_conv)
        self.shift_conv = Sequential(shift_conv)

    def call(self, feature, input):
        input_condition = self.condition_conv(input)
        scaled_feature = self.scale_conv(input_condition) * feature
        shifted_feature = scaled_feature + self.shift_conv(input_condition)

        return shifted_feature






