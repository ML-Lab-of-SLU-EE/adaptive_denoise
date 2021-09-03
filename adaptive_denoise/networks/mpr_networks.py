# from cyclegan_networks import *
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, Sequential, Model


'''Components'''

class CALayer(Model):
    def __init__(self, in_channel, reduction_ratio=16, bias=False):
        super(CALayer, self).__init__()
        self.in_channel = in_channel
        self.reduction_ratio = reduction_ratio
        self.bias = bias

        self.body = self.architecture_init()

    def architecture_init(self):
        body = [layers.Conv2D(filters=self.in_channel // self.reduction_ratio, kernel_size=1, use_bias=self.bias),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(filters=self.in_channel, kernel_size=1, use_bias=self.bias)]

        return Sequential(body)

    def call(self, inputs):
        x = tf.reduce_mean(tf.reduce_mean(inputs, axis=1, keepdims=True), axis=2, keepdims=True)

        x = self.body(x)
        x = tf.nn.sigmoid(x)
        # print('x.shape:', x.shape)
        return inputs * x


# TODO: CAB模块只用来提取特征，不改变尺寸和通道数
class CAB(Model):
    def __init__(self, in_channel, kernel_size, reduction, bias, act):  # n_feat是输出通道数
        super(CAB, self).__init__()

        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.bias = bias
        self.activation = act
        self.CA = CALayer(in_channel, reduction, bias=bias)
        self.body = self.architecture_init()

    def architecture_init(self):
        body = [layers.Conv2D(filters=self.in_channel, kernel_size=self.kernel_size, strides=1, padding='same', use_bias=self.bias),
                tfa.layers.InstanceNormalization(),
                self.activation,
                layers.Conv2D(filters=self.in_channel, kernel_size=self.kernel_size, strides=1, padding='same', use_bias=self.bias),
                tfa.layers.InstanceNormalization(),]

        return Sequential(body)

    def call(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class SkipUpSample(Model):
    def __init__(self, in_channel):
        super(SkipUpSample, self).__init__()
        self.up = Sequential([layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
                              layers.Conv2D(filters=in_channel, kernel_size=1, strides=1, use_bias=False),
                              tfa.layers.InstanceNormalization(),
                              layers.LeakyReLU(0.2)])

    def call(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class DownSample_new(Model):
    def __init__(self, in_channel, s_factor):
        super(DownSample_new, self).__init__()
        conv = [layers.Conv2D(filters=in_channel + s_factor, kernel_size=4, strides=2, use_bias=False, padding='same'),
                layers.LeakyReLU(alpha=0.2)]
        self.conv = Sequential(conv)

    def call(self, x):
        x = self.conv(x)
        return x

class UpSample(Model):
    def __init__(self, in_channels):
        super(UpSample, self).__init__()
        self.up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv = layers.Conv2D(filters=in_channels, kernel_size=1, strides=1, use_bias=False)

    def call(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class SkipUpSample(Model):
    def __init__(self, in_channel):
        super(SkipUpSample, self).__init__()
        self.up = Sequential([layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
                              layers.Conv2D(filters=in_channel, kernel_size=1, strides=1, use_bias=False)])

    def call(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class Encoder(Model):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = Sequential([CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)])
        self.encoder_level2 = Sequential([CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(2)])
        self.encoder_level3 = Sequential([CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)])

        self.down12 = DownSample_new(n_feat, scale_unetfeats)
        self.down23 = DownSample_new(n_feat + scale_unetfeats, scale_unetfeats)

        if csff:
            self.csff_enc1 = layers.Conv2D(filters=n_feat, kernel_size=1, use_bias=bias)
            self.csff_enc2 = layers.Conv2D(filters=n_feat + scale_unetfeats, kernel_size=1, use_bias=bias)
            self.csff_enc3 = layers.Conv2D(filters=n_feat + (scale_unetfeats * 2), kernel_size=1, use_bias=bias)

            self.csff_dec1 = layers.Conv2D(filters=n_feat, kernel_size=1, use_bias=bias)
            self.csff_dec2 = layers.Conv2D(filters=n_feat + scale_unetfeats, kernel_size=1, use_bias=bias)
            self.csff_dec3 = layers.Conv2D(filters=n_feat + (scale_unetfeats * 2), kernel_size=1, use_bias=bias)

    def call(self, x, encoder_outs=None, decoder_outs=None):

        enc1 = self.encoder_level1(x)

        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class CBAM(Model):
    def __init__(self, gate_channel, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.gate_channels = gate_channel
        self.reduction_ratio = reduction_ratio

        self.mlp = self.architecture_init()
        self.conv = layers.Conv2D(filters=1, kernel_size=7, strides=1, padding='same')

    def architecture_init(self):
        mlp = [layers.Flatten(),
               layers.Dense(units=self.gate_channels // self.reduction_ratio),
               layers.ReLU(),
               layers.Dense(units=self.gate_channels)]

        return Sequential(mlp)

    def call(self, x):
        # compute channel attention mask
        maxpool_channel = tf.reduce_max(tf.reduce_max(x, axis=1, keepdims=True), axis=2, keepdims=True)
        avgpool_channel = tf.reduce_mean(tf.reduce_mean(x, axis=1, keepdims=True), axis=2, keepdims=True)

        max_pool_result = self.mlp(maxpool_channel)
        avg_pool_result = self.mlp(avgpool_channel)

        channel_attention = tf.nn.sigmoid(max_pool_result + avg_pool_result)
        channel_refined_feature = channel_attention * x

        # compute spatial attention mask
        maxpool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        avgpool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        spatial_attention = tf.nn.sigmoid(self.conv(tf.concat([maxpool_spatial, avgpool_spatial], axis=-1)))

        refined_feature = channel_refined_feature * spatial_attention

        return refined_feature


class BAM(Model):
    def __init__(self, gate_channel, reduction_ratio=16):
        super(BAM, self).__init__()
        self.gate_channel = gate_channel
        self.reduction_ratio = reduction_ratio

        self.mlp, self.convs = self.architecture_init()

    def architecture_init(self):
        mlp = [layers.Flatten(),
               layers.Dense(units=self.gate_channel // self.reduction_ratio),
               layers.BatchNormalization(),
               layers.ReLU(),
               layers.Dense(units=self.gate_channel)]

        convs = [layers.Conv2D(filters=self.gate_channel // self.reduction_ratio, kernel_size=1, padding='same'),
                 layers.BatchNormalization(),
                 layers.ReLU(),
                 layers.Conv2D(filters=self.gate_channel // self.reduction_ratio, kernel_size=3, padding='same', dilation_rate=4),
                 layers.Conv2D(filters=1, kernel_size=1, padding='same')]

        return Sequential(mlp), Sequential(convs)

    def call(self, x):
        avgpool_channel = tf.reduce_mean(tf.reduce_mean(x, axis=1, keepdims=True), axis=2, keepdims=True)
        channel_attention = self.mlp(avgpool_channel)

        avgpool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        spatial_attention = self.convs(avgpool_spatial)

        BAM_attention = tf.nn.sigmoid(channel_attention + spatial_attention)
        refined_feature = x + x * BAM_attention

        return refined_feature


# 定义Decoder
class Decoder(Model):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = Sequential([CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)])
        self.decoder_level2 = Sequential([CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(2)])
        self.decoder_level3 = Sequential([CAB(n_feat+scale_unetfeats*2, kernel_size, reduction, bias=bias, act=act) for _ in range(2)])

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat)
        self.up32 = SkipUpSample(n_feat+scale_unetfeats)

    def call(self, outs):
        enc1, enc2, enc3 = outs

        # print('enc3的尺寸为：', enc3.shape)
        # print('enc2的尺寸为：', enc2.shape)
        # print('enc1的尺寸为：', enc1.shape)
        dec3 = self.decoder_level3(enc3)

        # print('dec3.shape', dec3.shape)
        # print('enc2.shape', enc2.shape)
        # print('skip_attn2(enc2)',self.skip_attn2(enc2).shape)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        # print('dec2.shape', dec2.shape)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


class SAM(Model):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = layers.Conv2D(filters=n_feat, kernel_size=kernel_size, strides=1, padding='same', use_bias=bias)  # 处理上一个stage的feature
        self.conv2 = layers.Conv2D(filters=3, kernel_size=kernel_size, strides=1, padding='same', use_bias=bias)  # 输出restore的图像，可以去掉
        self.conv3 = layers.Conv2D(filters=n_feat, kernel_size=kernel_size, strides=1, padding='same', use_bias=bias)  # 修改原始图片的通道数

    def call(self, feature, x_image):
        x1 = self.conv1(feature)
        img = self.conv2(feature) + x_image
        x2 = tf.nn.sigmoid(self.conv3(img))
        x1 = x1 * x2
        return x1, img


class ORB(Model):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):  # 1个ORB模块中有n_cab个CAB模块
        super(ORB, self).__init__()
        body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        body.append(layers.Conv2D(filters=n_feat, kernel_size=kernel_size, strides=1, padding='same'))
        self.body = Sequential(body)

    def call(self, x):
        res = self.body(x)
        res += x
        return res


class ORSNet(Model):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat)
        self.up_dec1 = UpSample(n_feat)

        self.up_enc2 = Sequential([UpSample(n_feat+scale_unetfeats), UpSample(n_feat)])
        self.up_dec2 = Sequential([UpSample(n_feat+scale_unetfeats), UpSample(n_feat)])

        self.conv_enc1 = layers.Conv2D(filters=(n_feat+scale_orsnetfeats), kernel_size=1, use_bias=bias)
        self.conv_enc2 = layers.Conv2D(filters=(n_feat + scale_orsnetfeats), kernel_size=1, use_bias=bias)
        self.conv_enc3 = layers.Conv2D(filters=(n_feat + scale_orsnetfeats), kernel_size=1, use_bias=bias)

        self.conv_dec1 = layers.Conv2D(filters=(n_feat + scale_orsnetfeats), kernel_size=1, use_bias=bias)
        self.conv_dec2 = layers.Conv2D(filters=(n_feat + scale_orsnetfeats), kernel_size=1, use_bias=bias)
        self.conv_dec3 = layers.Conv2D(filters=(n_feat + scale_orsnetfeats), kernel_size=1, use_bias=bias)

    def call(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x



'''Networks'''
'''Single images'''
class MPRNet_UNet_single_image_1stage(Model):
    def __init__(self, n_feat=64, scale_unetfeats=64, kernel_size=3, reduction=16, bias=False, act=layers.ReLU(), name='MPRNet_Generator'):
        super(MPRNet_UNet_single_image_1stage, self).__init__()
        self.shallow_featuer1 = Sequential([layers.Conv2D(filters=n_feat, kernel_size=kernel_size, use_bias=bias, padding='same'),
                                           CAB(in_channel=n_feat, kernel_size=kernel_size, reduction=reduction, bias=False, act=act)])

        # encoder components
        self.encoder_level1 = Sequential([CAB(n_feat,                         kernel_size, reduction, bias=bias, act=act) for _ in range(2)])
        self.encoder_level2 = Sequential([CAB(n_feat + scale_unetfeats,       kernel_size, reduction, bias=bias, act=act) for _ in range(2)])
        self.encoder_level3 = Sequential([CAB(n_feat + (scale_unetfeats * 3), kernel_size, reduction, bias=bias, act=act) for _ in range(2)])

        self.down12 = DownSample_new(n_feat, scale_unetfeats)
        self.down23 = DownSample_new(n_feat + scale_unetfeats, scale_unetfeats*2)

        # decoder components
        self.decoder_level1 = Sequential([CAB(n_feat,                         kernel_size, reduction, bias=bias, act=act) for _ in range(2)])
        self.decoder_level2 = Sequential([CAB(n_feat + scale_unetfeats,       kernel_size, reduction, bias=bias, act=act) for _ in range(2)])
        self.decoder_level3 = Sequential([CAB(n_feat + (scale_unetfeats * 3), kernel_size, reduction, bias=bias, act=act) for _ in range(2)])

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats)  # (60, 20)

        # to_rgb
        self.to_rgb = layers.Conv2D(filters=3, kernel_size=kernel_size, strides=1, padding='same', use_bias=bias)


    def call(self, x, encoder_outs=None, decoder_outs=None):

        shallow = self.shallow_featuer1(x)

        enc1 = self.encoder_level1(shallow)  # enc1.shape=[b, 256, 256, 64]
        x = self.down12(enc1)  # x.shape=[b, 128, 128, 64]

        enc2 = self.encoder_level2(x)  # enc2.shape=[b, 128, 128, 128]
        x = self.down23(enc2)  # x.shape=[b, 64, 64, 128]

        enc3 = self.encoder_level3(x)  # enc2.shape=[b, 64, 64, 256]

        dec3 = self.decoder_level3(enc3)
        # print('dec3.shape', dec3.shape)

        x = self.up32(dec3, self.skip_attn2(enc2))
        # print('first upsample:', x.shape)

        dec2 = self.decoder_level2(x)
        # print('dec2.shape', dec2.shape)
        x = self.up21(dec2, self.skip_attn1(enc1))

        dec1 = self.decoder_level1(x)
        # print('dec1.shape', dec1.shape)

        out = self.to_rgb(dec1)

        out = tf.nn.tanh(out)

        return out


class MPRNet_UNet_single_image_2stages(Model):
    def __init__(self, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False, act=layers.ReLU(), name='MPRNet_multi_Generator'):
        super(MPRNet_UNet_single_image_2stages, self).__init__()
        self.shallow_featuer1 = Sequential([layers.Conv2D(filters=n_feat, kernel_size=kernel_size, use_bias=bias, padding='same'),
                                           CAB(in_channel=n_feat, kernel_size=kernel_size, reduction=reduction, bias=False, act=act)])

        # encoder components
        self.encoder_level1 = Sequential([CAB(n_feat,                         kernel_size, reduction, bias=bias, act=act) for _ in range(2)])
        self.encoder_level2 = Sequential([CAB(n_feat + scale_unetfeats,       kernel_size, reduction, bias=bias, act=act) for _ in range(2)])
        self.encoder_level3 = Sequential([CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)])

        self.down12 = DownSample_new(n_feat, scale_unetfeats)
        self.down23 = DownSample_new(n_feat + scale_unetfeats, scale_unetfeats)

        # decoder components
        self.decoder_level1 = Sequential([CAB(n_feat,                         kernel_size, reduction, bias=bias, act=act) for _ in range(2)])
        self.decoder_level2 = Sequential([CAB(n_feat + scale_unetfeats,       kernel_size, reduction, bias=bias, act=act) for _ in range(2)])
        self.decoder_level3 = Sequential([CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)])

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats)  # (60, 20)

        # sam
        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)

        # concat
        self.concat12 = layers.Conv2D(filters=n_feat+scale_orsnetfeats, kernel_size=kernel_size, padding='same', use_bias=bias)

        # orsnet components
        self.orb1 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat)  # 对enc2和dec2进行上采样，shape: [b, 128, 128, 60]->[b, 256, 256, 40]
        self.up_dec1 = UpSample(n_feat)

        self.up_enc2 = Sequential([UpSample(n_feat + scale_unetfeats), UpSample(n_feat)])  # 对enc3和dec3进行两次上采样，shape: [b, 64, 64, 80]->[b, 128, 128, 60]->[b, 256, 256, 40]
        self.up_dec2 = Sequential([UpSample(n_feat + scale_unetfeats), UpSample(n_feat)])

        self.conv_enc1 = layers.Conv2D(filters=n_feat+scale_orsnetfeats, kernel_size=1, padding='same', use_bias=bias)  # 以下的6层卷积为csff结构
        self.conv_enc2 = layers.Conv2D(filters=n_feat + scale_orsnetfeats, kernel_size=1, padding='same', use_bias=bias)
        self.conv_enc3 = layers.Conv2D(filters=n_feat + scale_orsnetfeats, kernel_size=1, padding='same', use_bias=bias)

        self.conv_dec1 = layers.Conv2D(filters=n_feat + scale_orsnetfeats, kernel_size=1, padding='same', use_bias=bias)
        self.conv_dec2 = layers.Conv2D(filters=n_feat + scale_orsnetfeats, kernel_size=1, padding='same', use_bias=bias)
        self.conv_dec3 = layers.Conv2D(filters=n_feat + scale_orsnetfeats, kernel_size=1, padding='same', use_bias=bias)

        # tail
        self.tail = layers.Conv2D(filters=3, kernel_size=kernel_size, padding='same', use_bias=bias)


    def call(self, inputs, encoder_outs=None, decoder_outs=None):

        shallow = self.shallow_featuer1(inputs)  # shallow.shape=[b, 256, 256, 40]

        # 经过encoder1
        enc1 = self.encoder_level1(shallow) # enc1.shape=[b, 256, 256, 40]
        x = self.down12(enc1)  # x.shape=[b, 128, 128, 40]

        enc2 = self.encoder_level2(x)  # enc2.shape=[b, 128, 128,60]
        x = self.down23(enc2)  # x.shape=[b, 64, 64, 60]

        enc3 = self.encoder_level3(x)  # enc2.shape=[b, 64, 64, 80]

        dec3 = self.decoder_level3(enc3)  # dec3.shape=[b, 64, 64, 80]
        # print('dec3.shape', dec3.shape)

        x = self.up32(dec3, self.skip_attn2(enc2))
        # print('first upsample:', x.shape)

        dec2 = self.decoder_level2(x)  # dec2.shape=[b, 128, 128, 60]
        # print('dec2.shape', dec2.shape)
        x = self.up21(dec2, self.skip_attn1(enc1))

        dec1 = self.decoder_level1(x)  # dec1.shape=[b, 256, 256, 40]
        # print('dec1.shape', dec1.shape)

        # 经过sam1
        sam_feats, stage1_img = self.sam12(dec1, inputs)

        # 经过ORSNet
        orb1_in = self.concat12(tf.concat([shallow, sam_feats], axis=-1))  # orb1_in.shape=[b, 256, 256, 56]
        # print('orb1_in.shape', orb1_in.shape)

        orb1_out = self.orb1(orb1_in)  # orb1_out.shape=[b, 256, 256, 56]
        # print('orb1_out.shape', orb1_out.shape)
        orb2_in = orb1_out + self.conv_enc1(enc1) + self.conv_dec1(dec1)  # orb2_in.shape=[b, 256, 256, 56]
        # print('orb2_in.shape', orb2_in.shape)

        orb2_out = self.orb2(orb2_in)  #orb2_out.shape=[b, 256, 256, 56]
        # print('orb2_out.shape', orb2_out.shape)
        orb3_in = orb2_out + self.conv_enc2(self.up_enc1(enc2)) + self.conv_dec2(self.up_dec1(dec2))
        # print('orb3_in.shape', orb3_in.shape)

        orb3_out = self.orb3(orb3_in)
        # print('orb3_out.shape', orb3_out.shape)
        tail_in = orb3_out + self.conv_enc3(self.up_enc2(enc3)) + self.conv_dec3(self.up_dec2(dec3))
        out = self.tail(tail_in)
        out = tf.nn.tanh(out)

        return out


'''multi_images'''
class Cascade_MPRNet_UNet_3stages(Model):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False, name='MPR_Generator'):
        super(Cascade_MPRNet_UNet_3stages, self).__init__()

        act = layers.ReLU()
        self.shallow_feat1 = Sequential([layers.Conv2D(filters=n_feat, kernel_size=kernel_size, use_bias=bias, padding='same'), CAB(n_feat, kernel_size, reduction, bias=bias, act=act)])
        self.shallow_feat2 = Sequential([layers.Conv2D(filters=n_feat, kernel_size=kernel_size, use_bias=bias, padding='same'), CAB(n_feat, kernel_size, reduction, bias=bias, act=act)])
        self.shallow_feat3 = Sequential([layers.Conv2D(filters=n_feat, kernel_size=kernel_size, use_bias=bias, padding='same'), CAB(n_feat, kernel_size, reduction, bias=bias, act=act)])

        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage3_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam3 = SAM(n_feat, kernel_size=1, bias=bias)

        self.concat12 = layers.Conv2D(filters=n_feat, kernel_size=kernel_size, use_bias=bias, padding='same')
        # self.concat23 = layers.Conv2D(filters=n_feat+scale_orsnetfeats, kernel_size=kernel_size, use_bias=bias, padding='same')
        self.concat23 = layers.Conv2D(filters=n_feat, kernel_size=kernel_size, use_bias=bias, padding='same')
        self.tail = layers.Conv2D(filters=out_c, kernel_size=kernel_size, use_bias=bias, padding='same')

    def call(self, x3_img):
        # stage1
        x1 = self.shallow_feat1(x3_img)  # x1.shape=[b, 256, 256, 40]
        # print(x1.shape)
        feat1 = self.stage1_encoder(x1)  # feat1=[enc1, enc2, enc3]
        res1 = self.stage1_decoder(feat1) # res1=[dec1 dec2 dec3]
        x2_samfeats, stage1_img = self.sam12(res1[0], x3_img)
        # print('stage1_img.shape', stage1_img.shape)
        # print('x1_samfeats.shape', x2_samfeats.shape)

        # stage2
        x2 = self.shallow_feat2(x2_samfeats)
        x2_cat = self.concat12(tf.concat([x2, x2_samfeats], axis=-1))
        # print('x2_cat.shape', x2_cat.shape)
        feat2 = self.stage2_encoder(x2_cat, feat1, res1)
        res2 = self.stage2_decoder(feat2)
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)

        # stage3
        x3 = self.shallow_feat3(x3_samfeats)
        x3_cat = self.concat23(tf.concat([x3, x3_samfeats], axis=-1))
        # print('x3_cat.shape', x3_cat.shape)
        feat3 = self.stage3_encoder(x3_cat, feat2, res2)
        res3 = self.stage3_decoder(feat3)
        z, stage3_img = self.sam3(res3[0], x3_img)  # z isn't be used caused 1 conv layer not be used

        stage1_img = tf.nn.tanh(stage1_img)
        stage2_img = tf.nn.tanh(stage2_img)
        stage3_img = tf.nn.tanh(stage3_img)

        return [stage1_img, stage2_img, stage3_img]

'''Original MPRNet'''
class MPRNet(Model):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False, name='MPR_Generator'):
        super(MPRNet, self).__init__()

        act = layers.ReLU()
        self.shallow_feat1 = Sequential([layers.Conv2D(filters=n_feat, kernel_size=kernel_size, use_bias=bias, padding='same'), CAB(n_feat, kernel_size, reduction, bias=bias, act=act)])
        self.shallow_feat2 = Sequential([layers.Conv2D(filters=n_feat, kernel_size=kernel_size, use_bias=bias, padding='same'), CAB(n_feat, kernel_size, reduction, bias=bias, act=act)])
        self.shallow_feat3 = Sequential([layers.Conv2D(filters=n_feat, kernel_size=kernel_size, use_bias=bias, padding='same'), CAB(n_feat, kernel_size, reduction, bias=bias, act=act)])

        # csff过程
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)

        self.concat12 = layers.Conv2D(filters=n_feat, kernel_size=kernel_size, use_bias=bias, padding='same')
        self.concat23 = layers.Conv2D(filters=n_feat+scale_orsnetfeats, kernel_size=kernel_size, use_bias=bias, padding='same')
        self.tail = layers.Conv2D(filters=out_c, kernel_size=kernel_size, use_bias=bias, padding='same')

    def call(self, x3_img):
        H = x3_img.shape[1]
        W = x3_img.shape[2]

        # 第2阶段将图片分成2张
        x2top_img = x3_img[:, 0:int(H / 2), :, :]
        x2bot_img = x3_img[:, int(H / 2):H, :, :]
        # print('x2bot_img.shape', x2bot_img.shape)


        # 第4阶段将图片分成4张
        x1ltop_img = x2top_img[:, :, 0:int(W / 2), :]
        x1rtop_img = x2top_img[:, :, int(W / 2):W, :]
        x1lbot_img = x2bot_img[:, :, 0:int(W / 2), :]
        x1rbot_img = x2bot_img[:, :, int(W / 2):W, :]
        # print('x1rbot_img.shape', x1rbot_img.shape)


        # stage1
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        # print('x1ltop.shape', x1ltop.shape)


        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        # print('feat1_rbot.shape[0]', feat1_rbot[0].shape)

        # 拼接图片
        feat1_top = [tf.concat([k, v], 2) for k, v in zip(feat1_ltop, feat1_rtop)]
        feat1_bot = [tf.concat([k, v], 2) for k, v in zip(feat1_lbot, feat1_rbot)]

        # print('stage1拼接后图像的尺寸: ', feat1_top[0].shape)
        # print('stage1拼接后图像的尺寸: ', feat1_bot[1].shape)
        # print('stage1拼接后图像的尺寸: ', feat1_bot[2].shape)

        # 将图片传入stage1的decoder
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)
        # print('经过stage1的decoder后的尺寸: ', res1_top[0].shape)


        # 经过SAM模块
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        # print('经过stage1SAM模块后的图像尺寸：', stage1_img_bot.shape)
        # print('经过stage1SAM模块后的特征图尺寸：', x2bot_samfeats.shape)

        # 输出stage1的图片
        stage1_img = tf.concat([stage1_img_top, stage1_img_bot], 1)
        # print('stage1_img.shape', stage1_img.shape)
        # print('-'*50, 'stage1完成！！！', '-'*50)


        # stage2
        # 计算shallow_features
        x2top = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)
        # print('x2top.shape', x2top.shape)
        # print('x2top_samfeats.shape', x2top_samfeats.shape)

        # 将stage1的SAM和stage2的shallow_features进行拼接(通道数进行拼接)
        x2top_cat = self.concat12(tf.concat([x2top, x2top_samfeats], -1))
        # print('x2top_cat.shape', x2top_cat.shape)
        x2bot_cat = self.concat12(tf.concat([x2bot, x2bot_samfeats], -1))

        # 将拼接好的features经过stage2的encoder
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)
        # print('feat2_bot.shape', feat2_bot[0].shape)

        # 拼成一张feature
        feat2 = [tf.concat([k, v], 1) for k, v in zip(feat2_top, feat2_bot)]

        # print('feat2.shape', feat2[0].shape)

        # 将拼接好的feature经过stage2的decoder
        res2 = self.stage2_decoder(feat2)
        # print('-' * 50, 'stage2完成！！！', '-' * 50)


        # 经过SAM模块
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)

        # stage3
        x3 = self.shallow_feat3(x3_img)

        # 将stage2的SAM和stage3的shallow_features进行拼接(通道数进行拼接)
        x3_cat = self.concat23(tf.concat([x3, x3_samfeats], -1))

        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail(x3_cat)
        # print('-' * 50, 'stage3完成！！！', '-' * 50)

        # 缩放图像数值范围
        stage3_img = tf.nn.tanh(stage3_img + x3_img)
        stage2_img = tf.nn.tanh(stage2_img)
        stage1_img = tf.nn.tanh(stage1_img)

        return [stage3_img, stage2_img, stage1_img]


