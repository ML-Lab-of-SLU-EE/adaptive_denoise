import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, applications


def compute_gram(features):
    batch_size, width, height, num_channels = features.shape
    features = tf.reshape(features, [batch_size, -1, num_channels])
    gram = tf.matmul(features, features, transpose_a=True) / tf.cast(num_channels * width * height, dtype=tf.float32)
    return gram

# 定义感知损失，只计算content损失，没有style
def perceptual_loss(image1, image2):
    vgg_in = layers.Input([None, None, 3])
    vgg16 = applications.VGG16(include_top=False, input_tensor=vgg_in)
    for layer in vgg16.layers:
        layer.trainable = False  # 不更新vgg的参数
    vgg_out = vgg16.get_layer(name='block3_conv3').output
    content_model = Model(vgg_in, vgg_out)
    loss = keras.losses.mean_squared_error(content_model(image1), content_model(image2))

    return tf.reduce_mean(loss)

class PerceptualError(Model):
    def __init__(self):
        super(PerceptualError, self).__init__()
        self.layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
        self.basic_models = self.architecture_init()
        self.loss = keras.losses.mean_squared_error

    def architecture_init(self):
        basic_models = []
        vgg_in = layers.Input([None, None, 3])
        vgg16 = applications.VGG16(include_top=False, input_tensor=vgg_in)
        for layer_name in self.layer_names:
            vgg_out = vgg16.get_layer(name=layer_name).output
            model = Model(vgg_in, vgg_out)
            basic_models.append(model)
        # print('the length of basic_models is:', len(basic_models))

        return basic_models

    def call(self, A, A2B, B):
        # 计算style的损失，希望A2B和B在style上尽可能的接近
        style_loss = 0
        for model in self.basic_models:
            gram_B = compute_gram(model(B))
            gram_A2B = compute_gram(model(A2B))
            style_loss += tf.reduce_mean(self.loss(gram_B, gram_A2B))
        style_loss = tf.reduce_mean(style_loss)

        # 计算content的损失，希望A2B和A在content上尽可能相似
        # content_loss = tf.reduce_mean(self.loss(compute_gram(self.basic_models[2](A)), compute_gram(self.basic_models[2](A2B)))) # compute gram
        content_loss = tf.reduce_mean(self.loss(self.basic_models[2](A), self.basic_models[2](A2B)))  # without computing gram
        return style_loss + content_loss


class DarkChannelError(Model):
    def __init__(self, patch_size=35, mode='l1'):
        super(DarkChannelError, self).__init__()
        self.patch_size = patch_size
        self.mode = mode
        self.maxpool = layers.MaxPooling3D(pool_size=(patch_size, patch_size, 3), strides=1, padding='same')

    def call(self, img1, img2):
        dc1 = self.maxpool(tf.expand_dims(img1, axis=0))
        dc2 = self.maxpool(tf.expand_dims(img2, axis=0))
        if self.mode == 'l1':
            loss = tf.reduce_mean(tf.abs(dc1 - dc2))
        elif self.mode == 'l2':
            loss = tf.reduce_mean(tf.square(dc1 - dc2))

        return loss


class SSIMError(Model):
    def __init__(self):
        super(SSIMError, self).__init__()

    def call(self, x1, x2):
        ssim_loss = tf.image.ssim(x1, x2, max_val=1)
        ssim_loss = 1 - tf.reduce_mean(ssim_loss)
        return ssim_loss

class L1_TVError(Model):
    def __init__(self):
        super(L1_TVError, self).__init__()
        self.e = 1e-5 ** 2

    def call(self, x):
        h_tv = tf.abs(x[:, 1:, :, :] - x[:, :-1, :, :])
        h_tv = tf.reduce_mean(tf.sqrt(h_tv ** 2 + self.e))

        w_tv = tf.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        w_tv = tf.reduce_mean(tf.sqrt(w_tv ** 2 + self.e))

        return w_tv + h_tv

def TV_Loss(x):
    tv_loss = tf.reduce_mean(tf.image.total_variation(x))
    return tv_loss


def compute_batch_psnr(img1, img2):
    psnr = tf.image.psnr(img1, img2, max_val=1)
    return tf.reduce_mean(psnr)

def compute_batch_ssim(img1, img2):
    ssim = tf.image.ssim(img1, img2, max_val=1)
    return tf.reduce_mean(ssim)

