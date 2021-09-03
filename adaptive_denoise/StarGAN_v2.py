"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from utils import *
import time
from tensorflow.python.data.experimental import AUTOTUNE, prefetch_to_device
from ops import *

from glob import glob
from tqdm import tqdm
from tqdm.contrib import tenumerate
from copy import deepcopy
import PIL.Image
import networks as net




class StarGAN_v2():
    def __init__(self, args):
        super(StarGAN_v2, self).__init__()

        self.model_name = 'StarGAN_v2'
        self.phase = args.phase

        # checkpoint
        self.slice_unet_checkpoint_dir = args.unet_checkpoint_dir
        self.haze_cycle_checkpoint_dir = args.haze_cycle_checkpoint_dir
        self.rain_cycle_checkpoint_dir = args.haze_cycle_checkpoint_dir
        self.dark_cycle_checkpoint_dir = args.haze_cycle_checkpoint_dir
        self.star_checkpoint_dir = args.star_checkpoint_dir
        self.unet_checkpoint_dir = args.unet_checkpoint_dir

        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.ds_iter = args.ds_iter
        self.iteration = args.iteration

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.lr = args.lr
        self.f_lr = args.f_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.ema_decay = args.ema_decay

        """ Weight """
        self.adv_weight = args.adv_weight
        self.sty_weight = args.sty_weight
        self.ds_weight = args.ds_weight
        self.cyc_weight = args.cyc_weight
        self.supervised_weight = args.supervised_weight

        self.r1_weight = args.r1_weight

        """ Generator """
        self.latent_dim = args.latent_dim
        self.style_dim = args.style_dim
        self.num_style = args.num_style

        """ Mapping Network """
        self.hidden_dim = args.hidden_dim

        """ Discriminator """
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.checkpoint_dir = args.star_checkpoint_dir
        check_folder(self.checkpoint_dir)

        self.log_dir = os.path.join(args.log_dir, self.model_dir)
        check_folder(self.log_dir)

        self.result_dir = os.path.join(args.result_dir, self.model_dir)
        check_folder(self.result_dir)

        '''CycleGAN parameters'''


        dataset_path = './datasets'

        self.dataset_path = os.path.join(dataset_path, self.dataset_name, 'train')
        self.test_dataset_path = os.path.join(dataset_path, self.dataset_name, 'test')
        self.domain_list = sorted([os.path.basename(x) for x in glob(self.dataset_path + '/*')])  # domain_list = ['clean', 'dark', 'haze', 'rain']
        self.num_domains = len(self.domain_list)

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# domain_list : ", self.domain_list)

        print("# batch_size : ", self.batch_size)
        print("# max iteration : ", self.iteration)
        print("# ds iteration : ", self.ds_iter)

        print()

        print("##### Generator #####")
        print("# latent_dim : ", self.latent_dim)
        print("# style_dim : ", self.style_dim)
        print("# num_style : ", self.num_style)

        print()

        print("##### Mapping Network #####")
        print("# hidden_dim : ", self.hidden_dim)

        print()

        print("##### Discriminator #####")
        print("# spectral normalization : ", self.sn)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        if self.phase == 'train':
            """ Input Image"""

            # TODO:[noise, fake_haze, haze_domain, fake_rain, rain_domain, fake_dark, dark_domain, clean, clean_domain]
            img_class = new_Image_data(self.img_size, self.img_ch, self.dataset_path, self.domain_list, self.augment_flag)
            img_class.preprocess_multi()  # get 9 lists

            dataset_num = len(img_class.noise)
            print("Dataset number : ", dataset_num)

            img_and_domain = tf.data.Dataset.from_tensor_slices((img_class.noise,
                                                                 img_class.fake_haze, img_class.haze_domain,
                                                                 img_class.fake_rain, img_class.rain_domain,
                                                                 img_class.fake_dark, img_class.dark_domain,
                                                                 img_class.clean, img_class.clean_domain))  # construct a dataset

            gpu_device = '/gpu:0'

            img_and_domain = img_and_domain.shuffle(buffer_size=dataset_num, reshuffle_each_iteration=True).repeat()
            img_and_domain = img_and_domain.map(map_func=img_class.image_processing_multi, num_parallel_calls=AUTOTUNE).batch(self.batch_size, drop_remainder=True)
            img_and_domain = img_and_domain.apply(prefetch_to_device(gpu_device, buffer_size=AUTOTUNE))

            self.img_and_domain_iter = iter(img_and_domain)

            """ Network """
            # UNet model
            self.slice_generator = net.Cascade_MPRNet_UNet_3stages(name='Slice_generator')
            # self.slice_discriminator = new_Conv_Discriminator(name='Slice_discriminator')  # TODO: do not load UNet discriminator

            # moving_average unet model
            self.slice_generator_ema = deepcopy(self.slice_generator)

            # cyclegan model, 3 models
            self.haze_generator = net.new_ResNet_Generator_single_image(name='Haze_generator', use_mask=True, attention=None)
            self.rain_generator = net.new_ResNet_Generator_single_image(name='Rain_generator', use_mask=True, attention=None)
            self.dark_generator = net.new_ResNet_Generator_single_image(name='Dark_generator', use_mask=True, attention=None)
            self.haze_discriminator = net.new_Conv_Discriminator(name='Haze_discriminator')
            self.rain_discriminator = net.new_Conv_Discriminator(name='Rain_discriminator')
            self.dark_discriminator = net.new_Conv_Discriminator(name='Dark_discriminator')

            # moving_average cycle model
            self.haze_generator_ema = deepcopy(self.haze_generator)
            self.rain_generator_ema = deepcopy(self.rain_generator)
            self.dark_generator_ema = deepcopy(self.dark_generator)

            # stargan model
            self.generator = net.Generator(self.img_size, self.img_ch, self.style_dim, max_conv_dim=self.hidden_dim, sn=False, name='Generator')
            self.mapping_network = net.MappingNetwork(self.style_dim, self.hidden_dim, self.num_domains, sn=False, name='MappingNetwork')
            self.style_encoder = net.StyleEncoder(self.img_size, self.style_dim, self.num_domains, max_conv_dim=self.hidden_dim, sn=False, name='StyleEncoder')
            self.discriminator = net.Discriminator(self.img_size, self.num_domains, max_conv_dim=self.hidden_dim, sn=self.sn, name='Discriminator')

            # moving_average stargan model
            self.generator_ema = deepcopy(self.generator)
            self.mapping_network_ema = deepcopy(self.mapping_network)
            self.style_encoder_ema = deepcopy(self.style_encoder)

            """ Finalize model (build) """
            x = np.ones(shape=[self.batch_size, self.img_size, self.img_size, self.img_ch], dtype=np.float32)
            y = np.ones(shape=[self.batch_size, 1], dtype=np.int32)
            z = np.ones(shape=[self.batch_size, self.latent_dim], dtype=np.float32)
            s = np.ones(shape=[self.batch_size, self.style_dim], dtype=np.float32)

            # finalize UNet model
            _, _, _ = self.slice_generator(x)
            _, _, _ = self.slice_generator_ema(x)

            # finalize cyclegan model
            _ = self.haze_generator(x)
            _ = self.haze_generator_ema(x)
            _ = self.rain_generator(x)
            _ = self.rain_generator_ema(x)
            _ = self.dark_generator(x)
            _ = self.dark_generator_ema(x)
            _ = self.haze_discriminator(x)
            _ = self.rain_discriminator(x)
            _ = self.dark_discriminator(x)

            # finalize stargan model
            _ = self.mapping_network([z, y])
            _ = self.mapping_network_ema([z, y])
            _ = self.style_encoder([x, y])
            _ = self.style_encoder_ema([x, y])
            _ = self.generator([x, s])
            _ = self.generator_ema([x, s])
            _ = self.discriminator([x, y])


            """ Optimizer """
            # optimizer for cyclegan and unet
            self.gs_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)
            self.ds_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)

            # optimizer for stargan
            self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)
            self.e_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)
            self.f_optimizer = tf.keras.optimizers.Adam(learning_rate=self.f_lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)
            self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)


            """ Checkpoint """
            # TODO:UNet and CycleGAN did not load optimizer
            # unet checkpoint
            self.unet_ckpt = tf.train.Checkpoint(generator=self.slice_generator, generator_ema=self.slice_generator_ema)  # load generator and 3 discriminators
            self.unet_manager = tf.train.CheckpointManager(self.unet_ckpt, self.unet_checkpoint_dir, max_to_keep=1)

            if self.unet_manager.latest_checkpoint:
                self.unet_ckpt.restore(self.unet_manager.latest_checkpoint).expect_partial()
                print('Latest Slice UNet checkpoint restored !')
            else:
                print('Not restoring Slice UNet model from saved checkpoint')

            # cyclegan checkpoint, a function to load checkpoint(in cyclegan_network.py)
            read_cyclegan_ckpt_file(self.haze_generator, self.haze_generator_ema,
                                    self.haze_discriminator, self.haze_cycle_checkpoint_dir, name='Haze CycleGAN', phase='train')
            read_cyclegan_ckpt_file(self.rain_generator, self.rain_generator_ema,
                                    self.rain_discriminator, self.haze_cycle_checkpoint_dir, name='Rain CycleGAN', phase='train')
            read_cyclegan_ckpt_file(self.dark_generator, self.dark_generator_ema,
                                    self.dark_discriminator, self.haze_cycle_checkpoint_dir, name='Dark CycleGAN', phase='train')

            # stargan checkpoint
            self.ckpt = tf.train.Checkpoint(generator=self.generator, generator_ema=self.generator_ema,
                                            mapping_network=self.mapping_network, mapping_network_ema=self.mapping_network_ema,
                                            style_encoder=self.style_encoder, style_encoder_ema=self.style_encoder_ema,
                                            discriminator=self.discriminator,
                                            g_optimizer=self.g_optimizer, e_optimizer=self.e_optimizer, f_optimizer=self.f_optimizer,
                                            d_optimizer=self.d_optimizer)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=1)
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
                print('Latest Denoise StarGAN checkpoint restoredl !')
                print('start iteration : ', self.start_iteration)
            else:
                print('Not restoring Denoise StarGAN model from saved checkpoint !')



        else:
            """ Test """
            """ Network """
            # UNet network
            self.slice_generator_ema = net.Cascade_MPRNet_UNet_3stages(name='Slice_generator')

            # CycleGAN networks
            self.haze_generator_ema = net.new_ResNet_Generator(name='Haze_generator', use_mask=True, attention=None)
            self.rain_generator_ema = net.new_ResNet_Generator(name='Rain_generator', use_mask=True, attention=None)
            self.dark_generator_ema = net.new_ResNet_Generator(name='Dark_generator', use_mask=True, attention=None)

            # StarGAN networks
            self.generator_ema = net.Generator(self.img_size, self.img_ch, self.style_dim, max_conv_dim=self.hidden_dim, sn=False, name='Generator')
            self.mapping_network_ema = net.MappingNetwork(self.style_dim, self.hidden_dim, self.num_domains, sn=False, name='MappingNetwork')
            self.style_encoder_ema = net.StyleEncoder(self.img_size, self.style_dim, self.num_domains, max_conv_dim=self.hidden_dim, sn=False, name='StyleEncoder')

            """ Finalize model (build) """
            x = np.ones(shape=[self.batch_size, self.img_size, self.img_size, self.img_ch], dtype=np.float32)
            y = np.ones(shape=[self.batch_size, 1], dtype=np.int32)
            z = np.ones(shape=[self.batch_size, self.latent_dim], dtype=np.float32)
            s = np.ones(shape=[self.batch_size, self.style_dim], dtype=np.float32)

            _ = self.slice_generator_ema(x)
            _ = self.haze_generator_ema(x)
            _ = self.rain_generator_ema(x)
            _ = self.dark_generator_ema(x)
            _ = self.mapping_network_ema([z, y])
            _ = self.style_encoder_ema([x, y])
            _ = self.generator_ema([x, s])

            """ Checkpoint """
            # UNet checkpoint
            self.unet_ckpt = tf.train.Checkpoint(generator_ema=self.slice_generator_ema)
            self.unet_manager = tf.train.CheckpointManager(self.unet_ckpt, self.unet_checkpoint_dir)
            if self.unetr_manager.latest_checkpoint:
                self.unet_ckpt.restore(self.unet_manager.latest_checkpoint).expect_partial()
                print('Latest Slice UNet checkpoint restored!!')
            else:
                print('Not restoring Slice UNet from saved checkpoint')

            # CycleGAn checkpoint
            read_cyclegan_ckpt_file(self.haze_generator, self.haze_generator_ema,
                                    self.haze_discriminator, self.haze_cycle_checkpoint_dir, name='Haze CycleGAN',
                                    phase='test')
            read_cyclegan_ckpt_file(self.rain_generator, self.rain_generator_ema,
                                    self.rain_discriminator, self.haze_cycle_checkpoint_dir, name='Rain CycleGAN',
                                    phase='test')
            read_cyclegan_ckpt_file(self.dark_generator, self.dark_generator_ema,
                                    self.dark_discriminator, self.haze_cycle_checkpoint_dir, name='Dark CycleGAN',
                                    phase='test')

            # StarGAN checkpoint
            self.star_ckpt = tf.train.Checkpoint(generator_ema=self.generator_ema,
                                            mapping_network_ema=self.mapping_network_ema,
                                            style_encoder_ema=self.style_encoder_ema)
            self.star_manager = tf.train.CheckpointManager(self.star_ckpt, self.checkpoint_dir, max_to_keep=1)

            if self.star_manager.latest_checkpoint:
                self.star_ckpt.restore(self.star_manager.latest_checkpoint).expect_partial()
                print('Latest Denoise StarGAN checkpoint restored!!')
            else:
                print('Not restoring Denoise StarGAN from saved checkpoint')

    @tf.function
    def g_train_step(self, noise, fake_haze_gt, y_org_haze, fake_rain_gt, y_org_rain, fake_dark_gt, y_org_dark, y_trg, z_trg=None, x_ref=None):  # x_real=noise, y_org=?, x_ref=clean, y_trg=clean, # TODO: y_org is changed by y_org_haze
        with tf.GradientTape(persistent=True) as g_tape:
            if z_trg is not None:
                z_trg = z_trg
            if x_ref is not None:
                x_ref = x_ref

            # adversarial loss
            if z_trg is not None:
                s_trg = self.mapping_network([z_trg, y_trg])
            else:
                s_trg = self.style_encoder([x_ref, y_trg])

            '''UNet'''
            haze, rain, dark = self.slice_generator(noise)
            # TODO:add UNet discriminators to add a adv_g_loss

            '''CycleGAN'''
            fake_haze = self.haze_generator(haze)
            fake_rain = self.rain_generator(rain)
            fake_dark = self.dark_generator(dark)

            # pass cyclegan discriminators
            fake_haze_logit = self.haze_discriminator(fake_haze)
            fake_rain_logit = self.rain_discriminator(fake_rain)
            fake_dark_logit = self.dark_discriminator(fake_dark)

            dehazed = self.generator([fake_haze, s_trg])
            derained = self.generator([fake_rain, s_trg])
            enlighted = self.generator([fake_dark, s_trg])

            # pass stargan discriminators
            dehazed_fake_logit = self.discriminator([dehazed, y_trg])
            derained_fake_logit = self.discriminator([derained, y_trg])
            enlighted_fake_logit = self.discriminator([enlighted, y_trg])

            # compute cyclegan g_adv_loss
            g_haze_cyclegan_adv_loss = self.adv_weight * generator_loss(self.gan_type, fake_haze_logit)
            g_rain_cyclegan_adv_loss = self.adv_weight * generator_loss(self.gan_type, fake_rain_logit)
            g_dark_cyclegan_adv_loss = self.adv_weight * generator_loss(self.gan_type, fake_dark_logit)
            # TODO: if it can be add together
            g_cyclegan_adv = g_haze_cyclegan_adv_loss + g_rain_cyclegan_adv_loss + g_dark_cyclegan_adv_loss

            # compute stargan g_adv_loss
            g_haze_stargan_adv_loss = self.adv_weight * generator_loss(self.gan_type, dehazed_fake_logit)
            g_rain_stargan_adv_loss = self.adv_weight * generator_loss(self.gan_type, derained_fake_logit)
            g_dark_stargan_adv_loss = self.adv_weight * generator_loss(self.gan_type, enlighted_fake_logit)
            g_stargan_adv = g_haze_stargan_adv_loss + g_rain_stargan_adv_loss + g_dark_stargan_adv_loss

            # style reconstruction loss
            haze_s_pred = self.style_encoder([dehazed, y_trg])
            rain_s_pred = self.style_encoder([derained, y_trg])
            dark_s_pred = self.style_encoder([enlighted, y_trg])
            g_sty_loss = self.sty_weight * (L1_loss(haze_s_pred, s_trg) + L1_loss(rain_s_pred, s_trg) + L1_loss(dark_s_pred, s_trg))

            # TODO: delete diversity sensitive loss
            # cycle-consistency loss
            s_org_haze = self.style_encoder([fake_haze, y_org_haze])
            x_rec_haze = self.generator([dehazed, s_org_haze])
            g_cyc_loss_haze = self.cyc_weight * L1_loss(x_rec_haze, fake_haze)

            s_org_rain = self.style_encoder([fake_rain, y_org_rain])
            x_rec_rain = self.generator([derained, s_org_rain])
            g_cyc_loss_rain = self.cyc_weight * L1_loss(x_rec_rain, fake_rain)

            s_org_dark = self.style_encoder([fake_dark, y_org_dark])
            x_rec_dark = self.generator([enlighted, s_org_dark])
            g_cyc_loss_dark = self.cyc_weight * L1_loss(x_rec_dark, fake_dark)
            g_cyc_loss = g_cyc_loss_haze + g_cyc_loss_rain + g_cyc_loss_dark

            regular_loss = regularization_loss(self.generator)

            # supervised loss
            cyclegan_supervised_loss = self.supervised_weight * (L2_loss(fake_haze, fake_haze_gt) + L2_loss(fake_rain, fake_rain_gt) + L2_loss(fake_dark, fake_dark_gt))
            stargan_supervised_loss = self.supervised_weight * (L2_loss(dehazed, x_ref) + L2_loss(derained, x_ref) + L2_loss(enlighted, x_ref))

            g_loss = g_cyclegan_adv + g_stargan_adv + g_sty_loss + g_cyc_loss + regular_loss + stargan_supervised_loss + cyclegan_supervised_loss  # TODO update 3 cyclegans the same time

        g_train_variable = self.generator.trainable_variables + self.haze_generator.trainable_variables + self.rain_generator.trainable_variables + self.dark_generator.trainable_variables

        g_gradient = g_tape.gradient(g_loss, g_train_variable)
        self.g_optimizer.apply_gradients(zip(g_gradient, g_train_variable))

        if z_trg is not None:
            f_train_variable = self.mapping_network.trainable_variables
            e_train_variable = self.style_encoder.trainable_variables

            f_gradient = g_tape.gradient(g_loss, f_train_variable)
            e_gradient = g_tape.gradient(g_loss, e_train_variable)

            self.f_optimizer.apply_gradients(zip(f_gradient, f_train_variable))
            self.e_optimizer.apply_gradients(zip(e_gradient, e_train_variable))

        return g_cyclegan_adv, g_stargan_adv, g_sty_loss, g_cyc_loss, cyclegan_supervised_loss, stargan_supervised_loss, g_loss

    @tf.function
    def d_train_step(self, noise, fake_haze_gt, y_org_haze, fake_rain_gt, y_org_rain, fake_dark_gt, y_org_dark, y_trg, z_trg=None, x_ref=None):
        with tf.GradientTape() as d_tape:

            if z_trg is not None:
                s_trg = self.mapping_network([z_trg, y_trg])
            else:  # x_ref is not None
                s_trg = self.style_encoder([x_ref, y_trg])

            '''UNet'''
            haze, rain, dark = self.slice_generator(noise)

            '''CycleGAN'''
            fake_haze = self.haze_generator(haze)
            fake_rain = self.rain_generator(rain)
            fake_dark = self.dark_generator(dark)

            # cyclegan discriminator
            fake_haze_logit = self.haze_discriminator(fake_haze)
            fake_rain_logit = self.rain_discriminator(fake_rain)
            fake_dark_logit = self.dark_discriminator(fake_dark)

            real_haze_logit = self.haze_discriminator(fake_haze_gt)
            real_rain_logit = self.rain_discriminator(fake_rain_gt)
            real_dark_logit = self.dark_discriminator(fake_dark_gt)
            d_cyclegan_adv = self.adv_weight * (discriminator_loss(self.gan_type, real_haze_logit, fake_haze_logit) + discriminator_loss(self.gan_type, real_rain_logit, fake_rain_logit) + discriminator_loss(self.gan_type, real_dark_logit, fake_dark_logit))

            '''StarGAN'''
            dehazed = self.generator([fake_haze, s_trg])
            derained = self.generator([fake_rain, s_trg])
            enlighted = self.generator([fake_dark, s_trg])

            # stargan discriminator
            # TODO: real_logits may be a little chaos, it should be use x_ref instead of fake_haze_gt
            real_dehaze_logit = self.discriminator([fake_haze_gt, y_org_haze])
            real_derain_logit = self.discriminator([fake_rain_gt, y_org_rain])
            real_enlight_logit = self.discriminator([fake_dark_gt, y_org_dark])

            fake_dehaze_logit = self.discriminator([dehazed, y_trg])
            fake_derain_logit = self.discriminator([derained, y_trg])
            fake_enlight_logit = self.discriminator([enlighted, y_trg])
            d_stargan_adv = self.adv_weight * (discriminator_loss(self.gan_type, real_dehaze_logit, fake_dehaze_logit)  + discriminator_loss(self.gan_type, real_derain_logit, fake_derain_logit) + discriminator_loss(self.gan_type, real_enlight_logit, fake_enlight_logit))

            d_adv_loss = d_cyclegan_adv + d_stargan_adv

            if self.gan_type == 'gan-gp':
                d_adv_loss += self.r1_weight * (r1_gp_req(self.discriminator, fake_haze_gt, y_org_haze) + r1_gp_req(self.discriminator, fake_rain_gt, y_org_rain) + r1_gp_req(self.discriminator, fake_dark_gt, y_org_dark))

            regular_loss = regularization_loss(self.discriminator)

            d_loss = d_adv_loss + regular_loss

        # star_d_train_variable = self.discriminator.trainable_variables
        # cycle_d_train_variable = self.haze_discriminator.trainable_variables + self.rain_discriminator.trainable_variables + self.dark_discriminator.trainable_variables

        # star_d_gradient = d_tape.gradient(d_loss, star_d_train_variable)
        # cycle_d_gradient = d_tape.gradient(d_loss, cycle_d_train_variable)
        #
        # self.d_optimizer.apply_gradients(zip(star_d_gradient, star_d_train_variable))
        # self.ds_optimizer.apply_gradients(zip(cycle_d_gradient, cycle_d_train_variable))

        d_train_variable = self.discriminator.trainable_variables + self.haze_discriminator.trainable_variables + self.rain_discriminator.trainable_variables + self.dark_discriminator.trainable_variables
        d_gradient = d_tape.gradient(d_loss, d_train_variable)
        self.d_optimizer.apply_gradients(zip(d_gradient, d_train_variable))
        return d_adv_loss, d_loss

    def train(self):

        start_time = time.time()

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        ds_weight_init = self.ds_weight

        for idx in range(self.start_iteration, self.iteration):
            iter_start_time = time.time()

            # decay weight for diversity sensitive loss
            if self.ds_weight > 0:
                self.ds_weight = ds_weight_init - (ds_weight_init / self.ds_iter) * idx

            noise, fake_haze_gt, y_org_haze, fake_rain_gt, y_org_rain, fake_dark_gt, y_org_dark, x_ref, y_trg = next(self.img_and_domain_iter)  # get the input
            z_trg = tf.random.normal(shape=[self.batch_size, self.latent_dim])

            # update discriminator
            d_adv_loss_latent, d_loss_latent = self.d_train_step(noise, fake_haze_gt, y_org_haze, fake_rain_gt, y_org_rain, fake_dark_gt, y_org_dark, y_trg, z_trg=z_trg)
            d_adv_loss_ref, d_loss_ref = self.d_train_step(noise, fake_haze_gt, y_org_haze, fake_rain_gt, y_org_rain, fake_dark_gt, y_org_dark, y_trg, x_ref=x_ref)

            # update generator
            g_cyclegan_adv_latent, g_stargan_adv_latent, g_sty_loss_latent, g_cyc_loss_latent, cyclegan_supervised_loss_latent, stargan_supervised_loss_latent, g_loss_latent = self.g_train_step(noise, fake_haze_gt, y_org_haze, fake_rain_gt, y_org_rain, fake_dark_gt, y_org_dark, y_trg, z_trg=z_trg, x_ref=x_ref)
            g_cyclegan_adv_ref, g_stargan_adv_ref, g_sty_loss_ref, g_cyc_loss_ref, cyclegan_supervised_loss_ref, stargan_supervised_loss_ref, g_loss_ref = self.g_train_step(noise, fake_haze_gt, y_org_haze, fake_rain_gt, y_org_rain, fake_dark_gt, y_org_dark, y_trg, x_ref=x_ref)

            # compute moving average of network parameters
            moving_average(self.generator, self.generator_ema, beta=self.ema_decay)
            moving_average(self.mapping_network, self.mapping_network_ema, beta=self.ema_decay)
            moving_average(self.style_encoder, self.style_encoder_ema, beta=self.ema_decay)
            moving_average(self.haze_generator, self.haze_generator_ema, beta=self.ema_decay)
            moving_average(self.rain_generator, self.rain_generator_ema, beta=self.ema_decay)
            moving_average(self.dark_generator, self.dark_generator_ema, beta=self.ema_decay)


            if idx == 0 :
                # print and conpute paramaters
                haze_generator_params = self.haze_generator.count_params()
                haze_discriminator_params = self.haze_discriminator.count_params()
                rain_generator_params = self.rain_generator.count_params()
                rain_discriminator_params = self.rain_discriminator.count_params()
                dark_generator_params = self.dark_generator.count_params()
                dark_discriminator_params = self.dark_discriminator.count_params()
                print("Haze_generator network parameters : ", format(haze_generator_params, ','))
                print("Haze_discriminator network parameters : ", format(haze_discriminator_params, ','))
                print("Rain_generator network parameters : ", format(rain_generator_params, ','))
                print("Rain_discriminator network parameters : ", format(rain_discriminator_params, ','))
                print("Dark_generator network parameters : ", format(dark_generator_params, ','))
                print("Dark_discriminator network parameters : ", format(dark_discriminator_params, ','))

                g_params = self.generator.count_params()
                d_params = self.discriminator.count_params()
                print("G network parameters : ", format(g_params, ','))
                print("D network parameters : ", format(d_params, ','))

                print("Total network parameters : ", format(haze_generator_params + haze_discriminator_params +
                                                            rain_generator_params + rain_discriminator_params +
                                                            dark_generator_params + dark_discriminator_params +
                                                            g_params + d_params, ','))

            # save to tensorboard

            with train_summary_writer.as_default():
                tf.summary.scalar('g/latent/cyclegan_adv_loss', g_cyclegan_adv_latent, step=idx)
                tf.summary.scalar('g/latent/stargan_adv_loss', g_stargan_adv_latent, step=idx)
                tf.summary.scalar('g/latent/sty_loss', g_sty_loss_latent, step=idx)
                tf.summary.scalar('g/latent/cyc_loss', g_cyc_loss_latent, step=idx)
                tf.summary.scalar('g/latent/cyc_loss', g_cyc_loss_latent, step=idx)
                tf.summary.scalar('g/latent/cyclegan_supervised_loss', cyclegan_supervised_loss_latent, step=idx)
                tf.summary.scalar('g/latent/stargan_supervised_loss', stargan_supervised_loss_latent, step=idx)
                tf.summary.scalar('g/latent/loss', g_loss_latent, step=idx)

                tf.summary.scalar('g/ref/cyclegan_adv_loss', g_cyclegan_adv_ref, step=idx)
                tf.summary.scalar('g/ref/stargan_adv_loss', g_stargan_adv_ref, step=idx)
                tf.summary.scalar('g/ref/sty_loss', g_sty_loss_ref, step=idx)
                tf.summary.scalar('g/ref/cyc_loss', g_cyc_loss_ref, step=idx)
                tf.summary.scalar('g/ref/cyc_loss', g_cyc_loss_ref, step=idx)
                tf.summary.scalar('g/ref/cyclegan_supervised_loss', cyclegan_supervised_loss_ref, step=idx)
                tf.summary.scalar('g/ref/stargan_supervised_loss', stargan_supervised_loss_ref, step=idx)
                tf.summary.scalar('g/ref/loss', g_loss_ref, step=idx)

                tf.summary.scalar('g/ds_weight', self.ds_weight, step=idx)

                tf.summary.scalar('d/latent/adv_loss', d_adv_loss_latent, step=idx)
                tf.summary.scalar('d/latent/loss', d_loss_latent, step=idx)

                tf.summary.scalar('d/ref/adv_loss', d_adv_loss_ref, step=idx)
                tf.summary.scalar('d/ref/loss', d_loss_ref, step=idx)

            # save every self.save_freq
            if np.mod(idx + 1, self.save_freq) == 0:
                self.manager.save(checkpoint_number=idx + 1)

            # save every self.print_freq
            if np.mod(idx + 1, self.print_freq) == 0:


                latent_fake_save_path = './{}/latent_{:07d}.jpg'.format(self.sample_dir, idx + 1)
                ref_fake_save_path = './{}/ref_{:07d}.jpg'.format(self.sample_dir, idx + 1)

                # TODO:change the canvas
                # self.latent_canvas(x_real, latent_fake_save_path)
                # TODO: sample image
                # self.refer_canvas(noise, x_ref, y_trg, ref_fake_save_path, img_num=5)
                self.new_refer_canvas(noise, fake_haze_gt, fake_rain_gt, fake_dark_gt, x_ref, y_trg, ref_fake_save_path)


            print("iter: [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (
            idx, self.iteration, time.time() - iter_start_time, d_loss_latent+d_loss_ref, g_loss_latent+g_loss_ref))

        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

        print("Total train time: %4.4f" % (time.time() - start_time))

    @property
    def model_dir(self):

        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return "{}_{}_{}{}".format(self.model_name, self.dataset_name, self.gan_type, sn)

    def refer_canvas(self, x_real, x_ref, y_trg, path, img_num):  # training: img_num=5
        if type(img_num) == list:
            # In test phase
            src_img_num = img_num[0]
            ref_img_num = img_num[1]
        else:
            src_img_num = min(img_num, self.batch_size)
            ref_img_num = min(img_num, self.batch_size)  # batch_size=2, only display 2 src_img and 2 ref_img

        x_real = x_real[:src_img_num]
        x_ref = x_ref[:ref_img_num]
        y_trg = y_trg[:ref_img_num]

        canvas = PIL.Image.new('RGB', (self.img_size * (src_img_num + 1) + 10, self.img_size * (ref_img_num + 1) + 10),
                               'white')

        x_real_post = postprocess_images(x_real)
        x_ref_post = postprocess_images(x_ref)

        for col, src_image in enumerate(list(x_real_post)):
            canvas.paste(PIL.Image.fromarray(np.uint8(src_image), 'RGB'), ((col + 1) * self.img_size + 10, 0))

        for row, dst_image in enumerate(list(x_ref_post)):
            canvas.paste(PIL.Image.fromarray(np.uint8(dst_image), 'RGB'), (0, (row + 1) * self.img_size + 10))

            row_images = np.stack([dst_image] * src_img_num)
            row_images = preprocess_fit_train_image(row_images)
            row_images_y = np.stack([y_trg[row]] * src_img_num)

            s_trg = self.style_encoder_ema([row_images, row_images_y])
            image = self.generator_ema([x_real, s_trg])
            print('the range of gen image is: ', np.min(image), np.max(image))
            row_fake_images = postprocess_images(image)
            print('the range of fake image is: ', np.min(row_fake_images), np.max(row_fake_images))

            for col, image in enumerate(list(row_fake_images)):
                canvas.paste(PIL.Image.fromarray(np.uint8(image), 'RGB'),
                             ((col + 1) * self.img_size + 10, (row + 1) * self.img_size + 10))

        canvas.save(path)

    def new_refer_canvas(self, noise, fake_haze_gt, fake_rain_gt, fake_dark_gt, clean, y_trg, path):  # training: img_num=5
        # noise = noise[0]  # only sample one image from a batch
        # clean = clean[0]
        # y_trg = y_trg[0]

        canvas = PIL.Image.new('RGB', (self.img_size * 4 + 10, self.img_size * 4 + 10), 'white')  # size of the canvas is [4 * 256 + 10, 4 * 256 + 10]

        noise_post = postprocess_images(noise)  # x_real -> noise; x_ref -> clean
        fake_haze_gt_post = postprocess_images(fake_haze_gt)
        fake_rain_gt_post = postprocess_images(fake_rain_gt)
        fake_dark_gt_post = postprocess_images(fake_dark_gt)
        gt_post = [fake_haze_gt_post, fake_rain_gt_post, fake_dark_gt_post]

        # paste the ground truth images
        canvas.paste(PIL.Image.fromarray(np.uint8(noise_post[0]), 'RGB'), (0, 0))
        for col, gt_image in enumerate(gt_post):
            canvas.paste(PIL.Image.fromarray(np.uint8(gt_image[0]), 'RGB'), ((col + 1) * self.img_size + 10, 0))

        # paste unet results
        noise_in = preprocess_fit_train_image(noise)
        slices = self.slice_generator(noise_in)  # slices=[haze, rain, dark]
        for col, slice in tenumerate(list(slices)):
            slice_out = postprocess_images(slice)
            canvas.paste(PIL.Image.fromarray(np.uint8(slice_out[0]), 'RGB'), ((col + 1) * self.img_size + 10, 1 * self.img_size))

        # paste cyclegan results
        fake_haze = self.haze_generator_ema(slices[0])
        fake_rain = self.rain_generator_ema(slices[1])
        fake_dark = self.dark_generator_ema(slices[2])
        gen_fake_images = [fake_haze, fake_rain, fake_dark]
        for col, gen_fake_image in enumerate(list(gen_fake_images)):
            gen_fake_image = postprocess_images(gen_fake_image)
            canvas.paste(PIL.Image.fromarray(np.uint8(gen_fake_image[0]), 'RGB'), ((col + 1) * self.img_size + 10, 2 * self.img_size))

        # paste stargan results
        clean = preprocess_fit_train_image(clean)
        s_trg = self.style_encoder_ema([clean, y_trg])
        dehazed = self.generator_ema([fake_haze, s_trg])
        derained = self.generator_ema([fake_rain, s_trg])
        enlighted = self.generator_ema([fake_dark, s_trg])
        denoised_images = [dehazed, derained, enlighted]
        for col, denoised_image in enumerate(denoised_images):
            denoised_image = postprocess_images(denoised_image)
            canvas.paste(PIL.Image.fromarray(np.uint8(denoised_image[0]), 'RGB'), ((col + 1) * self.img_size + 10, 3 * self.img_size))

        canvas.save(path)

    def latent_canvas(self, x_real, path):
        canvas = PIL.Image.new('RGB', (self.img_size * (self.num_domains + 1) + 10, self.img_size * self.num_style), 'white')

        x_real = tf.expand_dims(x_real[0], axis=0)
        src_image = postprocess_images(x_real)[0]
        canvas.paste(PIL.Image.fromarray(np.uint8(src_image), 'RGB'), (0, 0))

        domain_fix_list = tf.constant([idx for idx in range(self.num_domains)])

        z_trgs = tf.random.normal(shape=[self.num_style, self.latent_dim])

        for row in range(self.num_style):
            z_trg = tf.expand_dims(z_trgs[row], axis=0)

            for col, y_trg in enumerate(list(domain_fix_list)):
                y_trg = tf.reshape(y_trg, shape=[1, 1])
                s_trg = self.mapping_network_ema([z_trg, y_trg])
                x_fake = self.generator_ema([x_real, s_trg])
                x_fake = postprocess_images(x_fake)

                col_image = x_fake[0]

                canvas.paste(PIL.Image.fromarray(np.uint8(col_image), 'RGB'), ((col + 1) * self.img_size + 10, row * self.img_size))

        canvas.save(path)

    def test(self, merge=True, merge_size=0):
        source_path = os.path.join(self.test_dataset_path, 'src_imgs')
        source_images = glob(os.path.join(source_path, '*.png')) + glob(os.path.join(source_path, '*.jpg'))
        source_images = sorted(source_images)

        # reference-guided synthesis
        print('reference-guided synthesis')
        reference_path = os.path.join(self.test_dataset_path, 'ref_imgs')
        reference_images = []
        reference_domain = []

        for idx, domain in enumerate(self.domain_list):
            image_list = glob(os.path.join(reference_path, domain) + '/*.png') + glob(
                os.path.join(reference_path, domain) + '/*.jpg')
            image_list = sorted(image_list)
            domain_list = [[idx]] * len(image_list)  # [ [0], [0], ... , [0] ]

            reference_images.extend(image_list)
            reference_domain.extend(domain_list)

        if merge:
            src_img = None
            ref_img = None
            ref_img_domain = None

            if merge_size == 0:
                # [len_src_imgs : len_ref_imgs] matching
                for src_idx, src_img_path in tenumerate(source_images):
                    src_name, src_extension = os.path.splitext(src_img_path)
                    src_name = os.path.basename(src_name)

                    src_img_ = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    src_img_ = tf.expand_dims(src_img_, axis=0)



                    if src_idx == 0:
                        src_img = src_img_
                    else:
                        src_img = tf.concat([src_img, src_img_], axis=0)

                for ref_idx, (ref_img_path, ref_img_domain_) in tenumerate(zip(reference_images, reference_domain)):
                    ref_name, ref_extension = os.path.splitext(ref_img_path)
                    ref_name = os.path.basename(ref_name)

                    ref_img_ = load_images(ref_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    ref_img_ = tf.expand_dims(ref_img_, axis=0)
                    ref_img_domain_ = tf.expand_dims(ref_img_domain_, axis=0)

                    if ref_idx == 0:
                        ref_img = ref_img_
                        ref_img_domain = ref_img_domain_
                    else:
                        ref_img = tf.concat([ref_img, ref_img_], axis=0)
                        ref_img_domain = tf.concat([ref_img_domain, ref_img_domain_], axis=0)

                save_path = './{}/ref_all.jpg'.format(self.result_dir)

                self.refer_canvas(src_img, ref_img, ref_img_domain, save_path,
                                  img_num=[len(source_images), len(reference_images)])

            else:
                # [merge_size : merge_size] matching
                src_size = 0
                for src_idx, src_img_path in tenumerate(source_images):
                    src_name, src_extension = os.path.splitext(src_img_path)
                    src_name = os.path.basename(src_name)

                    src_img_ = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    src_img_ = tf.expand_dims(src_img_, axis=0)

                    if src_size < merge_size:
                        if src_idx % merge_size == 0:
                            src_img = src_img_
                        else:
                            src_img = tf.concat([src_img, src_img_], axis=0)
                        src_size += 1

                        if src_size == merge_size:
                            src_size = 0

                            ref_size = 0
                            for ref_idx, (ref_img_path, ref_img_domain_) in enumerate(
                                    zip(reference_images, reference_domain)):
                                ref_name, ref_extension = os.path.splitext(ref_img_path)
                                ref_name = os.path.basename(ref_name)

                                ref_img_ = load_images(ref_img_path, self.img_size,
                                                       self.img_ch)  # [img_size, img_size, img_ch]
                                ref_img_ = tf.expand_dims(ref_img_, axis=0)
                                ref_img_domain_ = tf.expand_dims(ref_img_domain_, axis=0)

                                if ref_size < merge_size:
                                    if ref_idx % merge_size == 0:
                                        ref_img = ref_img_
                                        ref_img_domain = ref_img_domain_
                                    else:
                                        ref_img = tf.concat([ref_img, ref_img_], axis=0)
                                        ref_img_domain = tf.concat([ref_img_domain, ref_img_domain_], axis=0)

                                    ref_size += 1
                                    if ref_size == merge_size:
                                        ref_size = 0

                                        save_path = './{}/ref_{}_{}.jpg'.format(self.result_dir, src_idx + 1, ref_idx + 1)

                                        self.refer_canvas(src_img, ref_img, ref_img_domain, save_path,
                                                          img_num=merge_size)

        else:
            # [1:1] matching
            for src_img_path in tqdm(source_images):
                src_name, src_extension = os.path.splitext(src_img_path)
                src_name = os.path.basename(src_name)

                src_img = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                src_img = tf.expand_dims(src_img, axis=0)

                for ref_img_path, ref_img_domain in zip(reference_images, reference_domain):
                    ref_name, ref_extension = os.path.splitext(ref_img_path)
                    ref_name = os.path.basename(ref_name)

                    ref_img = load_images(ref_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    ref_img = tf.expand_dims(ref_img, axis=0)
                    ref_img_domain = tf.expand_dims(ref_img_domain, axis=0)

                    save_path = './{}/ref_{}_{}{}'.format(self.result_dir, src_name, ref_name, src_extension)

                    self.refer_canvas(src_img, ref_img, ref_img_domain, save_path, img_num=1)

        # latent-guided synthesis
        print('latent-guided synthesis')
        for src_img_path in tqdm(source_images):
            src_name, src_extension = os.path.splitext(src_img_path)
            src_name = os.path.basename(src_name)

            src_img = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
            src_img = tf.expand_dims(src_img, axis=0)

            save_path = './{}/latent_{}{}'.format(self.result_dir, src_name, src_extension)

            self.latent_canvas(src_img, save_path)
