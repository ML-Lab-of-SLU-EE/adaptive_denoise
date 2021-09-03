import imlib as im
import pylib as py
import tf2lib as tl
import tqdm
import data
import utils
import networks as net
import numpy as np
# from UNet_data import *
from tensorflow.python.data.experimental import AUTOTUNE, prefetch_to_device
from losses import *



# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='haze_transform')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=286)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=10)
py.arg('--epoch_decay', type=int, default=5)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])

# model
py.arg('--generator_model', default='MPRNet_1stage', choices=['ResNet', 'UNet', 'MPRNet_1stage', 'MPRNet_2stages'])
py.arg('--attention', default=None, choices=['CBAM', 'CAB', 'BAM', None])

# tricks
py.arg('--dark_channel_loss', type=bool, default=False)
py.arg('--perceptual_loss', type=bool, default=False)
py.arg('--total_variation_loss', type=bool, default=False)
py.arg('--ssim_loss', type=bool, default=False)

# weights
py.arg('--supervised_loss_weight', type=float, default=10.0)
py.arg('--darkchannel_loss_weight', type=float, default=0.1)
py.arg('--perceptual_loss_weight', type=float, default=0.1)
py.arg('--total_variation_loss_weight', type=float, default=0.1)
py.arg('--ssim_loss_weight', type=float, default=5.0)

args = py.args()

# output_dir
output_dir = py.join('slice_output', args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

# cycleGAN的数据读取方法
# train dataset
noise_img_paths = sorted(py.glob(py.join(args.datasets_dir, args.dataset, 'train', 'noise'), '*.jpg'))
haze_img_paths = sorted(py.glob(py.join(args.datasets_dir, args.dataset, 'train', 'haze'), '*.jpg'))

print(noise_img_paths[:5])
print(haze_img_paths[:5])

single_dataset, len_dataset = data.make_zip_dataset(noise_img_paths, haze_img_paths, args.batch_size, args.load_size, args.crop_size, training=False, repeat=False, shuffle=False)

# test dataset
noise_img_paths_test = sorted(py.glob(py.join(args.datasets_dir, args.dataset, 'test', 'noise'), '*.jpg'))
haze_img_paths_test = sorted(py.glob(py.join(args.datasets_dir, args.dataset, 'test', 'haze'), '*.jpg'))
single_dataset_test, _ = data.make_zip_dataset(noise_img_paths, haze_img_paths, args.batch_size, args.load_size, args.crop_size, training=False, repeat=True, shuffle=False)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================
if args.generator_model == 'UNet':
    generator = net.UNet_Generator_single_image(name='UNet_Generator')
elif args.generator_model == 'ResNet':
    generator = net.new_ResNet_Generator_single_image(name='MPRNet_Generator')
elif args.generator_model == 'MPRNet_1stage':
    generator = net.MPRNet_UNet_single_image_1stage(name='MPRNet_1stage_Generator')
elif args.generator_model == 'MPRNet_2stages':
    generator = net.MPRNet_UNet_single_image_2stages(name='MPRNet_2stages_Generator')


x = np.ones([1, 256, 256, 3])
_ = generator(x)

generator.summary()
print('Generator model is:', args.generator_model)

# initiate loss functions
l1_loss_fn = tf.losses.MeanAbsoluteError()
l2_loss_fn = tf.losses.MeanSquaredError()
darkchannel_loss_fn = DarkChannelError()
perceptual_loss_fn = PerceptualError()
ssim_loss_fn = SSIMError()
tv_loss_fn = L1_TVError()

G_lr_scheduler = utils.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(noise, gt_image):
    with tf.GradientTape() as t:
        sliced = generator(noise, training=True)

        # sepurvised loss
        l1_loss = l1_loss_fn(sliced, gt_image)
        # l2_loss = l2_loss_fn(sliced, gt_image)

        supervised_loss = l1_loss

        # darkchannel loss only for haze image
        if args.dark_channel_loss:
            dc_loss = darkchannel_loss_fn(sliced, gt_image)
        else:
            dc_loss = 0

        # ssim loss
        if args.ssim_loss:
            ssim_loss = ssim_loss_fn(sliced, gt_image)
        else:
            ssim_loss = 0

        ssim_loss = ssim_loss

        # tv_loss
        if args.total_variation_loss:
            tv_loss = tv_loss_fn(sliced)
        else:
            tv_loss = 0

        tv_loss = tv_loss

        # perceptual_loss
        if args.perceptual_loss:
            perceptual_loss = perceptual_loss_fn(noise, sliced, gt_image)

        else:
            perceptual_loss = 0

        perceptual_loss = perceptual_loss

        G_loss = dc_loss * args.darkchannel_loss_weight + supervised_loss * args.supervised_loss_weight + ssim_loss * args.ssim_loss_weight + tv_loss * args.total_variation_loss_weight + perceptual_loss * args.perceptual_loss_weight



    G_grad = t.gradient(G_loss, generator.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, generator.trainable_variables))

    return sliced, {'supervised_loss': supervised_loss,
                    'ssim_loss': ssim_loss,
                    'tv_loss': tv_loss,
                    'dc_loss': dc_loss}

@tf.function
def sample(noise):
    sliced = generator(noise, training=False)
    # sliced = tf.nn.tanh(sliced)
    return sliced


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(generator=generator, ep_cnt=ep_cnt), py.join(output_dir, 'checkpoints'), max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)


# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(single_dataset_test)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for noise, gt_image in tqdm.tqdm(single_dataset, desc='Inner Epoch Loop', total=len_dataset):
            # G_loss_dict, D_loss_dict = train_step(noise, real_haze)
            sliced, G_loss_dict = train_G(noise, gt_image)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % 100 == 0:
                # create psnr and ssim files
                psnr_save_file = py.join(output_dir, 'psnr.txt')
                ssim_save_file = py.join(output_dir, 'ssim.txt')
                test_noise, test_gt_image = next(test_iter)
                test_sliced = sample(test_noise)

                psnr_value = compute_batch_psnr(test_sliced, test_gt_image)
                ssim_value = compute_batch_ssim(test_sliced, test_gt_image)
                with open(psnr_save_file, 'a') as psnr_file:
                    psnr_file.write(str(psnr_value.numpy()) + ',')

                with open(ssim_save_file, 'a') as ssim_file:
                    ssim_file.write(str(ssim_value.numpy()) + ',')

                canvas = tf.ones_like(sliced)
                img = im.immerge(np.concatenate([test_noise, test_sliced, test_gt_image], axis=0), n_rows=1)
                im.imwrite(img, py.join(sample_dir, 'iter-%05d.jpg' % G_optimizer.iterations.numpy()))

        # save checkpoint
        checkpoint.save(ep)


