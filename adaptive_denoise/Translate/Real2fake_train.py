import sys
sys.path.append('..')
import functools
import imlib as im
import numpy as np
import pylib as py
import tf2lib as tl
import tf2gan as gan
import tqdm
import utils
import data
import networks as net
from losses import *




# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='haze_transform')
py.arg('--datasets_dir', default='../datasets')
py.arg('--load_size', type=int, default=286)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])

# model
py.arg('--generator_model', default='ResNet_TFL', choices=['ResNet', 'UNet', 'MPRNet', 'ResNet_TFL'])
py.arg('--attention', default=None, choices=['CBAM', 'CAB', 'BAM', None])
py.arg('--use_mask', type=bool, default=False, choices=[True, False])

# tricks
py.arg('--darkchannel_loss', type=bool, default=False)
py.arg('--perceptual_loss', type=bool, default=False)
py.arg('--total_variation_loss', type=bool, default=False)
py.arg('--ssim_loss', type=bool, default=False)

# weights
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--adv_loss_weight', type=float, default=1.0)  # add later
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--dark_channel_loss_weight', type=float, default=1.0)  # add later
py.arg('--perceptual_loss_weight', type=float, default=1.0)  # add later
py.arg('--darkchannel_loss_weight', type=float, default=1.0)

args = py.args()

# output_dir
output_dir = py.join('../translate_output', args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainA'), '*.jpg')
B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainB'), '*.jpg')
A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, args.batch_size, args.load_size, args.crop_size, training=True, repeat=False)

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')
A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size, args.crop_size, training=False, repeat=True)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================
# initiate networks

if args.generator_model == 'ResNet':
    G_A2B = net.new_ResNet_Generator_single_image(name='ResNet_Generator_A', use_mask=args.use_mask, attention=args.attention)
    G_B2A = net.new_ResNet_Generator_single_image(name='ResNet_Generator_B', use_mask=args.use_mask, attention=args.attention)
elif args.generator_model == 'ResNet_TFL':
    G_A2B = net.new_ResNet_Generator_single_image_with_tfl(name='ResNet_Generator_A', attention=args.attention)
    G_B2A = net.new_ResNet_Generator_single_image_with_tfl(name='ResNet_Generator_B', attention=args.attention)
elif args.generator_model == 'UNet':
    G_A2B = net.UNet_Generator_single_image(name='UNet_Generator_A')
    G_B2A = net.UNet_Generator_3_images(name='UNet_Generator_B')
elif args.generator_model == 'MPRNet':
    G_A2B = net.MPRNet(name='MPRNet_Generator_A')
    G_B2A = net.MPRNet(name='MPRNet_Generator_B')

D_A = net.new_Conv_Discriminator(name='Conv_Discriminator_A')
D_B = net.new_Conv_Discriminator(name='Conv_Discriminator_B')


x = tf.random.normal([1, 256, 256, 3])
_ = G_A2B(x)
_ = G_B2A(x)
G_A2B.summary()


d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()
darkchannel_loss_fn = DarkChannelError()
perceptual_loss_fn = PerceptualError()


G_lr_scheduler = utils.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = utils.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B):
    with tf.GradientTape() as t:
        A2B = G_A2B(A, training=True)
        B2A = G_B2A(B, training=True)
        A2B2A = G_B2A(A2B, training=True)
        B2A2B = G_A2B(B2A, training=True)
        A2A = G_B2A(A, training=True)
        B2B = G_A2B(B, training=True)

        A2B_d_logits = D_B(A2B, training=True)  # A2B_d_logits.shape=[b, 32, 32, 1]
        B2A_d_logits = D_A(B2A, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)

        # compute darkchannel loss
        if args.darkchannel_loss:
            A2B_dc_loss = darkchannel_loss_fn(A, A2B)
            B2A_dc_loss = darkchannel_loss_fn(B, B2A)
        else:
            A2B_dc_loss = B2A_dc_loss = 0

        # compute perceptual loss
        if args.perceptual_loss:
            A2B_perceptual_loss = perceptual_loss_fn(A, A2B, B)
            B2A_perceptual_loss = perceptual_loss_fn(B, B2A, A)
        else:
            A2B_perceptual_loss = B2A_perceptual_loss = 0

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight + (A2B_perceptual_loss + B2A_perceptual_loss) * args.perceptual_loss_weight + (A2B_dc_loss + B2A_dc_loss) * args.darkchannel_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)  # TODO：同时更新两个model的参数使用+，而不是列表
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss,
                      'A2B_dc_loss': A2B_dc_loss,
                      'B2A_dc_loss': B2A_dc_loss,
                      'A2B_perceptual_loss': A2B_perceptual_loss,
                      'A2B_perceptual_loss': A2B_perceptual_loss}


@tf.function
def train_D(A, B, A2B, B2A):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        B2A_d_logits = D_A(B2A, training=True)
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=args.gradient_penalty_mode)
        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp}


def train_step(A, B):
    A2B, B2A, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B, A2B, B2A)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B):
    A2B = G_A2B(A, training=False)
    B2A = G_B2A(B, training=False)
    A2B2A = G_B2A(A2B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return A2B, B2A, A2B2A, B2A2B


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_B2A=G_B2A,
                                D_A=D_A,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(A_B_dataset_test)
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
        for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
            G_loss_dict, D_loss_dict = train_step(A, B)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % 100 == 0:
                psnr_save_file = py.join(output_dir, 'psnr.txt')
                ssim_save_file = py.join(output_dir, 'ssim.txt')

                A, B = next(test_iter)
                A2B, B2A, A2B2A, B2A2B = sample(A, B)
                psnr_value = compute_batch_psnr(A, A2B)
                ssim_value = compute_batch_ssim(A, A2B)
                with open(psnr_save_file, 'a') as psnr_file:
                    psnr_file.write(str(psnr_value.numpy()) + ',')

                with open(ssim_save_file, 'a') as ssim_file:
                    ssim_file.write(str(ssim_value.numpy()) + ',')
                img = im.immerge(np.concatenate([A, A2B, A2B2A, B, B2A, B2A2B], axis=0), n_rows=2)
                im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))


        # save checkpoint
        checkpoint.save(ep)
