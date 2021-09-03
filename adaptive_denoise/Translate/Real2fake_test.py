import sys
sys.path.append('..')
import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
from tensorflow import image
import networks as net
import data

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
py.arg('--experiment_dir', default='merge_slice')
# py.arg('--generator_model', default='MPRNet_1stage', choices=['ResNet', 'UNet', 'MPRNet_1stage', 'MPRNet_2stages'])
test_args = py.args()

args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg') + py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.png')
B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg') + py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.png')
A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.load_size, args.crop_size, training=False, drop_remainder=False, shuffle=False, repeat=1)
B_dataset_test = data.make_dataset(B_img_paths_test, args.batch_size, args.load_size, args.crop_size, training=False, drop_remainder=False, shuffle=False, repeat=1)

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

# resotre
# tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()
cyc_ckpt_path = py.join(args.experiment_dir, 'checkpoints')
cycle_ckpt = tf.train.Checkpoint(G_A2B=G_A2B, G_B2A=G_B2A)
manager = tf.train.CheckpointManager(cycle_ckpt, cyc_ckpt_path, max_to_keep=5)
if manager.latest_checkpoint:
    cycle_ckpt.restore(manager.latest_checkpoint).expect_partial()
    print('Cyclegan model restored successfully !!!')

@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    return A2B, A2B2A


@tf.function
def sample_B2A(B):
    B2A = G_B2A(B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return B2A, B2A2B


# run
save_dir = py.join(args.experiment_dir, 'samples_testing', 'real2fake')
py.mkdir(save_dir)
i = 0

psnr_A_fB = []
ssim_A_fB = []
psnr_A_ideA = []
ssim_A_ideA = []

for A in A_dataset_test:
    A2B, A2B2A = sample_A2B(A)
    # print('A2B:', A2B)
    for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
        img = np.concatenate([A_i.numpy(), A2B_i.numpy(), A2B2A_i.numpy()], axis=1)
        im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))
        i += 1
    psnr_A_fB.append(image.psnr(A, A2B, max_val=1.0))
    psnr_A_ideA.append(image.psnr(A, A2B2A, max_val=1.0))

    ssim_A_fB.append(image.ssim(A, A2B, max_val=1.0))
    ssim_A_ideA.append(image.ssim(A, A2B2A, max_val=1.0))


psnr_A_fB = tf.reduce_mean(psnr_A_fB[:-1])
psnr_A_ideA = tf.reduce_mean(psnr_A_ideA[:-1])
ssim_A_fB = tf.reduce_mean(ssim_A_fB[:-1])
ssim_A_ideA = tf.reduce_mean(ssim_A_ideA[:-1])

save_dir = py.join(args.experiment_dir, 'samples_testing', 'fake2real')
py.mkdir(save_dir)
i = 0

psnr_B_fA = []
psnr_B_ideB = []
ssim_B_fA = []
ssim_B_ideB = []

for B in B_dataset_test:
    B2A, B2A2B = sample_B2A(B)
    for B_i, B2A_i, B2A2B_i in zip(B, B2A, B2A2B):
        img = np.concatenate([B_i.numpy(), B2A_i.numpy(), B2A2B_i.numpy()], axis=1)
        im.imwrite(img, py.join(save_dir, py.name_ext(B_img_paths_test[i])))
        i += 1
    psnr_B_fA.append(image.psnr(B, B2A, max_val=1.0))
    psnr_B_ideB.append(image.psnr(B, B2A2B, max_val=1.0))
    ssim_B_fA.append(image.ssim(B, B2A, max_val=1.0))
    ssim_B_ideB.append(image.ssim(B, B2A2B, max_val=1.0))

psnr_B_fA = tf.reduce_mean(psnr_B_fA[:-1])
psnr_B_ideB = tf.reduce_mean(psnr_B_ideB[:-1])
ssim_B_fA = tf.reduce_mean(ssim_B_fA[:-1])
ssim_B_ideB = tf.reduce_mean(ssim_B_ideB[:-1])

print('PSNR of A & fake_B is :', psnr_A_fB)
print('SSIM of A & fake_B is :', ssim_A_fB)

print('PSNR of B & fake_A is :', psnr_B_fA)
print('SSIM of B & fake_A is :', ssim_B_fA)
