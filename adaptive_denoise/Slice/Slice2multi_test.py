import sys
sys.path.append('..')
import imlib as im
import pylib as py
import tf2lib as tl
import data
import networks as net
import numpy as np
from tensorflow.python.data.experimental import AUTOTUNE, prefetch_to_device
from losses import *




# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir', default='merge_slice')
# py.arg('--generator_model', default='MPRNet_1stage', choices=['ResNet', 'UNet', 'MPRNet_1stage', 'MPRNet_2stages'])
test_args = py.args()

args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

# read train dataset
noise_img_paths = sorted(py.glob(py.join(args.datasets_dir, args.dataset, 'train', 'noise'), '*.jpg'))
haze_img_paths = sorted(py.glob(py.join(args.datasets_dir, args.dataset, 'train', 'haze'), '*.jpg'))
rain_img_paths = sorted(py.glob(py.join(args.datasets_dir, args.dataset, 'train', 'rain'), '*.jpg'))
dark_img_paths = sorted(py.glob(py.join(args.datasets_dir, args.dataset, 'train', 'dark'), '*.jpg'))

multi_dataset_test, _ = data.make_zip_multi_dataset(noise_img_paths, haze_img_paths, rain_img_paths, dark_img_paths, args.batch_size, args.load_size, args.crop_size, training=False, repeat=False)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================
if args.generator_model == 'Casecade_UNet':
    generator = net.Cascade_MPRNet_UNet_3stages(name='CaseCade_UNet_Generator')  # self write
elif args.generator_model == 'Original_MPRNet':
    generator = net.MPRNet(name='Original_MPRNet_Generator')
elif args.generator_model == 'UNet_slice':
    generator = net.UNet_Generator_3_images(name='UNet_Generator')

x = np.ones([1, 256, 256, 3])
_ = generator(x)

generator.summary()
print('Generator model is:', args.generator_model)


# load checkpoint
tl.Checkpoint(dict(generator=generator), py.join(args.experiment_dir, 'checkpoints')).restore()


# test step
@tf.function
def sample(noise):
    sliced = generator(noise, training=False)
    return sliced

# run
save_dir = py.join(args.experiment_dir, 'samples_testing')
py.mkdir(save_dir)

psnr = []
ssim = []
i = 1
for noise, gt_haze, gt_rain, gt_dark in multi_dataset_test:
    images = generator(noise)
    canvas = tf.ones_like(gt_haze)
    img = im.immerge(np.concatenate([noise, gt_dark, gt_haze, gt_rain, canvas, images[0], images[1], images[2]], axis=0), n_rows=2)

    im.imwrite(img, py.join(save_dir, 'image-%05d.jpg' % i))
    single_psnr = (tf.image.psnr(images[0], gt_dark, max_val=1.0) + tf.image.psnr(images[1], gt_haze, max_val=1.0) + tf.image.psnr(images[2], gt_rain, max_val=1.0)) / 3
    print('image%d' % i)
    print('single psnr is: ', single_psnr)
    psnr.append(single_psnr)
    single_ssim = (tf.image.ssim(images[0], gt_dark, max_val=1.0) + tf.image.ssim(images[1], gt_haze, max_val=1.0) + tf.image.ssim(images[2], gt_rain, max_val=1.0)) / 3
    print('single ssim is: ', single_ssim)
    ssim.append(single_ssim)
    i += 1

psnr = tf.reduce_mean(psnr)
ssim = tf.reduce_mean(ssim)

print('PSNR is: ', psnr)
print('SSIM is: ', ssim)



