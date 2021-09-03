import imlib as im
import pylib as py
import tf2lib as tl
import data
import networks as net
import tensorflow as tf
import numpy as np
from tensorflow.python.data.experimental import AUTOTUNE, prefetch_to_device



# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir', default='slice')
# py.arg('--generator_model', default='MPRNet_1stage', choices=['ResNet', 'UNet', 'MPRNet_1stage', 'MPRNet_2stages'])
test_args = py.args()

args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

# read train dataset
noise_img_paths = sorted(py.glob(py.join(args.datasets_dir, args.dataset, 'train', 'noise'), '*.jpg'))
haze_img_paths = sorted(py.glob(py.join(args.datasets_dir, args.dataset, 'train', 'haze'), '*.jpg'))

print(noise_img_paths[:5])
print(haze_img_paths[:5])

single_dataset, len_dataset = data.make_zip_dataset(noise_img_paths, haze_img_paths, args.batch_size, args.load_size, args.crop_size, training=False, repeat=False, shuffle=False)

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
for noise, gt_image in single_dataset:
    sliced = generator(noise)
    img = im.immerge(np.concatenate([noise, sliced, gt_image], axis=0), n_rows=1)
    im.imwrite(img, py.join(save_dir, 'image-%05d.jpg' % i))
    single_psnr = tf.image.psnr(sliced, gt_image, max_val=1.0)
    print('single psnr is: ', single_psnr)
    psnr.append(single_psnr)
    single_ssim = tf.image.ssim(sliced, gt_image, max_val=1.0)
    print('single ssim is: ', single_ssim)
    ssim.append(single_ssim)
    i += 1

psnr = tf.reduce_mean(psnr)
ssim = tf.reduce_mean(ssim)

print('PSNR is: ', psnr)
print('SSIM is: ', ssim)



