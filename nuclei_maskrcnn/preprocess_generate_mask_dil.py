from scipy.ndimage.morphology import binary_dilation
import os
import numpy as np
import skimage.io

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR,"stage1_train")
TRAIN_DIL_DATA_PATH = os.path.join(DATA_DIR,"stage1_train_dil_masks")

train_ids = next(os.walk(TRAIN_DATA_PATH))[1]

def load_mask(image_id):
    # Load the instance masks (a binary mask per instance)
    # return a a bool array of shape [H, W, instance count]
    mask_dir = os.path.join(TRAIN_DATA_PATH, image_id, 'masks')
    mask_files = next(os.walk(mask_dir))[2]
    num_inst = len(mask_files)

    # get the shape of the image
    mask0 = skimage.io.imread(os.path.join(mask_dir, mask_files[0]))
    class_ids = np.ones(len(mask_files), np.int32)
    mask = np.zeros([mask0.shape[0], mask0.shape[1], num_inst])
    for k in range(num_inst):
        mask[:, :, k] = skimage.io.imread(os.path.join(mask_dir, mask_files[k]))
    return mask

for train_id in train_ids:
    masks = load_mask(train_id)
    masks_dil = np.zeros(masks.shape)
    for k in range(masks.shape[2]):
        dilmask = binary_dilation(binary_dilation(masks[:, :, k]))
        masks_dil[:,:,k] = dilmask.astype('uint8')*255 - masks[:,:,k]

    np.save(TRAIN_DIL_DATA_PATH+'/'+train_id+'_dil_masks.npy',masks_dil.astype('uint8'))
    skimage.io.imsave(TRAIN_DIL_DATA_PATH+'/'+train_id+'_eg.png',masks_dil[:,:,0].astype('uint8'))

