import numpy as np
import random
import skimage.io
from scipy.stats import mode
# import matplotlib.pyplot as plt
from skimage import transform
from scipy.ndimage.morphology import binary_dilation
from glob import glob
import math
import pickle
# import cv2
import os

crop_size = 256

def read_image_labels(dir_ori, image_id):

    image_file = glob(dir_ori+'{}/images/*.png'.format(image_id,image_id))
    mask_file = dir_ori+'{}/masks/*.png'.format(image_id)
    image = skimage.io.imread(image_file[0])
    masks = skimage.io.imread_collection(mask_file).concatenate()
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width, num_masks), np.uint8)
    for index in range(0, num_masks):
        labels[:,:,index] = masks[index,:,:]
    return image, labels

def data_aug(dir_aug, image, label, split_num, seedk, angel, resize_rate):

    for i in range(split_num+1):

        if i == 0:
            new_image = image.copy()
            new_label = label.copy()
        else:
            random.seed(seedk*5+i)
            np.random.seed(seedk*5+i)

            fliprotate = random.randint(0, 2)
            size1 = image.shape[0]
            size2 = image.shape[1]
            rsize = random.randint(np.floor(resize_rate*size1), size1)
            dsize = random.randint(np.floor(resize_rate*size2), size2)
            w_s = random.randint(0,size1 - rsize)
            h_s = random.randint(0,size2 - dsize)
            sh = random.random()/10.
            scale1 = random.randint(90, 110) / 100.
            scale2 = random.randint(90, 110) / 100.
            rotate_angel = random.random()/180*np.pi*angel
            affine_tf = transform.AffineTransform(scale=(scale1,scale2),shear=sh,rotation=rotate_angel)
            new_image = transform.warp(image, inverse_map=affine_tf,mode='edge')
            new_label = transform.warp(label, inverse_map=affine_tf,mode='edge')
            new_image = new_image[w_s:np.int(size1*0.9),h_s:np.int(size2*0.9),:]
            new_label = new_label[w_s:np.int(size1*0.9),h_s:np.int(size2*0.9),:]
            if fliprotate == 0:
                new_image = np.rot90(new_image)
                new_label = np.rot90(new_label)
            elif fliprotate == 1:
                new_image = new_image[::-1, :, :]
                new_label = new_label[::-1, :, :]
                new_image = new_image[:, ::-1, :]
                new_label = new_label[:, ::-1, :]
            elif fliprotate == 2:
                new_image = new_image[::-1, :, :]
                new_label = new_label[::-1, :, :]
                new_image = new_image[:, ::-1, :]
                new_label = new_label[:, ::-1, :]
                new_image = np.rot90(new_image)
                new_label = np.rot90(new_label)

        num_x = int(np.ceil(new_label.shape[0] * 1./ crop_size))
        num_y = int(np.ceil(new_label.shape[1] * 1./ crop_size))
        run_x = new_label.shape[0] / num_x
        run_y = new_label.shape[1] / num_y
        x0 = (crop_size - run_x) / 2
        y0 = (crop_size - run_y) / 2

        if np.max(new_image) <= 1:
            new_image = np.uint8(np.round(new_image * 255))
        if np.max(new_label) <= 1:
            new_label[new_label < 0.5] = 0
            new_label[new_label >= 0.5] = 255

        for nx in range(num_x):
            for ny in range(num_y):

                nxy = nx * num_y + ny

                new_image_crop = np.zeros((crop_size,crop_size,3), np.uint8)
                new_label_crop = np.zeros((crop_size,crop_size,new_label.shape[2]), np.uint8)
                new_image_crop[x0:x0+run_x, y0:y0+run_y,:] = new_image[nx*run_x:nx*run_x+run_x, ny*run_y:ny*run_y+run_y, :3].copy()
                new_label_crop[x0:x0+run_x, y0:y0+run_y,:] = new_label[nx*run_x:nx*run_x+run_x, ny*run_y:ny*run_y+run_y, :].copy()

                masks = np.zeros((new_label_crop.shape[0],new_label_crop.shape[1]), np.uint16)
                for index in range(0,new_label_crop.shape[2]):
                    masks[new_label_crop[:,:,index]>0] = index+1

                for index in range(0, new_label_crop.shape[2]):
                    mask_tmp = np.zeros((new_label_crop.shape[0], new_label_crop.shape[1]), np.uint8)
                    mask_tmp[masks==index+1] = 255

                    edge_detect = np.sum(mask_tmp[x0, :] > 0)
                    edge_detect = edge_detect + np.sum(mask_tmp[x0+run_x-1, :] > 0)
                    edge_detect = edge_detect + np.sum(mask_tmp[:, y0] > 0)
                    edge_detect = edge_detect + np.sum(mask_tmp[:, y0+run_y-1] > 0)

                    if edge_detect > 0:
                        suffix = 'b' # boundary
                    else:
                        suffix = 'i' # inside

                    if np.sum(mask_tmp>0) > 9:
                        if not os.path.exists(dir_aug+'{}_{}_{}/images/'.format(image_id, i, nxy)):
                            os.makedirs(dir_aug+'{}_{}_{}/images/'.format(image_id, i, nxy))
                        if not os.path.exists(dir_aug+'{}_{}_{}/masks/'.format(image_id, i, nxy)):
                            os.makedirs(dir_aug+'{}_{}_{}/masks/'.format(image_id, i, nxy))

                        aug_img_dir = dir_aug+'{}_{}_{}/images/{}.png'.format(image_id, i, nxy, image_id)
                        # plt.imsave(fname=aug_img_dir, arr=new_image_crop
                        skimage.io.imsave(fname=aug_img_dir, arr=new_image_crop)
                        aug_mask_dir = dir_aug+'{}_{}_{}/masks/{}_{}{}.png'.format(image_id, i, nxy, image_id, index,suffix)
                        skimage.io.imsave(fname=aug_mask_dir, arr=mask_tmp)
    return

for n_test in range(0,2):

    if n_test == 0:
        dir_ori = './data/stage1_train/'
        dir_aug = './data/stage1_train_crop/'
    else:
        dir_ori = './data/external_processed/'
        dir_aug = './data/external_processed_crop/'

    id_all      = glob(dir_ori+'*')
    seedk       = 0

    pickle_in  = open("image_group_train.pickle")
    image_group_train = pickle.load(pickle_in)

    for id_curr in id_all:
        image_id = id_curr.split('/')[-1]
        image, labels = read_image_labels(dir_ori, image_id)

        split_num = 1
        for group in range(1,9):
            if image_id in image_group_train[group]:
                split_num = int(math.ceil(100./len(image_group_train[group])))
                print  n_test, seedk, image_id, group, split_num, image.shape[0], image.shape[1]

        if n_test == 1:
            print  n_test, seedk, image_id, split_num, image.shape[0], image.shape[1]

        data_aug(dir_aug, image, labels, split_num, seedk, angel=10, resize_rate=0.8)
        seedk += 1




