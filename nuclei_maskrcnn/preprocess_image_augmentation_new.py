import numpy as np
import random
import skimage.io
import matplotlib.pyplot as plt
from skimage import transform
from glob import glob
import math
import pickle
import cv2
import os

# dir_ori = './data/stage1_train/'
# dir_aug = './data/augmented/'

dir_ori = './data/external_processed_0326/'
dir_aug = './data/external_processed_0326/'

def read_image_labels(image_id):

    image_file = dir_ori+'{}/images/*.png'.format(image_id)
    mask_file = dir_ori+'{}/masks/*.png'.format(image_id)
    image = np.squeeze(skimage.io.imread_collection(image_file).concatenate())
    masks = skimage.io.imread_collection(mask_file).concatenate()
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width, num_masks), np.uint8)
    for index in range(0, num_masks):
        labels[:,:,index] = masks[index,:,:]
    return image, labels

def data_aug(image, label, split_num, resize_rate):

    seedk = 0

    for i in range(split_num):

        for j in range(1,3):

            new_image = image.copy()
            new_label = label.copy()

            if j == 1:
                new_image = new_image[:, ::-1, :]
                new_label = new_label[:, ::-1, :]
            elif j == 2:
                new_image = new_image[::-1, :, :]
                new_label = new_label[::-1, :, :]
            elif j == 3:
                new_image = new_image[::-1, :, :]
                new_label = new_label[::-1, :, :]
                new_image = new_image[:, ::-1, :]
                new_label = new_label[:, ::-1, :]
            elif j == 4:
                new_image = np.rot90(new_image)
                new_label = np.rot90(new_label)
            elif j == 5:
                new_image = np.rot90(new_image)
                new_label = np.rot90(new_label)
                new_image = new_image[::-1, :, :]
                new_label = new_label[::-1, :, :]
            elif j == 6:
                new_image = np.rot90(new_image)
                new_label = np.rot90(new_label)
                new_image = new_image[:, ::-1, :]
                new_label = new_label[:, ::-1, :]
            elif j == 7:
                new_image = np.rot90(new_image)
                new_label = np.rot90(new_label)
                new_image = new_image[:, ::-1, :]
                new_label = new_label[:, ::-1, :]
                new_image = new_image[::-1, :, :]
                new_label = new_label[::-1, :, :]

            random.seed(seedk)
            np.random.seed(seedk)

            scale1 = random.randint(80, 120) / 100.
            scale2 = random.randint(80, 120) / 100.
            new_image = transform.rescale(new_image, scale=(scale1,scale2),mode='edge',clip=False)
            new_label = transform.rescale(new_label, scale=(scale1,scale2),mode='edge',clip=False)

            size1 = new_image.shape[0]
            size2 = new_image.shape[1]
            rsize = random.randint(np.floor(resize_rate*size1), size1)
            dsize = random.randint(np.floor(resize_rate*size2), size2)
            w_s = random.randint(0, size1 - rsize)
            h_s = random.randint(0, size2 - dsize)
            new_image = new_image[w_s:w_s+rsize,h_s:h_s+dsize,:]
            new_label = new_label[w_s:w_s+rsize,h_s:h_s+dsize,:]

            masks = np.zeros((new_label.shape[0],new_label.shape[1]), np.uint16)
            for index in range(0,new_label.shape[2]):
                masks[new_label[:,:,index]>0] = index+1

            for index in range(0, new_label.shape[2]):
                mask_tmp = np.zeros((new_label.shape[0], new_label.shape[1]), np.uint8)
                mask_tmp[masks==index+1] = 255
                if np.sum(mask_tmp>0) >= 9:
                    if not os.path.exists(dir_aug+'{}_{}/images/'.format(image_id, seedk)):
                        os.makedirs(dir_aug+'{}_{}/images/'.format(image_id, seedk))
                    if not os.path.exists(dir_aug+'{}_{}/masks/'.format(image_id, seedk)):
                        os.makedirs(dir_aug+'{}_{}/masks/'.format(image_id, seedk))

                    aug_img_dir = dir_aug+'{}_{}/images/{}.png'.format(image_id, seedk, image_id)
                    if not os.path.exists(aug_img_dir):
                        plt.imsave(fname=aug_img_dir, arr=new_image)
                    aug_mask_dir = dir_aug+'{}_{}/masks/{}_{}a.png'.format(image_id, seedk, image_id, index)
                    cv2.imwrite(filename=aug_mask_dir, img=mask_tmp)

            seedk += 1
    return

id_all              = glob(dir_ori+'*')
pickle_in           = open("image_group_train.pickle")
image_group_train   = pickle.load(pickle_in)
split_nums          = [1,1,9,3,3,3,9,6]

for id_curr in id_all:
    image_id = id_curr.split('/')[-1]
    image, labels = read_image_labels(image_id)
    print image.shape, labels.shape

    split_num = 1
    for group in range(1,9):
        if image_id in image_group_train[group]:
            split_num = split_nums[group-1]

    print image_id, group, split_num
    data_aug(image, labels, split_num, resize_rate=0.75)



