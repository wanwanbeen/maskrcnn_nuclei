import numpy as np
import random
import skimage.io
import matplotlib.pyplot as plt
from skimage import transform
from glob import glob
import cv2
import os

def read_image_labels(image_id):

    image_file = './data/stage1_train/{}/images/{}.png'.format(image_id,image_id)
    mask_file = './data/stage1_train/{}/masks/*.png'.format(image_id)
    image = skimage.io.imread(image_file)
    masks = skimage.io.imread_collection(mask_file).concatenate()
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width, num_masks), np.uint8)
    for index in range(0, num_masks):
        labels[:,:,index] = masks[index,:,:]
    return image, labels

def data_aug(image, label, split_num, seedk,angel, resize_rate):

    for i in range(split_num):

        random.seed(seedk*5+i)
        np.random.seed(seedk*5+i)

        flip1 = random.randint(0, 1)
        flip2 = random.randint(0, 1)
        size1 = image.shape[0]
        size2 = image.shape[1]
        rsize = random.randint(np.floor(resize_rate*size1), size1)
        dsize = random.randint(np.floor(resize_rate*size2), size2)
        w_s = random.randint(0,size1 - rsize)
        h_s = random.randint(0,size2 - dsize)
        sh = random.random()/10
        scale1 = random.randint(90, 110) / 100.
        scale2 = random.randint(90, 110) / 100.
        rotate_angel = random.random()/180*np.pi*angel
        affine_tf = transform.AffineTransform(scale=(scale1,scale2),shear=sh,rotation=rotate_angel)
        new_image = transform.warp(image, inverse_map=affine_tf,mode='edge')
        new_label = transform.warp(label, inverse_map=affine_tf,mode='edge')
        new_image = new_image[w_s:np.int(size1*0.9),h_s:np.int(size2*0.9),:]
        new_label = new_label[w_s:np.int(size1*0.9),h_s:np.int(size2*0.9),:]
        if flip1:
            new_image = new_image[:,::-1,:]
            new_label = new_label[:,::-1,:]
        if flip2:
            new_image = new_image[::-1,:,:]
            new_label = new_label[::-1,:,:]

        masks = np.zeros((new_label.shape[0],new_label.shape[1]), np.uint16)
        for index in range(0,new_label.shape[2]):
            masks[new_label[:,:,index]>0] = index+1

        for index in range(0, new_label.shape[2]):
            mask_tmp = np.zeros((new_label.shape[0], new_label.shape[1]), np.uint8)
            mask_tmp[masks==index+1] = 255
            if np.sum(mask_tmp>0) > 9:

                if not os.path.exists('./data/augmented/{}_{}/images/'.format(image_id, i)):
                    os.makedirs('./data/augmented/{}_{}/images/'.format(image_id, i))
                if not os.path.exists("./data/augmented/{}_{}/masks/".format(image_id, i)):
                    os.makedirs('./data/augmented/{}_{}/masks/'.format(image_id, i))

                aug_img_dir = "./data/augmented/{}_{}/images/{}.png".format(image_id, i, image_id)
                if not os.path.exists(aug_img_dir):
                    plt.imsave(fname=aug_img_dir, arr=new_image)

                aug_mask_dir = "./data/augmented/{}_{}/masks/{}_{}a.png".format(image_id, i, image_id, index)
                cv2.imwrite(filename=aug_mask_dir, img=mask_tmp)

    return

id_all      = glob('./data/stage1_train/*')
split_num   = 10
seedk       = 0
for id_curr in id_all:
    image_id = id_curr.split('/')[-1]
    image, labels = read_image_labels(image_id)

    split_num = 1 if np.sum(image[:,:,0]-image[:,:,1]) == 0 \
                     and np.sum(image[:,:,1]-image[:,:,2]) == 0 else 5

    data_aug(image, labels, split_num, seedk, angel=10, resize_rate=0.8)
    seedk += 1



