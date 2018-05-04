###########################################
# Library and Path
###########################################

import os
import random
import pickle
import numpy as np
import skimage.io
from skimage.color import gray2rgb, label2rgb
from sklearn.model_selection import train_test_split
from scipy.ndimage import binary_fill_holes
import pandas as pd
import time
import glob
import cv2

from nuclei_config import Config
import nuclei_utils as utils
import nuclei_model as modellib

GPU_option = 0
log_name = "logs"

# Directory of the project and models
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, log_name)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of nuclei data
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR,"stage1_train")
TRAIN_DATA_EXT_PATH = os.path.join(DATA_DIR,"external_processed")
TRAIN_DATA_MOSAIC_PATH = os.path.join(ROOT_DIR,"mosaic","stage1_train_mosaic")
TEST_DATA_MOSAIC_PATH = os.path.join(ROOT_DIR,"mosaic","stage1_test_mosaic")
TEST_DATA_PATH = os.path.join(DATA_DIR,"stage1_test")
TEST_MASK_SAVE_PATH = os.path.join(DATA_DIR,"stage1_masks_test")
TEST_VAL_MASK_SAVE_PATH = os.path.join(DATA_DIR,"stage1_masks_val")

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_option)

import tensorflow as tf
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
session = tf.Session(config=config_tf)

label_all = False
vsave_flag = True
val_flag = True
val_group_flag = True
test_flag = True
test_group_flag = True

model_path = '~/nuclei_maskrcnn/logs/nuclei_train20180000T0000/mask_rcnn_nuclei_train_0000.h5'
###########################################
# Train vs. validation split
###########################################

train_ids = []
val_ids = []
rep_id = [2,2,8,6,4,4,8]

df = pd.read_csv('image_group_train.csv')
ids = df['id']
groups = df['group']
istrain = df['istrain']
mosaic_ids = df['mosaic_id']
train_ids_mosaic = np.unique(mosaic_ids[istrain==1])[1:]

for k in range(len(ids)):
    if istrain[k]:
        for iter_tmp in range(rep_id[groups[k] - 1]):
            train_ids.append(ids[k])
        if label_all:
            val_ids.append(ids[k])
    else:
        val_ids.append(ids[k])

for k in range(len(train_ids)):
    train_ids[k] = os.path.join(TRAIN_DATA_PATH, train_ids[k], 'images', train_ids[k] + '.png')
train_ids_ext = glob.glob(TRAIN_DATA_EXT_PATH + '/*/images/*.png')
train_ids.extend(train_ids_ext)

for k, id_m in enumerate(train_ids_mosaic):
    train_ids_mosaic[k] = os.path.join(TRAIN_DATA_MOSAIC_PATH, id_m, 'images', id_m + '_image.png')
train_ids.extend(train_ids_mosaic)

test_ids = next(os.walk(TEST_DATA_PATH))[1]
test_ids_m = next(os.walk(TEST_DATA_MOSAIC_PATH))[1]

print('train = ' + str(len(train_ids)) + ' external = ' + str(len(train_ids_ext)))

dim_min = 512
dim_max = 1024

foldername_suffix = '_' + str(dim_min) + '_' + str(dim_max) + '_mod_bound_all'

###########################################
# Inference Config
###########################################
class TrainingConfig(Config):
    NAME = "nuclei_train"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    NUM_CLASSES = 1 + 1
    VALIDATION_STEPS = 3

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_TRAIN_ANCHORS_PER_IMAGE = 384
    TRAIN_ROIS_PER_IMAGE = 512
    MAX_GT_INSTANCES = 400
    DETECTION_MAX_INSTANCES = 400
    RPN_NMS_THRESHOLD = 0.7
    IMAGE_PADDING = True

    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    BACKBONE_NAME = 'resnet50'

class InferenceConfig(TrainingConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    SAVE_PROB_MASK = False
    POST_NMS_ROIS_INFERENCE = 1000

    if '201803' in model_path:
        RPN_ANCHOR_SCALES = (16, 32, 64, 128)
        RPN_TRAIN_ANCHORS_PER_IMAGE = 256
        TRAIN_ROIS_PER_IMAGE = 256

inference_config = InferenceConfig(dim_max, dim_min)
inference_config.display()

###########################################
# Load Nuclei Dataset
###########################################

class NucleiDataset(utils.Dataset):
    def load_image(self, image_id):
        # Load the specified image and return a [H,W,3] Numpy array.
        image = skimage.io.imread(self.image_info[image_id]['path'])
        if image.ndim != 3:
            image = gray2rgb(image)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        return image
    def load_mask(self, image_id):
        # Load the instance masks (a binary mask per instance)
        # return a a bool array of shape [H, W, instance count]
        mask_dir = os.path.dirname(self.image_info[image_id]['path']).replace('images', 'masks')
        mask_files = next(os.walk(mask_dir))[2]
        num_inst = len(mask_files)
        # get the shape of the image
        mask0 = skimage.io.imread(os.path.join(mask_dir, mask_files[0]))
        class_ids = np.ones(len(mask_files), np.int32)
        mask = np.zeros([mask0.shape[0], mask0.shape[1], num_inst])
        for k in range(num_inst):
            mask[:, :, k] = skimage.io.imread(os.path.join(mask_dir, mask_files[k]))
        return mask, class_ids

###########################################
# Data prepare
###########################################

dataset_val = NucleiDataset()
dataset_val.add_class("cell", 1, "nulcei")
for k, val_id in enumerate(val_ids):
    dataset_val.add_image("cell", k, os.path.join(TRAIN_DATA_PATH, val_id, 'images', val_id + '.png'))
dataset_val.prepare()

dataset_test = NucleiDataset()
dataset_test.add_class("cell", 1, "nulcei")
for k, test_id in enumerate(test_ids):
    dataset_test.add_image("cell", k, os.path.join(TEST_DATA_PATH, test_id, 'images', test_id + '.png'))
dataset_test.prepare()

dataset_test_m = NucleiDataset()
dataset_test_m.add_class("cell", 1, "nulcei")
for k, test_id_m in enumerate(test_ids_m):
    dataset_test_m.add_image("cell", k, os.path.join(TEST_DATA_MOSAIC_PATH, test_id_m, 'images', test_id_m + '_image.png'))
dataset_test_m.prepare()

###########################################
# Validation / Test
###########################################

def compute_val(vflip=False, hflip=False):
    model_inf = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model_inf.load_weights(model_path, by_name=True)
    model_name = model_path.split('/')[-2]
    model_epoch = model_path.split('/')[-1].split('.')[0][-5:]
    model_name = model_name + model_epoch
    if vflip:
        model_name = model_name + '_vflip'
    if hflip:
        model_name = model_name + '_hflip'

    if vsave_flag and not os.path.exists(TEST_VAL_MASK_SAVE_PATH + '/' + model_name):
        os.makedirs(TEST_VAL_MASK_SAVE_PATH + '/' + model_name)

    APs = []
    for image_id in dataset_val.image_ids:
        # test_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        #     modellib.load_image_gt_noresize(dataset_val, inference_config, image_id, use_mini_mask=False)
        test_image = dataset_val.load_image(image_id)
        gt_mask, _ = dataset_val.load_mask(image_id)

        if vflip:
            test_image = cv2.flip(test_image, 0)
        if hflip:
            test_image = cv2.flip(test_image, 1)
        results = model_inf.detect([test_image], verbose=0)
        r = results[0]
        if vflip:
            test_image = cv2.flip(test_image, 0)
            r["masks"] = cv2.flip(r["masks"], 0)
        if hflip:
            test_image = cv2.flip(test_image, 1)
            r["masks"] = cv2.flip(r["masks"], 1)
        masks = r["masks"]
        tmp = masks[:, :, 0].copy()
        if tmp.shape[0] == 0:
            masks = np.zeros(test_image.shape)
        for i in range(masks.shape[2]):
            tmp = masks[:, :, i].copy()
            tmp[0,:] = tmp[1,:]
            tmp[-1,:] = tmp[-2,:]
            tmp[:,0] = tmp[:,1]
            tmp[:,-1] = tmp[:,-2]
            masks[:,:,i] = tmp.copy()
        r["masks"] = masks.copy()
        AP = utils.sweep_iou_mask_ap(gt_mask, r["masks"], r["scores"])
        APs.append(AP)
        print np.mean(APs)

        train_id = dataset_val.image_info[image_id]['path'].split('/')[-1][:-4]
        rmaskcollapse_gt = np.zeros((test_image.shape[0], test_image.shape[1]))
        for i in range(gt_mask.shape[2]):
            rmaskcollapse_gt = rmaskcollapse_gt + gt_mask[:, :, i] * (i + 1)

        masks = r["masks"]
        rmaskcollapse = np.zeros((test_image.shape[0], test_image.shape[1]))
        for i in range(masks.shape[2]):
            rmaskcollapse = rmaskcollapse + masks[:, :, i] * (i + 1)

        tmp1 = rmaskcollapse_gt.copy()
        tmp1[rmaskcollapse_gt > 0] = 1
        tmp2 = rmaskcollapse.copy()
        tmp2[rmaskcollapse > 0] = 2
        overlap = tmp1+tmp2

        skimage.io.imsave(
            TEST_VAL_MASK_SAVE_PATH + '/' + model_name + '/ap_' + '%.2f' % AP + '_' + train_id + '_mask.png',
            np.concatenate((label2rgb(rmaskcollapse_gt, bg_label=0), test_image / 255.,
                            label2rgb(rmaskcollapse, bg_label=0),
                            label2rgb(overlap, bg_label=0)), axis=1))

        np.save(TEST_VAL_MASK_SAVE_PATH + '/' + model_name + '/' + train_id + '_mask.npy', rmaskcollapse)
        np.save(TEST_VAL_MASK_SAVE_PATH + '/' + model_name + '/' + train_id + '_gtmask.npy', rmaskcollapse_gt)

    print("mAP: ", np.mean(APs))
    del model_inf

def compute_val_group(vflip=False, hflip=False):
    min_dim = []
    max_dim =[]
    min_dim_config = []
    max_dim_config = []

    for image_id in dataset_val.image_ids:
        test_image = dataset_val.load_image(image_id)
        min_dim.append(min(test_image.shape[:2]))
        max_dim.append(max(test_image.shape[:2]))
        min_dim_config.append(int(np.floor(min_dim[-1] / 64.) * 64.))
        max_dim_config.append(int(np.floor(np.sqrt(max_dim[-1] * min_dim[-1] * 4) / 64.) * 64))

    max_dim_config = np.array(max_dim_config)
    min_dim_config = np.array(min_dim_config)
    max_dim = np.array(max_dim)
    min_dim = np.array(min_dim)

    max_min_dim_config = max_dim_config * 10000 + min_dim_config
    max_min_dim_config_sort_ind = np.argsort(max_min_dim_config)
    max_dim_config = max_dim_config[max_min_dim_config_sort_ind]
    min_dim_config = min_dim_config[max_min_dim_config_sort_ind]
    max_min_dim_config = max_min_dim_config[max_min_dim_config_sort_ind]
    max_dim = max_dim[max_min_dim_config_sort_ind]
    min_dim = min_dim[max_min_dim_config_sort_ind]
    test_image_ids = dataset_val.image_ids[max_min_dim_config_sort_ind]

    APs = []
    hm_models = 0  # how many models used
    for i, image_id in enumerate(test_image_ids):
        if (i == 0) or (max_min_dim_config[i] != max_min_dim_config[i - 1]):
            inference_config_group = InferenceConfig(max_dim_config[i], min_dim_config[i])
            model_inf = modellib.MaskRCNN(mode="inference",
                                          config=inference_config_group,
                                          model_dir=MODEL_DIR)
            assert model_path != "", "Provide path to trained weights"
            print("Loading weights from ", model_path)
            model_inf.load_weights(model_path, by_name=True)
            model_name = model_path.split('/')[-2]
            model_epoch = model_path.split('/')[-1].split('.')[0][-5:]
            model_name = model_name + model_epoch + '_group'
            if vflip:
                model_name = model_name + '_vflip'
            if hflip:
                model_name = model_name + '_hflip'

            if not os.path.exists(TEST_VAL_MASK_SAVE_PATH + '/' + model_name):
                os.makedirs(TEST_VAL_MASK_SAVE_PATH + '/' + model_name)

            hm_models += 1

        print max_dim[i], min_dim[i], max_dim_config[i], min_dim_config[i]
        # test_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        #     modellib.load_image_gt_noresize(dataset_val, inference_config_group, image_id, use_mini_mask=False)
        test_image = dataset_val.load_image(image_id)
        gt_mask, _ = dataset_val.load_mask(image_id)

        test_id = dataset_val.image_info[image_id]['path'].split('/')[-1][:-4]
        if vflip:
            test_image = cv2.flip(test_image, 0)
        if hflip:
            test_image = cv2.flip(test_image, 1)
        results = model_inf.detect([test_image], verbose=0)
        r = results[0]
        if vflip:
            test_image = cv2.flip(test_image, 0)
            r["masks"] = cv2.flip(r["masks"], 0)
        if hflip:
            test_image = cv2.flip(test_image, 1)
            r["masks"] = cv2.flip(r["masks"], 1)
        masks = r["masks"]
        tmp = masks[:, :, 0].copy()
        if tmp.shape[0] == 0:
            masks = np.zeros(test_image.shape)
        for i in range(masks.shape[2]):
            tmp = masks[:, :, i].copy()
            tmp[0, :] = tmp[1, :]
            tmp[-1, :] = tmp[-2, :]
            tmp[:, 0] = tmp[:, 1]
            tmp[:, -1] = tmp[:, -2]
            masks[:, :, i] = tmp.copy()
        r["masks"] = masks.copy()
        AP = utils.sweep_iou_mask_ap(gt_mask, r["masks"], r["scores"])
        APs.append(AP)
        print np.mean(APs)

        rmaskcollapse_gt = np.zeros((test_image.shape[0], test_image.shape[1]))
        for i in range(gt_mask.shape[2]):
            rmaskcollapse_gt = rmaskcollapse_gt + gt_mask[:, :, i] * (i + 1)

        masks = r["masks"]
        rmaskcollapse = np.zeros((test_image.shape[0], test_image.shape[1]))
        for i in range(masks.shape[2]):
            rmaskcollapse = rmaskcollapse + masks[:, :, i] * (i + 1)

        tmp1 = rmaskcollapse_gt.copy()
        tmp1[rmaskcollapse_gt > 0] = 1
        tmp2 = rmaskcollapse.copy()
        tmp2[rmaskcollapse > 0] = 2
        overlap = tmp1 + tmp2

        skimage.io.imsave(
            TEST_VAL_MASK_SAVE_PATH + '/' + model_name + '/ap_' + '%.2f' % AP + '_' + test_id + '_mask.png',
            np.concatenate((label2rgb(rmaskcollapse_gt, bg_label=0),
                            test_image / 255.,
                            label2rgb(rmaskcollapse, bg_label=0),
                            label2rgb(overlap, bg_label=0)), axis=1))
        np.save(TEST_VAL_MASK_SAVE_PATH + '/' + model_name + '/' + test_id + '_mask.npy', rmaskcollapse)
        np.save(TEST_VAL_MASK_SAVE_PATH + '/' + model_name + '/' + test_id + '_gtmask.npy', rmaskcollapse_gt)

    print("mAP: ", np.mean(APs))
    del model_inf

def compute_test(vflip=False,hflip=False):

    model_inf = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model_inf.load_weights(model_path, by_name=True)
    model_name = model_path.split('/')[-2]
    model_epoch = model_path.split('/')[-1].split('.')[0][-5:]
    model_name = model_name + model_epoch
    if vflip:
        model_name = model_name + '_vflip'
    if hflip:
        model_name = model_name + '_hflip'

    new_test_ids = []
    rles = []
    if not os.path.exists(TEST_MASK_SAVE_PATH + '/' + model_name):
        os.makedirs(TEST_MASK_SAVE_PATH + '/' + model_name)

    for image_id in dataset_test.image_ids:
        test_image = dataset_test.load_image(image_id)
        if vflip:
            test_image = cv2.flip(test_image, 0)
        if hflip:
            test_image = cv2.flip(test_image, 1)

        results = model_inf.detect([test_image], verbose=0)
        r = results[0]
        if vflip:
            test_image = cv2.flip(test_image, 0)
            r["masks"] = cv2.flip(r["masks"], 0)
        if hflip:
            test_image = cv2.flip(test_image, 1)
            r["masks"] = cv2.flip(r["masks"], 1)
        masks = r["masks"]
        tmp = masks[:, :, 0].copy()
        if tmp.shape[0] == 0:
            masks = np.zeros(test_image.shape)
        for i in range(masks.shape[2]):
            tmp = masks[:, :, i].copy()
            tmp[0, :] = tmp[1, :]
            tmp[-1, :] = tmp[-2, :]
            tmp[:, 0] = tmp[:, 1]
            tmp[:, -1] = tmp[:, -2]
            masks[:, :, i] = tmp.copy()
        r["masks"] = masks.copy()
        test_id = dataset_test.image_info[image_id]['path'].split('/')[-1][:-4]

        rmaskcollapse = np.zeros((r['masks'].shape[0], r['masks'].shape[1]))
        for i in range(r['masks'].shape[2]):
            rmaskcollapse = rmaskcollapse + r['masks'][:, :, i] * (i + 1)

        tmp = test_image[:, :, 0].copy()
        tmp[rmaskcollapse > 0] = 250
        test_image[:, :, 0] = tmp.copy()
        tmp = test_image[:, :, 1].copy()
        tmp[rmaskcollapse > 0] = 200
        test_image[:, :, 1] = tmp.copy()
        tmp = test_image[:, :, 2].copy()
        tmp[rmaskcollapse > 0] = 80
        test_image[:, :, 2] = tmp.copy()

        skimage.io.imsave(TEST_MASK_SAVE_PATH + '/' + model_name + '/' + test_id + '_mask.png',
                          np.concatenate((dataset_test.load_image(image_id) / 255.,
                                          label2rgb(rmaskcollapse, bg_label=0), test_image /255.), axis=1))
        np.save(TEST_MASK_SAVE_PATH + '/' + model_name + '/' + test_id + '_mask.npy',rmaskcollapse)
        for i in range(r['masks'].shape[2]):
            rle = list(utils.prob_to_rles(r['masks'][:, :, i]))
            rles.extend(rle)
            new_test_ids.append(test_id)
    del model_inf

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('submission/sub-test-'+model_name+'.csv', index=False)

def compute_test_group(vflip=False, hflip=False):
    min_dim = []
    max_dim = []
    min_dim_config = []
    max_dim_config = []

    for image_id in dataset_test.image_ids:
        test_image = dataset_test.load_image(image_id)
        min_dim.append(min(test_image.shape[:2]))
        max_dim.append(max(test_image.shape[:2]))
        min_dim_config.append(int(np.floor(min_dim[-1] / 64.) * 64.))
        max_dim_config.append(int(np.floor(np.sqrt(max_dim[-1] * min_dim[-1] * 4) / 64.) * 64))

    max_dim_config = np.array(max_dim_config)
    min_dim_config = np.array(min_dim_config)
    max_dim = np.array(max_dim)
    min_dim = np.array(min_dim)

    max_min_dim_config = max_dim_config * 10000 + min_dim_config
    max_min_dim_config_sort_ind = np.argsort(max_min_dim_config)
    max_dim_config = max_dim_config[max_min_dim_config_sort_ind]
    min_dim_config = min_dim_config[max_min_dim_config_sort_ind]
    max_min_dim_config = max_min_dim_config[max_min_dim_config_sort_ind]
    max_dim = max_dim[max_min_dim_config_sort_ind]
    min_dim = min_dim[max_min_dim_config_sort_ind]
    test_image_ids = dataset_test.image_ids[max_min_dim_config_sort_ind]

    new_test_ids = []
    rles = []

    hm_models = 0  # how many models used
    for i, image_id in enumerate(test_image_ids):
        if (i == 0) or (max_min_dim_config[i] != max_min_dim_config[i - 1]):
            inference_config_group = InferenceConfig(max_dim_config[i], min_dim_config[i])
            model_inf = modellib.MaskRCNN(mode="inference",
                                          config=inference_config_group,
                                          model_dir=MODEL_DIR)
            assert model_path != "", "Provide path to trained weights"
            print("Loading weights from ", model_path)
            model_inf.load_weights(model_path, by_name=True)
            model_name = model_path.split('/')[-2]
            model_epoch = model_path.split('/')[-1].split('.')[0][-5:]
            model_name = model_name + model_epoch + '_group'
            if vflip:
                model_name = model_name + '_vflip'
            if hflip:
                model_name = model_name + '_hflip'

            if not os.path.exists(TEST_MASK_SAVE_PATH + '/' + model_name):
                os.makedirs(TEST_MASK_SAVE_PATH + '/' + model_name)

            hm_models += 1

        test_image = dataset_test.load_image(image_id)
        if vflip:
            test_image = cv2.flip(test_image, 0)
        if hflip:
            test_image = cv2.flip(test_image, 1)

        print min(test_image.shape[:2]), max(test_image.shape[:2]), max_dim_config[i], min_dim_config[i]
        test_id = dataset_test.image_info[image_id]['path'].split('/')[-1][:-4]

        results = model_inf.detect([test_image], verbose=0)
        r = results[0]
        if vflip:
            test_image = cv2.flip(test_image, 0)
            r["masks"] = cv2.flip(r["masks"], 0)
        if hflip:
            test_image = cv2.flip(test_image, 1)
            r["masks"] = cv2.flip(r["masks"], 1)
        masks = r["masks"]
        tmp = masks[:, :, 0].copy()
        if tmp.shape[0] == 0:
            masks = np.zeros(test_image.shape)
        for i in range(masks.shape[2]):
            tmp = masks[:, :, i].copy()
            tmp[0, :] = tmp[1, :]
            tmp[-1, :] = tmp[-2, :]
            tmp[:, 0] = tmp[:, 1]
            tmp[:, -1] = tmp[:, -2]
            masks[:, :, i] = tmp.copy()
        r["masks"] = masks.copy()
        masks = r["masks"]
        rmaskcollapse = np.zeros((test_image.shape[0], test_image.shape[1]))
        for i in range(masks.shape[2]):
            rmaskcollapse = rmaskcollapse + masks[:, :, i] * (i + 1)

        tmp = test_image[:, :, 0].copy()
        tmp[rmaskcollapse > 0] = 250
        test_image[:, :, 0] = tmp.copy()
        tmp = test_image[:, :, 1].copy()
        tmp[rmaskcollapse > 0] = 200
        test_image[:, :, 1] = tmp.copy()
        tmp = test_image[:, :, 2].copy()
        tmp[rmaskcollapse > 0] = 80
        test_image[:, :, 2] = tmp.copy()

        skimage.io.imsave(TEST_MASK_SAVE_PATH + '/' + model_name + '/' + test_id + '_mask.png',
                          np.concatenate((dataset_test.load_image(image_id) / 255.,
                                          label2rgb(rmaskcollapse, bg_label=0), test_image / 255.), axis=1))
        np.save(TEST_MASK_SAVE_PATH + '/' + model_name + '/' + test_id + '_mask.npy', rmaskcollapse)
        for i in range(r['masks'].shape[2]):
            rle = list(utils.prob_to_rles(r['masks'][:, :, i]))
            rles.extend(rle)
            new_test_ids.append(test_id)

    del model_inf
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('submission/sub-test-'+model_name+'.csv', index=False)

if val_flag:
    compute_val(vflip=False, hflip=False)
    compute_val(vflip=True, hflip=False)
    compute_val(vflip=False, hflip=True)

if val_group_flag:
    compute_val_group(vflip=False, hflip=False)
    compute_val_group(vflip=True, hflip=False)
    compute_val_group(vflip=False, hflip=True)

if test_flag:
    compute_test(vflip=False, hflip=False)
    compute_test(vflip=True, hflip=False)
    compute_test(vflip=False, hflip=True)

if test_group_flag:
    compute_test_group(vflip=False, hflip=False)
    compute_test_group(vflip=True, hflip=False)
    compute_test_group(vflip=False, hflip=True)