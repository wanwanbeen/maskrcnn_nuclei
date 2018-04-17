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
import pandas as pd
import time
import glob

from nuclei_config import Config
import nuclei_utils as utils
import nuclei_model as modellib

GPU_option = 0
log_name = "logs_par" if GPU_option else "logs"

# Directory of the project and models
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, log_name)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of nuclei data
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR,"stage1_train")
TRAIN_DATA_EXT_PATH = os.path.join(DATA_DIR,"external_processed_0329")
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

train_flag = True
train_head = True
train_all  = True

epoch_number_head = 12
epoch_number_all_fast = 6
epoch_number_all_slow = 8

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
        for iter_tmp in range(rep_id[groups[k]-1]):
            train_ids.append(ids[k])
    else:
        val_ids.append(ids[k])

for k in range(len(train_ids)):
    train_ids[k] = os.path.join(TRAIN_DATA_PATH,train_ids[k],'images',train_ids[k]+'.png')
train_ids_ext = glob.glob(TRAIN_DATA_EXT_PATH+'/*/images/*.png')
train_ids.extend(train_ids_ext)

for k,id_m in enumerate(train_ids_mosaic):
    train_ids_mosaic[k] = os.path.join(TRAIN_DATA_MOSAIC_PATH,id_m,'images',id_m+'_image.png')
train_ids.extend(train_ids_mosaic)

test_ids = next(os.walk(TEST_DATA_PATH))[1]
test_ids_m = next(os.walk(TEST_DATA_MOSAIC_PATH))[1]

print('train = '+str(len(train_ids))+' external = '+str(len(train_ids_ext)))

###########################################
# Training Config
###########################################

class TrainingConfig(Config):

    NAME = "nuclei_train"
    IMAGES_PER_GPU = 4
    GPU_COUNT = 1

    NUM_CLASSES = 1 + 1
    VALIDATION_STEPS = 50
    STEPS_PER_EPOCH = len(train_ids)/IMAGES_PER_GPU

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

    LEARNING_RATE = 0.0001

    AUGMENTATION = True
    RAND_SCALE_TRAIN = True
    SCALE_HIGH_INIT = 1.5
    RM_BOUND = True
    BACKBONE_NAME = 'resnet50'
    OPTIMIZER = 'adam'
    ADD_NOISE = False

config_head = TrainingConfig(512,256)
config_head.display()

class TrainingAllConfig(TrainingConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    STEPS_PER_EPOCH = len(train_ids)/IMAGES_PER_GPU
    
config_all = TrainingAllConfig(512,256)

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
# Prepare data
###########################################

random.seed(1234)
random.shuffle(train_ids)
random.shuffle(val_ids)

dataset_train = NucleiDataset()
dataset_train.add_class("cell", 1, "nulcei")
dataset_train.add_class("cell", -1, "boundary")
for k, train_id in enumerate(train_ids):
    dataset_train.add_image("cell", k, train_id)
dataset_train.prepare()

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
# Begin Training
###########################################

if train_flag:

    # Train the head branches
    if train_head:
        model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config_head)
        init_with = "coco"
        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        else:
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        # model_path = model.find_last()[1]
        # model.load_weights(model_path, by_name=True)
        epoch_init = epoch_number_head
        model.train(dataset_train, dataset_val, learning_rate=config_head.LEARNING_RATE, epochs=epoch_init, layers='heads')
        del model

    # Fine tune all layers
    if train_all:
        model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config_all)
        model_path = model.find_last()[1]
        # model_path = '/home/jieyang/code/TOOK18/nuclei_maskrcnn/logs/nuclei_train20180403T0043/mask_rcnn_nuclei_train_0015.h5'
        model_epoch = int(model_path.split('/')[-1].split('.')[0][-4:])
        model.load_weights(model_path, by_name=True)
        epoch_init_fast = model_epoch + epoch_number_all_fast
        epoch_init_slow = epoch_init_fast + epoch_number_all_slow
        model.train(dataset_train, dataset_val, learning_rate=config_all.LEARNING_RATE, epochs=epoch_init_fast,layers="all")
        model.train(dataset_train, dataset_val, learning_rate=config_all.LEARNING_RATE/10., epochs=epoch_init_slow, layers="all")
        del model