###########################################
# Library and Path
###########################################
import sys
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
import gc

from config_var import Config
import utils
import model_nop6 as modellib

GPU_option = 1
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
TRAIN_DATA_EXT_PATH = os.path.join(DATA_DIR,"external_processed_crop")
TRAIN_DATA_AUG_PATH = os.path.join(DATA_DIR,"stage1_train_crop")
TEST_DATA_PATH = os.path.join(DATA_DIR,"stage1_test")
TEST_MASK_SAVE_PATH = os.path.join(DATA_DIR,"stage1_masks_test")
TEST_VAL_MASK_SAVE_PATH = os.path.join(DATA_DIR,"stage1_masks_val")

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_option)

import tensorflow as tf
import keras.backend as K
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
session = tf.Session(config=config_tf)
train_flag = True
train_head = True
train_head_init = True
train_all  = True
train_all_init = True
vsave_flag = False

epoch_number_init = 10
epoch_number_iter = 0

###########################################
# Training Config
###########################################

class TrainingConfig(Config):

    NAME = "nuclei_train"
    IMAGES_PER_GPU = 4
    GPU_COUNT = 1

    USE_MINI_MASK = True

    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 128

    VALIDATION_STEPS = 3
    STEPS_PER_EPOCH = 2000 # 13448 # 1898*2

    RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    TRAIN_ROIS_PER_IMAGE = 256
    MAX_GT_INSTANCES = 200
    DETECTION_MAX_INSTANCES = 200
    RPN_NMS_THRESHOLD = 0.5
    IMAGE_PADDING = True

    POST_NMS_ROIS_TRAINING = 2000
    BACKBONE_STRIDES = [4, 8, 16, 32]

    LEARNING_RATE = 0.005

class TrainingConfig_all(TrainingConfig):
    IMAGES_PER_GPU = 2
    STEPS_PER_EPOCH = 4000

config = TrainingConfig(256)
config.display()
config_all = TrainingConfig_all(256)

###########################################
# Inference Config
###########################################

class InferenceConfig(TrainingConfig):
    IMAGE_MIN_DIM = 256
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    POST_NMS_ROIS_INFERENCE = 1000
    MAX_GT_INSTANCES = 400
    DETECTION_MAX_INSTANCES = 300

inference_config = InferenceConfig(256)
inference_config.display()

###########################################
# Load Nuclei Dataset
###########################################

class NucleiDataset_train(utils.Dataset):
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
            if mask_files[k].split('.')[-2][-1]=='b':
                class_ids[k] = -1
        return mask, class_ids

class NucleiDataset_val(NucleiDataset_train):
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
# TrainVal Split
###########################################

h = open(os.path.join(ROOT_DIR, "image_group.pickle"))
d = pickle.load(h)
train_ids = []
val_ids = []
for k in range(1,7):
    train_id, val_id = train_test_split(d[k],train_size=0.8, random_state=1234)
    train_ids.extend(train_id)
    val_ids.extend(val_id)

train_ids_aug = []
for train_id in train_ids:
    train_ids_aug.extend(glob.glob(TRAIN_DATA_AUG_PATH+'/'+train_id+'*/images/*.png'))

train_ids = [] # the original files cropped are also in the aug folder, so [] here
# for k in range(len(train_ids)):
#     train_ids[k] = os.path.join(TRAIN_DATA_PATH,train_ids[k],'images',train_ids[k]+'.png')
train_ids_ext = glob.glob(TRAIN_DATA_EXT_PATH+'/*/images/*.png')

train_ids.extend(train_ids_aug)
train_ids.extend(train_ids_ext)

print(len(train_ids_ext),len(train_ids_aug),len(train_ids))

random.seed(1234)
random.shuffle(train_ids)
random.shuffle(val_ids)

test_ids = next(os.walk(TEST_DATA_PATH))[1]
dataset_train = NucleiDataset_val()
dataset_train.add_class("cell", 1, "nulcei")
# dataset_train.add_class("cell", -1, "boundary")
for k, train_id in enumerate(train_ids):
    dataset_train.add_image("cell", k, train_id)
dataset_train.prepare()

dataset_val = NucleiDataset_val()
dataset_val.add_class("cell", 1, "nulcei")
for k, val_id in enumerate(val_ids):
    dataset_val.add_image("cell", k, os.path.join(TRAIN_DATA_PATH, val_id, 'images', val_id + '.png'))
dataset_val.prepare()

dataset_test = NucleiDataset_val()
dataset_test.add_class("cell", 1, "nulcei")
for k, test_id in enumerate(test_ids):
    dataset_test.add_image("cell", k, os.path.join(TEST_DATA_PATH, test_id, 'images', test_id + '.png'))
dataset_test.prepare()

###########################################
# Validation Setting
###########################################

def compute_mAP_val(maxiap, use_last=True):
    model_inf0 = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
    if use_last:
        model_path = model_inf0.find_last()[1]
    else:
        model_path = model_inf0.find_2nd2_last()[1]
    assert model_path != "", "Provide path to trained weights"
    model_name = model_path.split('/')[-2]
    model_epoch = model_path.split('/')[-1].split('.')[0][-5:]
    model_name = model_name + model_epoch
    del model_inf0

    if vsave_flag and not os.path.exists(TEST_VAL_MASK_SAVE_PATH + '/' + model_name):
        os.makedirs(TEST_VAL_MASK_SAVE_PATH + '/' + model_name)

    APs = []
    iAP = 0
    for image_id in dataset_val.image_ids:
        img_val = dataset_val.load_image(image_id)
        img_max_dim = max(img_val.shape)
        img_max_dim = min(np.int(np.ceil(img_max_dim / 32.) * 32.), 1024)
        model_inf = modellib.MaskRCNN(mode="inference", config=InferenceConfig(img_max_dim), model_dir=MODEL_DIR)
        model_inf.load_weights(model_path, by_name=True)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt_noresize(dataset_val, InferenceConfig(img_max_dim), image_id, use_mini_mask=False)
        results = model_inf.detect([image], verbose=0)
        r = results[0]
        AP = utils.sweep_iou_mask_ap(gt_mask, r["masks"], r["scores"])
        APs.append(AP)
        print np.mean(APs)

        if vsave_flag:
            train_id = dataset_val.image_info[image_id]['path'].split('/')[-1][:-4]
            rmaskcollapse_gt = np.zeros((image.shape[0], image.shape[1]))
            for i in range(gt_mask.shape[2]):
                rmaskcollapse_gt = rmaskcollapse_gt + gt_mask[:, :, i] * (i + 1)
            rmaskcollapse_gt = label2rgb(rmaskcollapse_gt, bg_label=0)
            masks = r["masks"]
            rmaskcollapse = np.zeros((image.shape[0], image.shape[1]))
            for i in range(masks.shape[2]):
                rmaskcollapse = rmaskcollapse + masks[:, :, i] * (i + 1)
            rmaskcollapse = label2rgb(rmaskcollapse, bg_label=0)

            skimage.io.imsave(
                TEST_VAL_MASK_SAVE_PATH + '/' + model_name + '/ap_' + '%.2f' % AP + '_' + train_id + '_mask.png',
                np.concatenate((rmaskcollapse_gt, image / 255., rmaskcollapse), axis=1))
        del model_inf
        for i in range(10): gc.collect()
        K.clear_session()

        iAP += 1
        if iAP > maxiap:
            break

    print("mAP: ", np.mean(APs))
    return np.mean(APs)

###########################################
# Begin Training
###########################################

if train_flag:

    # Train the head branches
    if train_head:
        model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)
        init_with = "coco"
        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        else:
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        if train_head_init:
            model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epoch_number_init, layers='heads')
            del model
            for i in range(10): gc.collect()
            K.clear_session()

        val_mAP = []
        epoch_add  = 0
        while epoch_add < epoch_number_iter:
            val_mAP.append(compute_mAP_val(5))
            if epoch_add >= 2 and val_mAP[-1] <= val_mAP[-2] and val_mAP[-1] <= val_mAP[-3]:
                break
            model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)
            model_path = model.find_last()[1]
            assert model_path != "", "Provide path to trained weights"
            model_epoch = int(model_path.split('/')[-1].split('.')[0][-4:])
            model.load_weights(model_path, by_name=True)
            lr_new = config.LEARNING_RATE / (1. + 2. * epoch_add)
            model.train(dataset_train, dataset_val, learning_rate=lr_new, epochs=model_epoch+2, layers='heads')
            epoch_add += 1
            del model
            for i in range(10): gc.collect()
            K.clear_session()

    # Fine tune all layers
    if train_all:

        if train_all_init:
            model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config_all)
            model_path = model.find_2nd2_last()[1]
            assert model_path != "", "Provide path to trained weights"
            model_epoch = int(model_path.split('/')[-1].split('.')[0][-4:])
            model.load_weights(model_path, by_name=True)
            model.train(dataset_train, dataset_val, learning_rate=config_all.LEARNING_RATE, epochs=model_epoch+epoch_number_init, layers='all')
            del model
            for i in range(10): gc.collect()
            K.clear_session()

        val_mAP = []
        epoch_add = 0
        while epoch_add < epoch_number_iter:
            val_mAP.append(compute_mAP_val(5))
            if epoch_add >= 2 and val_mAP[-1] <= val_mAP[-2] and val_mAP[-1] <= val_mAP[-3]:
                break
            model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config_all)
            model_path = model.find_last()[1]
            assert model_path != "", "Provide path to trained weights"
            model_epoch = int(model_path.split('/')[-1].split('.')[0][-4:])
            model.load_weights(model_path, by_name=True)
            lr_new = config.LEARNING_RATE / (1. + 2. * epoch_add)
            model.train(dataset_train, dataset_val, learning_rate=lr_new, epochs=model_epoch+2, layers='all')
            epoch_add += 1
            del model
            for i in range(10): gc.collect()
            K.clear_session()

# compute_mAP_val(500, use_last=False)
# sys.exit()

###########################################
# Begin Testing
###########################################
model_inf0 = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
model_path = model_inf0.find_2nd2_last()[1]
assert model_path != "", "Provide path to trained weights"
model_name = model_path.split('/')[-2]
model_epoch = model_path.split('/')[-1].split('.')[0][-5:]
model_name = model_name + model_epoch

if not os.path.exists(TEST_MASK_SAVE_PATH + '/' + model_name):
    os.makedirs(TEST_MASK_SAVE_PATH + '/' + model_name)

new_test_ids = []
rles = []

for image_id in dataset_test.image_ids:
    img_test = dataset_test.load_image(image_id)
    img_max_dim = max(img_test.shape)
    img_max_dim = np.int(np.ceil(img_max_dim*1./32.)*32.)
    model_inf = modellib.MaskRCNN(mode="inference", config=InferenceConfig(img_max_dim), model_dir=MODEL_DIR)
    # model_inf.keras_model.summary()
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model_inf.load_weights(model_path, by_name=True)

    results = model_inf.detect([img_test], verbose=0)
    del model_inf
    for i in range(10): gc.collect()
    K.clear_session()

    r = results[0]
    test_id = dataset_test.image_info[image_id]['path'].split('/')[-1][:-4]

    rmaskcollapse = np.zeros((r['masks'].shape[0], r['masks'].shape[1]))
    for i in range(r['masks'].shape[2]):
        rmaskcollapse = rmaskcollapse + r['masks'][:, :, i] * (i + 1)

    skimage.io.imsave(TEST_MASK_SAVE_PATH + '/' + model_name + '/' + test_id + '_mask.png',
                      np.concatenate((dataset_test.load_image(image_id) / 255.,
                                      label2rgb(rmaskcollapse, bg_label=0)), axis=1))

    for i in range(r['masks'].shape[2]):
        rle = list(utils.prob_to_rles(r['masks'][:, :, i]))
        rles.extend(rle)
        new_test_ids.append(test_id)

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('submission/sub-test-'+model_name+'.csv', index=False)
