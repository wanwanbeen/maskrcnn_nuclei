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

from config import Config
import utils
import model_resnet50_aug_rmb as modellib

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
train_head = False
train_all  = True
vsave_flag = False

epoch_number_init_head = 10
epoch_number_init_4plus = 10
epoch_number_init_all = 30
epoch_number_iter = 0

###########################################
# Training Config
###########################################

class TrainingConfig(Config):

    NAME = "nuclei_train"
    IMAGES_PER_GPU = 4
    GPU_COUNT = 1

    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512

    VALIDATION_STEPS = 3
    STEPS_PER_EPOCH = (1544+228)/IMAGES_PER_GPU

    RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    TRAIN_ROIS_PER_IMAGE = 256
    MAX_GT_INSTANCES = 500
    DETECTION_MAX_INSTANCES = 500
    RPN_NMS_THRESHOLD = 0.7
    IMAGE_PADDING = True
    # USE_MINI_MASK = False
    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    LEARNING_RATE = 0.0001

config_head = TrainingConfig()
config_head.display()

class TrainingAllConfig(TrainingConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = (1544+228)/IMAGES_PER_GPU
    
config_all = TrainingAllConfig()

###########################################
# Inference Config
###########################################

class InferenceConfig(TrainingConfig):
    DETECTION_MAX_INSTANCES = 1000
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

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
# TrainVal Split
###########################################

h = open(os.path.join(ROOT_DIR, "image_group_train.pickle"))
d = pickle.load(h)
train_ids = []
val_ids = []
rep_id = [2,2,8,6,4,4,8]
for k in range(1,8):
    train_id, val_id = train_test_split(d[k],train_size=0.8, random_state=1234)
    for iter_tmp in range(rep_id[k-1]):
        train_ids.extend(train_id)
    val_ids.extend(val_id)

for k in range(len(train_ids)):
    train_ids[k] = os.path.join(TRAIN_DATA_PATH,train_ids[k],'images',train_ids[k]+'.png')
train_ids_ext = glob.glob(TRAIN_DATA_EXT_PATH+'/*/images/*.png')

train_ids.extend(train_ids_ext)

test_ids = next(os.walk(TEST_DATA_PATH))[1]
test_ids_m = next(os.walk(TEST_DATA_MOSAIC_PATH))[1]

print(len(train_ids),len(train_ids_ext))

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
# Validation Setting
###########################################

def compute_mAP_val():
    model_inf = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
    model_path = model_inf.find_last()[1]
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model_inf.load_weights(model_path, by_name=True)
    model_name = model_path.split('/')[-2]
    model_epoch = model_path.split('/')[-1].split('.')[0][-5:]
    model_name = model_name + model_epoch

    if vsave_flag and not os.path.exists(TEST_VAL_MASK_SAVE_PATH + '/' + model_name):
        os.makedirs(TEST_VAL_MASK_SAVE_PATH + '/' + model_name)

    APs = []
    for image_id in dataset_val.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt_noresize(dataset_val, inference_config, image_id, use_mini_mask=False)
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
    print("mAP: ", np.mean(APs))
    del model_inf
    return np.mean(APs)

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
        val_mAP = []
        epoch_init = epoch_number_init_head
        epoch_add  = 0
        model.train(dataset_train, dataset_val, learning_rate=config_head.LEARNING_RATE, epochs=epoch_init, layers='heads')
        del model

    # Fine tune all layers
    if train_all:
        model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config_all)
        model_path = model.find_last()[1]
        model_epoch = int(model_path.split('/')[-1].split('.')[0][-4:])
        model.load_weights(model_path, by_name=True)
        val_mAP = []
        epoch_init = model_epoch + epoch_number_init_4plus
        epoch_add = 0
        # model.train(dataset_train, dataset_val, learning_rate=config_all.LEARNING_RATE, epochs=epoch_init, layers="4+")
        model.train(dataset_train, dataset_val, learning_rate=config_all.LEARNING_RATE/10., epochs=epoch_number_init_head+epoch_number_init_4plus+epoch_number_init_all, layers="all")
        del model

compute_mAP_val()

###########################################
# Begin Testing
###########################################

model_inf = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
# os.remove(model_inf.find_last()[1])
model_path = model_inf.find_last()[1]
# model_path = '/home/jieyang/code/TOOK18/nuclei_maskrcnn/logs/nuclei_train20180211T2323/mask_rcnn_nuclei_train_0019.h5'
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model_inf.load_weights(model_path, by_name=True)
model_name = model_path.split('/')[-2]
model_epoch = model_path.split('/')[-1].split('.')[0][-5:]
model_name = model_name + model_epoch

new_test_ids = []
rles = []
if not os.path.exists(TEST_MASK_SAVE_PATH + '/' + model_name):
    os.makedirs(TEST_MASK_SAVE_PATH + '/' + model_name)

for image_id in dataset_test.image_ids:

    results = model_inf.detect([dataset_test.load_image(image_id)], verbose=0)
    r = results[0]
    test_id = dataset_test.image_info[image_id]['path'].split('/')[-1][:-4]

    rmaskcollapse = np.zeros((r['masks'].shape[0], r['masks'].shape[1]))
    for i in range(r['masks'].shape[2]):
        rmaskcollapse = rmaskcollapse + r['masks'][:, :, i] * (i + 1)

    skimage.io.imsave(TEST_MASK_SAVE_PATH + '/' + model_name + '/' + test_id + '_mask.png',
                      np.concatenate((dataset_test.load_image(image_id) / 255.,
                                      label2rgb(rmaskcollapse, bg_label=0)), axis=1))
    np.save(TEST_MASK_SAVE_PATH + '/' + model_name + '/' + test_id + '_mask.npy',rmaskcollapse)
    for i in range(r['masks'].shape[2]):
        rle = list(utils.prob_to_rles(r['masks'][:, :, i]))
        rles.extend(rle)
        new_test_ids.append(test_id)

new_test_ids_m = []
rles_m = []
for image_id in dataset_test_m.image_ids:

    results = model_inf.detect([dataset_test_m.load_image(image_id)], verbose=0)
    r = results[0]
    test_id_m = dataset_test_m.image_info[image_id]['path'].split('/')[-1][:-4]

    rmaskcollapse = np.zeros((r['masks'].shape[0], r['masks'].shape[1]))
    for i in range(r['masks'].shape[2]):
        rmaskcollapse = rmaskcollapse + r['masks'][:, :, i] * (i + 1)

    skimage.io.imsave(TEST_MASK_SAVE_PATH + '/' + model_name + '/' + test_id_m + '_mask.png',
                      np.concatenate((dataset_test_m.load_image(image_id) / 255.,
                                      label2rgb(rmaskcollapse, bg_label=0)), axis=1))
    np.save(TEST_MASK_SAVE_PATH + '/' + model_name + '/' + test_id_m + '_mask.npy',rmaskcollapse)

    ul_shape = np.load(dataset_test_m.image_info[image_id]['path'].replace('images/','').replace('_image.png','_ul_shappe.npy'))[0:2]

    f = open(dataset_test_m.image_info[image_id]['path'].replace('images/','').replace('_image.png','_list.txt'),'r')
    ids = f.read().split('\n')
    # print ids
    
    for i in range(r['masks'].shape[2]):
        if np.any(r['masks'][:ul_shape[0], :ul_shape[1], i]):
            rle = list(utils.prob_to_rles(r['masks'][:ul_shape[0], :ul_shape[1], i]))
            rles_m.extend(rle)
            new_test_ids_m.append(ids[0].split('/')[-3])

    for i in range(r['masks'].shape[2]):
        if np.any(r['masks'][:ul_shape[0], ul_shape[1]:, i]):
            rle = list(utils.prob_to_rles(r['masks'][:ul_shape[0], ul_shape[1]:, i]))
            rles_m.extend(rle)
            new_test_ids_m.append(ids[1].split('/')[-3])

    for i in range(r['masks'].shape[2]):
        if np.any(r['masks'][:ul_shape[0], ul_shape[1]:, i]):
            rle = list(utils.prob_to_rles(r['masks'][ul_shape[0]:, :ul_shape[1], i]))
            rles_m.extend(rle)
            new_test_ids_m.append(ids[2].split('/')[-3])

    for i in range(r['masks'].shape[2]):
        if np.any(r['masks'][ul_shape[0]:, ul_shape[1]:, i]):
            rle = list(utils.prob_to_rles(r['masks'][ul_shape[0]:, ul_shape[1]:, i]))
            rles_m.extend(rle)
            new_test_ids_m.append(ids[3].split('/')[-3])

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('submission/sub-test-'+model_name+'.csv', index=False)

rles = [rle for i, rle in enumerate(rles) if new_test_ids[i] not in ids] 
new_test_ids = [new_test_id for new_test_id in new_test_ids if new_test_id not in ids] 
rles = rles.extend(rles_m)
new_test_ids.extend(new_test_ids_m)

sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('submission/sub-test-mosaic-'+model_name+'.csv', index=False)


