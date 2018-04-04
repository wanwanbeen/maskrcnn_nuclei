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
import model_resnet50_adam as modellib

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
TRAIN_DATA_EXT_PATH = os.path.join(DATA_DIR,"external_processed_0326")
TRAIN_DATA_AUG_PATH = os.path.join(DATA_DIR,"augmented_0326")
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
vsave_flag = False

epoch_number_init = 5
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
    IMAGE_MAX_DIM = 960

    VALIDATION_STEPS = 3
    STEPS_PER_EPOCH = 7368/IMAGES_PER_GPU

    RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    TRAIN_ROIS_PER_IMAGE = 256
    MAX_GT_INSTANCES = 400
    DETECTION_MAX_INSTANCES = 300
    RPN_NMS_THRESHOLD = 0.7
    IMAGE_PADDING = True
    # USE_MINI_MASK = False
    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    LEARNING_RATE = 0.0002


config_head = TrainingConfig()
config_head.display()

class TrainingAllConfig(TrainingConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    STEPS_PER_EPOCH = 7368 / IMAGES_PER_GPU
    
config_all = TrainingAllConfig()

###########################################
# Inference Config
###########################################

class InferenceConfig(TrainingConfig):

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

###########################################
# Load Nuclei Dataset
###########################################
def isboundary(mask):
    boundary_mask = np.zeros(mask.shape)
    boundary_mask[0, :] = 1
    boundary_mask[-1, :] = 1
    boundary_mask[:, 0] = 1
    boundary_mask[:, -1] = 1
    return np.any(boundary_mask * mask)

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
        mask_size=[]
        for k in range(num_inst):
            mask[:, :, k] = skimage.io.imread(os.path.join(mask_dir, mask_files[k]))
            mask_size.append(np.count_nonzero(mask[:,:,k]))

        mean_mask_size = sum(mask_size) / float(len(mask_size))
        for k in range(num_inst):
            if (isboundary(mask[:, :, k]) & mask_size<mean_mask_size*0.2) | mask_size<mean_mask_size*0.2:
                class_ids[k] = -1

        return mask, class_ids

###########################################
# TrainVal Split
###########################################

h = open(os.path.join(ROOT_DIR, "image_group_train.pickle"))
d = pickle.load(h)
train_ids = []
val_ids = []
for k in range(1,8): # range(1,7):
    train_id, val_id = train_test_split(d[k],train_size=0.8, random_state=1234)
    train_ids.extend(train_id)
    val_ids.extend(val_id)

train_ids_aug = []
for train_id in train_ids:
    train_ids_aug.extend(glob.glob(TRAIN_DATA_AUG_PATH+'/'+train_id+'*/images/*.png'))

for k in range(len(train_ids)):
    train_ids[k] = os.path.join(TRAIN_DATA_PATH,train_ids[k],'images',train_ids[k]+'.png')
train_ids_ext = glob.glob(TRAIN_DATA_EXT_PATH+'/*/images/*.png')

train_ids.extend(train_ids_aug)
train_ids.extend(train_ids_ext)

print(len(train_ids_ext),len(train_ids_aug),len(train_ids))

random.seed(1234)
random.shuffle(train_ids)
random.shuffle(val_ids)

test_ids = next(os.walk(TEST_DATA_PATH))[1]
dataset_train = NucleiDataset()
dataset_train.add_class("cell", 1, "nulcei")
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
        epoch_init = epoch_number_init
        epoch_add  = 0
        model.train(dataset_train, dataset_val, learning_rate=config_head.LEARNING_RATE, epochs=epoch_init, layers='heads')
        while epoch_add < epoch_number_iter:
            val_mAP.append(compute_mAP_val())
            if epoch_add >= 2 and val_mAP[-1] < val_mAP[-2] and val_mAP[-1] < val_mAP[-3]:
                break
            epoch_add += 1
            model.train(dataset_train, dataset_val, learning_rate=config_head.LEARNING_RATE, epochs=epoch_init+epoch_add, layers='heads')
        del model

    # Fine tune all layers
    if train_all:
        model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config_all)
        model_path = model.find_last()[1]
        model_epoch = int(model_path.split('/')[-1].split('.')[0][-4:])
        model.load_weights(model_path, by_name=True)
        val_mAP = []
        epoch_init = model_epoch + epoch_number_init
        epoch_add = 0
        model.train(dataset_train, dataset_val, learning_rate=config_all.LEARNING_RATE/10., epochs=epoch_init, layers="all")
        while epoch_add < epoch_number_iter:
            val_mAP.append(compute_mAP_val())
            if epoch_add >= 2 and val_mAP[-1] < val_mAP[-2] and val_mAP[-1] < val_mAP[-3]:
                break
            epoch_add += 1
            model.train(dataset_train, dataset_val, learning_rate=config_all.LEARNING_RATE/10., epochs=epoch_init+epoch_add, layers='all')
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

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('submission/sub-test-'+model_name+'.csv', index=False)
