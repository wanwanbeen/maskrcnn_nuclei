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

from config_nuclei import Config
import utils_nuclei as utils
import model_nuclei as modellib

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

vsave_flag = True
test_flag = False

model_path = '/home/jieyang/code/TOOK18/nuclei_maskrcnn/logs/nuclei_train20180403T0043/mask_rcnn_nuclei_train_0025.h5'

###########################################
# Train vs. validation split
###########################################

train_ids = []
val_ids = []
rep_id = [2,2,8,6,4,4,8]

df = pd.read_csv('image_group_train_pickle.csv')
ids = df['id']
groups = df['group']
istrain = df['istrain']
mosaic_ids = df['mosaic_id']
train_ids_mosaic = np.unique(mosaic_ids[istrain==1])[1:]
label_all = True

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

for dim_min in [256, 512, 1024]:
    for dim_max in [512, 1024, 2048]:

        foldername_suffix = '_' + str(dim_min) + '_' + str(dim_max) + '_mod_bound_all'

        ###########################################
        # Inference Config
        ###########################################
        class TrainingConfig(Config):
            NAME = "nuclei_train"
            IMAGES_PER_GPU = 1
            GPU_COUNT = 1

            NUM_CLASSES = 1 + 1
            IMAGE_MIN_DIM = 256
            IMAGE_MAX_DIM = 960

            VALIDATION_STEPS = 3
            STEPS_PER_EPOCH = 7368

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

            BACKBONE_NAME = 'resnet50'

        class InferenceConfig(TrainingConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

            IMAGE_MIN_DIM = dim_min
            IMAGE_MAX_DIM = dim_max
            POST_NMS_ROIS_INFERENCE = 1000

        inference_config = InferenceConfig()
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

        ###########################################
        # Validation Setting
        ###########################################

        def compute_mAP_val(model_path):
            model_inf = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
            assert model_path != "", "Provide path to trained weights"
            print("Loading weights from ", model_path)
            model_inf.load_weights(model_path, by_name=True)
            model_name = model_path.split('/')[-2]
            model_epoch = model_path.split('/')[-1].split('.')[0][-5:]
            model_name = model_name + model_epoch
            model_name = model_name + foldername_suffix

            if vsave_flag and not os.path.exists(TEST_VAL_MASK_SAVE_PATH + '/' + model_name):
                os.makedirs(TEST_VAL_MASK_SAVE_PATH + '/' + model_name)

            APs = []
            for image_id in dataset_val.image_ids:
                image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                    modellib.load_image_gt_noresize(dataset_val, inference_config, image_id, use_mini_mask=False)
                results = model_inf.detect([image], verbose=0)
                r = results[0]
                masks = r["masks"]
                tmp = masks[:, :, 0].copy()
                if tmp.shape[0] == 0:
                    masks = np.zeros(image.shape)
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

                if vsave_flag:
                    train_id = dataset_val.image_info[image_id]['path'].split('/')[-1][:-4]
                    rmaskcollapse_gt = np.zeros((image.shape[0], image.shape[1]))
                    for i in range(gt_mask.shape[2]):
                        rmaskcollapse_gt = rmaskcollapse_gt + gt_mask[:, :, i] * (i + 1)

                    masks = r["masks"]
                    rmaskcollapse = np.zeros((image.shape[0], image.shape[1]))
                    for i in range(masks.shape[2]):
                        rmaskcollapse = rmaskcollapse + masks[:, :, i] * (i + 1)

                    tmp1 = rmaskcollapse_gt.copy()
                    tmp1[rmaskcollapse_gt > 0] = 1
                    tmp2 = rmaskcollapse.copy()
                    tmp2[rmaskcollapse > 0] = 2
                    overlap = tmp1+tmp2

                    skimage.io.imsave(
                        TEST_VAL_MASK_SAVE_PATH + '/' + model_name + '/ap_' + '%.2f' % AP + '_' + train_id + '_mask.png',
                        np.concatenate((label2rgb(rmaskcollapse_gt, bg_label=0), image / 255.,
                                        label2rgb(rmaskcollapse, bg_label=0),
                                        label2rgb(overlap, bg_label=0)), axis=1))

                    np.save(TEST_VAL_MASK_SAVE_PATH + '/' + model_name + '/ap_' + '%.2f' % AP + '_' + train_id + '_mask.npy', rmaskcollapse)

                    np.save(TEST_VAL_MASK_SAVE_PATH + '/' + model_name + '/ap_' + '%.2f' % AP + '_' + train_id + '_gtmask.npy', rmaskcollapse_gt)

            print("mAP: ", np.mean(APs))
            del model_inf
            return np.mean(APs)

        compute_mAP_val(model_path)

        ###########################################
        # Begin Testing
        ###########################################

        if test_flag:
            model_inf = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
            assert model_path != "", "Provide path to trained weights"
            print("Loading weights from ", model_path)
            model_inf.load_weights(model_path, by_name=True)
            model_name = model_path.split('/')[-2]
            model_epoch = model_path.split('/')[-1].split('.')[0][-5:]
            model_name = model_name + model_epoch
            model_name = model_name + foldername_suffix
            new_test_ids = []
            rles = []
            if not os.path.exists(TEST_MASK_SAVE_PATH + '/' + model_name):
                os.makedirs(TEST_MASK_SAVE_PATH + '/' + model_name)

            for image_id in dataset_test.image_ids:
                dataset_test.load_image(image_id)
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
