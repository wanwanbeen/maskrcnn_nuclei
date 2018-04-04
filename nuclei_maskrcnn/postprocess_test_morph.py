# test morphological operations
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_fill_holes
import glob
import numpy as np
import pandas as pd
import os
import skimage.io
from skimage.color import gray2rgb, label2rgb
from skimage.measure import label
import utils

VALIDATION_DATA_PATH = '/home/jieyang/code/TOOK18/nuclei_maskrcnn/data/stage1_masks_val/pretrain_nuclei_train20180318T1606_0016/'
val_data = glob.glob(VALIDATION_DATA_PATH+'*_mask.npy')

TEST_DATA_PATH = '/home/jieyang/code/TOOK18/nuclei_maskrcnn/data/stage1_masks_test/pretrain_nuclei_train20180318T1606_0016/'
test_data = glob.glob(TEST_DATA_PATH+'*.npy')

TEST_MASK_SAVE_PATH = '/home/jieyang/code/TOOK18/nuclei_maskrcnn/data/test_morph/test/'
VAL_MASK_SAVE_PATH = '/home/jieyang/code/TOOK18/nuclei_maskrcnn/data/test_morph/val/'

operation_string = 'closing'

if not os.path.exists(VAL_MASK_SAVE_PATH + '/' + operation_string):
    os.makedirs(VAL_MASK_SAVE_PATH + '/' + operation_string)
if not os.path.exists(TEST_MASK_SAVE_PATH + '/' + operation_string):
    os.makedirs(TEST_MASK_SAVE_PATH + '/' + operation_string)

def getLargestCC(segmentation):
    labels = label((segmentation).astype(np.int8))
    largestCC = labels == np.argmax(np.bincount(labels.flat))+1
    return largestCC

def morph_operation(segmentation):
    return binary_closing(segmentation)
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
APs = []
for vd in val_data:
    rmaskcollapse = np.load(vd)
    uids = np.unique(rmaskcollapse)[1:]
    masks_new = np.zeros((rmaskcollapse.shape[0], rmaskcollapse.shape[1], len(uids)))
    for i, uid in enumerate(uids):
        mask = rmaskcollapse == uid
        mask = morph_operation(getLargestCC(binary_fill_holes(mask)))
        masks_new[:, :, i] = mask

    rmaskcollapse_new = np.zeros(rmaskcollapse.shape)
    for i in range(masks_new.shape[2]):
        rmaskcollapse_new = rmaskcollapse_new + masks_new[:, :, i] * (i + 1)

    val_id = vd.split('/')[-1]
    val_id = val_id.split('_mask.npy')[0]
    val_id = val_id.split('_')[-1]

    rmaskcollapse_gt = np.load(vd.replace('mask.npy','gtmask.npy'))
    uids_gt = np.unique(rmaskcollapse_gt)[1:]
    gt_masks = np.zeros((rmaskcollapse_gt.shape[0], rmaskcollapse_gt.shape[1], len(uids_gt)))
    for i, uid in enumerate(uids_gt):
        mask = rmaskcollapse_gt == uid
        gt_masks[:, :, i] = mask

    AP = utils.sweep_iou_mask_ap(gt_masks, masks_new, np.ones((gt_masks.shape[2])))

    skimage.io.imsave(VAL_MASK_SAVE_PATH + '/' + operation_string + '/ap_' + '%.2f' % AP +'_'+ val_id + '_mask.png',
                                      label2rgb(rmaskcollapse_new, bg_label=0))
    np.save(VAL_MASK_SAVE_PATH + '/' + operation_string + '/ap_' + '%.2f' % AP + '_' + val_id + '_mask.npy', rmaskcollapse_new)

    APs.append(AP)
    print np.mean(APs)

print("mAP: ", np.mean(APs))

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
new_test_ids = []
rles = []
for td in test_data:
    rmaskcollapse = np.load(td)
    uids = np.unique(rmaskcollapse)[1:]
    masks_new = np.zeros((rmaskcollapse.shape[0],rmaskcollapse.shape[1],len(uids)))
    for i, uid in enumerate(uids):
        mask = rmaskcollapse==uid
        mask = morph_operation(getLargestCC(binary_fill_holes(mask)))
        masks_new[:,:,i] = mask

    rmaskcollapse_new = np.zeros(rmaskcollapse.shape)
    for i in range(masks_new.shape[2]):
        rmaskcollapse_new = rmaskcollapse_new + masks_new[:, :, i] * (i + 1)

    test_id = td.split('/')[-1]
    test_id = test_id.split('_mask.npy')[0]

    skimage.io.imsave(TEST_MASK_SAVE_PATH + '/' + operation_string + '/' + test_id + '_mask.png',
                                      label2rgb(rmaskcollapse_new, bg_label=0))
    np.save(TEST_MASK_SAVE_PATH + '/' + operation_string + '/' + test_id + '_mask.npy',rmaskcollapse_new)
    for i in range(masks_new.shape[2]):
        rle = list(utils.prob_to_rles(masks_new[:, :, i]))
        rles.extend(rle)
        new_test_ids.append(test_id)

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv(TEST_MASK_SAVE_PATH + '/' + operation_string + '.csv', index=False)
