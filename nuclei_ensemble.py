__authors__="Jie Yang and Xinyang Feng"

###########################################
# ensemble
###########################################
import glob
import skimage.io
from networkx.algorithms.components.connected import connected_components
import os
import numpy as np
from skimage.color import label2rgb
import pandas as pd
from nuclei_utils import deoverlap_masks, prob_to_rles, to_graph, compute_overlaps_masks, sweep_iou_mask_ap

TEST_IMAGE_PATH = '~/data/stage1_test_image/'
TRAIN_IMAGE_PATH = '~/nuclei_maskrcnn/data/stage1_train_image/'
TEST_MASK_SAVE_PATH = '~/nuclei_maskrcnn/data/stage1_masks_test/'
VAL_MASK_SAVE_PATH = '~/nuclei_maskrcnn/data/stage1_masks_val/'
VAL_MASK_ENSEMBLE_SAVE_PATH = '~/nuclei_maskrcnn/data/stage1_masks_val_ensemble/'
TEST_MASK_ENSEMBLE_SAVE_PATH = '~/nuclei_maskrcnn/data/stage1_masks_test_ensemble/'

def ensemble_func(ensemble_dirs = None, iou_threshold = 0.5, model_name = '', test_flag=True):
    if test_flag:
        MASK_ENSEMBLE_SAVE_PATH = TEST_MASK_ENSEMBLE_SAVE_PATH
    else:
        MASK_ENSEMBLE_SAVE_PATH = VAL_MASK_ENSEMBLE_SAVE_PATH

    MASK_ENSEMBLE_SAVE_PATH = os.path.join(MASK_ENSEMBLE_SAVE_PATH,model_name)
    if not os.path.exists(MASK_ENSEMBLE_SAVE_PATH):
        os.mkdir(MASK_ENSEMBLE_SAVE_PATH)

    test_ids = glob.glob(ensemble_dirs[0]+'*_mask.npy')
    for i in range(len(test_ids)):
        test_ids[i]=test_ids[i].split('/')[-1].split('_mask')[0]

    num_models = len(ensemble_dirs)
    rles = []
    new_test_ids = []
    APs = []
    for test_id in test_ids:
        print test_id
        detection_results = []
        for ensemble_dir in ensemble_dirs:
            detection_results.append(np.load(os.path.join(ensemble_dir,test_id+'_mask.npy')))

        addon = []
        for i in range(1,len(detection_results)):
            addon.append(np.max(detection_results[i - 1]))
            detection_results[i][np.where(detection_results[i]>0)] = detection_results[i][np.where(detection_results[i]>0)]+addon[i-1]

        masks = np.zeros((detection_results[0].shape[0], detection_results[0].shape[1], len(np.unique(detection_results))-1))

        count_inst_thresh = []
        count_inst = 0
        for k in range(len(detection_results)):
            for val in (np.unique(detection_results[k])[1:]):
                masks[:, :, count_inst] = detection_results[k] == val
                count_inst += 1
            count_inst_thresh.append(count_inst)

        overlaps = compute_overlaps_masks(masks, masks)

        overlaps_match = overlaps>iou_threshold

        node_lists = []
        for k in range(overlaps_match.shape[0]):
            node_lists.append(np.where(overlaps_match[k,:])[0])

        G = to_graph(node_lists)
        merge_lists = [list(c) for c in connected_components(G)]

        merge_lists = [merge_list for merge_list in merge_lists if len(merge_list) >= np.ceil(num_models*0.5)]

        masks_new = np.zeros((masks.shape[0],masks.shape[1],len(merge_lists))).astype(np.float32)
        for i, nodes in enumerate(merge_lists):
            for node in nodes:
                masks_new[:,:,i] += masks[:,:,node]
            masks_new[:,:,i] /= len(nodes)
        masks_new = masks_new>iou_threshold
        masks_new = deoverlap_masks(masks_new)

        masks_new_c = np.zeros((masks_new.shape[0],masks_new.shape[1]))
        for i in range(masks_new.shape[2]):
            masks_new_c += masks_new[:,:,i]*(i+1)

        all_masks = label2rgb(masks_new_c, bg_label=0)

        for d in (detection_results):
            all_masks = np.concatenate((all_masks, label2rgb(d, bg_label=0)),axis=1)

        if test_flag:
            image = skimage.io.imread(TEST_IMAGE_PATH + test_id + '.png')[:,:,:3]
        else:
            image = skimage.io.imread(TRAIN_IMAGE_PATH + test_id + '.png')[:, :, :3]

        image0 = image.copy()
        print image0.shape, all_masks.shape
        skimage.io.imsave(
            MASK_ENSEMBLE_SAVE_PATH +'/'+ test_id + '_masks.png',
            np.concatenate((image0 / 255.,all_masks), axis=1))

        tmp = masks_new[:, :, 0].copy()
        if tmp.shape[0] == 0:
            masks_new = np.zeros(image.shape)
        for i in range(masks_new.shape[2]):
            tmp = masks_new[:, :, i].copy()
            tmp[0, :] = tmp[1, :]
            tmp[-1, :] = tmp[-2, :]
            tmp[:, 0] = tmp[:, 1]
            tmp[:, -1] = tmp[:, -2]
            masks_new[:, :, i] = tmp.copy()
        masks_new = masks_new.copy()

        tmp = image[:, :, 0].copy()
        tmp[masks_new_c > 0] = 250
        image[:, :, 0] = tmp.copy()
        tmp = image[:, :, 1].copy()
        tmp[masks_new_c > 0] = 200
        image[:, :, 1] = tmp.copy()
        tmp = image[:, :, 2].copy()
        tmp[masks_new_c > 0] = 80
        image[:, :, 2] = tmp.copy()

        skimage.io.imsave(MASK_ENSEMBLE_SAVE_PATH + '/plot2_' + test_id + '_mask.png',
                          np.concatenate((image0 / 255.,
                                          label2rgb(masks_new_c, bg_label=0), image / 255.), axis=1))
        np.save(MASK_ENSEMBLE_SAVE_PATH + '/' + test_id + '_mask.npy', masks_new_c)
        for i in range(masks_new.shape[2]):
            rle = list(prob_to_rles(masks_new[:, :, i]))
            rles.extend(rle)
            new_test_ids.append(test_id)

        if not test_flag:
            gt_masks_c = np.load(os.path.join(ensemble_dirs[0], test_id + '_gtmask.npy'))
            gt_unique_objects = np.unique(gt_masks_c)[1:]
            gt_masks = np.zeros((gt_masks_c.shape[0],gt_masks_c.shape[1],len(gt_unique_objects)))
            for i, val in enumerate(gt_unique_objects):
                gt_masks[:, :, i] = gt_masks_c == val

            AP = sweep_iou_mask_ap(gt_masks, masks_new, np.ones((masks_new.shape[2])))
            APs.append(AP)
            print np.mean(APs)

    if test_flag:
        # Create submission DataFrame
        sub = pd.DataFrame()
        sub['ID'] = new_test_ids
        sub['RLE'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        sub.to_csv('RLE-test-'+model_name+'_ensemble.csv', index=False)
	
###########################################
# main
###########################################

test_flag = False

if test_flag:
    ensemble_input_dir = TEST_MASK_SAVE_PATH
else:
    ensemble_input_dir = VAL_MASK_SAVE_PATH

model_name = 'nuclei_train20180000T0000_0000'
ensemble_dirs = [ensemble_input_dir + '/' + model_name + '/',
                 ensemble_input_dir + '/' + model_name + '_vflip/',
                 ensemble_input_dir + '/' + model_name + '_hflip/']

model_name = 'ensemble'
print ensemble_dirs
ensemble_func(ensemble_dirs=ensemble_dirs, model_name=model_name, test_flag=test_flag)