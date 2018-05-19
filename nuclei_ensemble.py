__authors__="Jie Yang and Xinyang Feng"

import glob
import skimage.io
import argparse
from networkx.algorithms.components.connected import connected_components
import os
import numpy as np
from skimage.color import label2rgb
import pandas as pd
from nuclei_utils import deoverlap_masks, prob_to_rles, to_graph, compute_overlaps_masks, sweep_iou_mask_ap

def ensemble_func(ensemble_dirs = None, MASK_ENSEMBLE_SAVE_PATH ='', IMAGE_PATH = '', test_flag = True, iou_threshold = 0.5):

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

        image = skimage.io.imread(IMAGE_PATH + '/' + test_id + '/images/' + test_id + '.png')[:,:,:3]
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

        skimage.io.imsave(MASK_ENSEMBLE_SAVE_PATH + '/ensemble_' + test_id + '_mask.png',
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

    # if test_flag:
    #     sub = pd.DataFrame()
    #     sub['ID'] = new_test_ids
    #     sub['RLE'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    #     sub.to_csv('RLE-test-ensemble.csv', index=False)

###########################################
# main
###########################################

def main_ensemble(params):

    TRAIN_IMAGE_PATH = params['TRAIN_IMAGE_PATH']
    TEST_IMAGE_PATH = params['TEST_IMAGE_PATH']
    VAL_MASK_SAVE_PATH = params['VAL_MASK_SAVE_PATH']
    TEST_MASK_SAVE_PATH = params['TEST_MASK_SAVE_PATH']
    VAL_MASK_ENSEMBLE_SAVE_PATH = params['VAL_MASK_ENSEMBLE_SAVE_PATH']
    TEST_MASK_ENSEMBLE_SAVE_PATH = params['TEST_MASK_ENSEMBLE_SAVE_PATH']
    test_flag = params['test_flag']
    model_names = params['model_names']

    if test_flag:
        IMAGE_PATH = TEST_IMAGE_PATH
        ensemble_input_dir = TEST_MASK_SAVE_PATH
        MASK_ENSEMBLE_SAVE_PATH = TEST_MASK_ENSEMBLE_SAVE_PATH
    else:
        IMAGE_PATH = TRAIN_IMAGE_PATH
        ensemble_input_dir = VAL_MASK_SAVE_PATH
        MASK_ENSEMBLE_SAVE_PATH = VAL_MASK_ENSEMBLE_SAVE_PATH

    if not os.path.exists(MASK_ENSEMBLE_SAVE_PATH):
        os.makedirs(MASK_ENSEMBLE_SAVE_PATH)
    ensemble_dirs = []
    for models in model_names:
        ensemble_dirs.append(ensemble_input_dir + '/' + models + '/')
        ensemble_dirs.append(ensemble_input_dir + '/' + models + '_vflip/')
        ensemble_dirs.append(ensemble_input_dir + '/' + models + '_hflip/')

    print ensemble_dirs
    ensemble_func(ensemble_dirs=ensemble_dirs, MASK_ENSEMBLE_SAVE_PATH=MASK_ENSEMBLE_SAVE_PATH,
                  IMAGE_PATH=IMAGE_PATH,test_flag=test_flag)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--TRAIN_IMAGE_PATH', default='data/train',
                        help='directory of training images')
    parser.add_argument('--TEST_IMAGE_PATH', default='data/test',
                        help='directory of test images')

    parser.add_argument('--VAL_MASK_SAVE_PATH', default='data/masks_val',
                        help='directory of segmentation masks for validation images')
    parser.add_argument('--TEST_MASK_SAVE_PATH', default='data/masks_test',
                        help='directory of segmentation masks for test images')

    parser.add_argument('--VAL_MASK_ENSEMBLE_SAVE_PATH', default='data/masks_val_ensemble',
                        help='directory of segmentation masks for validation images after ensemble')
    parser.add_argument('--TEST_MASK_ENSEMBLE_SAVE_PATH', default='data/masks_test_ensemble',
                        help='directory of segmentation masks for test images after ensemble')

    parser.add_argument('--test_flag', default=True,
                        help='if yes: ensemble on test images, otherwise on validation images')

    parser.add_argument('--model_names', default=['nuclei_train20180518T2026_0024','nuclei_train20180519T0111_0025'],
                        help='models to ensemble')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main_ensemble(params)