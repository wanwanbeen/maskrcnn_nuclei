__authors__="Jie Yang and Xinyang Feng"

###########################################
# mask2rle after postprocessing
###########################################

import argparse
import numpy as np
import glob
import pandas as pd

def train_val_split(root_dir, TRAIN_DIR, MOSAIC_TRAIN_DIR):

    split_csv = 'image_group_train.csv'
    if len(glob.glob(root_dir+split_csv)) == 0:

        image_ids = glob.glob(TRAIN_DIR+'/*/images')

        ids = []
        for image_id in image_ids:
            ids.append(image_id.split('/')[-2])
        ids.sort()
        istrain = np.zeros(len(ids))
        np.random.seed(0)
        train_list = np.random.permutation(len(ids))[:int(len(ids)*0.8)]
        istrain[train_list] = 1

        mos_id = [0]*len(ids)
        image_ids = glob.glob(MOSAIC_TRAIN_DIR + '/*/*_list.txt')
        for image_id in image_ids:
            text_file = open(image_id,'r')
            lines = text_file.read().split('\n')
            text_file.close()
            mosaic_id = image_id.split('/')[-2]
            for id in lines:
                id = id.split('/')[-1][:-4]
                print id
                mos_id[ids.index(id)] = mosaic_id

        print(len(ids))
        print len(istrain)
        print len(mos_id)
        sub = pd.DataFrame()
        sub['id'] = ids
        sub['istrain'] = istrain
        sub['mosaic_id'] = mos_id
        sub.to_csv(split_csv, index=False)

    return None

def main_split(params):

    root_dir = params['ROOT_DIR']
    MOSAIC_TRAIN_DIR = params['MOSAIC_TRAIN_DIR']
    TRAIN_DIR = params['TRAIN_DIR']
    train_val_split(root_dir, TRAIN_DIR, MOSAIC_TRAIN_DIR)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--ROOT_DIR', default='', help='root directory of the project')
    parser.add_argument('--TRAIN_DIR', default='data/train', help='directory of training image')
    parser.add_argument('--MOSAIC_TRAIN_DIR', default='data/mosaic_train', help='directory of mosaic training image')

    args = parser.parse_args()
    params = vars(args)
    main_split(params)