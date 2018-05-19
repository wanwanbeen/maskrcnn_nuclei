__authors__="Jie Yang and Xinyang Feng"

###########################################
# mask2rle after postprocessing
###########################################

import argparse
import numpy as np
import os
import glob
import pandas as pd
import cv2
from scipy import ndimage

def postprocess(dir_input, dir_output):

    dir_mask = glob.glob(dir_input +'/*_mask.npy')
    colors_1 = [80, 200, 250]
    colors_2 = [250, 200, 80]
    kernel = np.ones((3,3), np.uint8)

    for i in range(len(dir_mask)):
        id = dir_mask[i].split('/')[-1][:-9]
        I = cv2.imread(dir_input + '/ensemble_'+ id + '_mask.png')
        M = np.load(dir_mask[i])
        I = I[:, :I.shape[1]/3,:]
        tmp1 = I[:,:,0]
        tmp2 = I[:,:,1]
        tmp3 = I[:,:,2]
        if np.sum((tmp1-tmp2).astype(int)) == 0 & np.sum((tmp2-tmp3).astype(int)) == 0:
            gray_image = True
        else:
            gray_image = False

        V = np.unique(M)[1:]
        Mnew = np.zeros(M.shape)

        for n in V:
            tmp = np.zeros(M.shape)
            tmp[M == n] = 1

            if gray_image:
                tmp_d = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel).astype(int)
            else:
                tmp_d = cv2.dilate(tmp, kernel).astype(int)

            tmp_f = ndimage.binary_fill_holes(tmp_d).astype(int)
            tmp_f[0,:] = tmp_f[1,:]
            tmp_f[:,0] = tmp_f[:,1]
            tmp_f[-1,:] = tmp_f[-2,:]
            tmp_f[:,-1] = tmp_f[:,-2]

            Mnew[tmp_f > 0] = n

        Im = I.copy()
        for z in range(I.shape[2]):
            tmp = Im[:,:, z]
            tmp[M > 0] = colors_1[z]
            Im[:,:, z] = tmp

        Imnew = I.copy()
        for z in range(I.shape[2]):
            tmp = Imnew[:,:, z]
            tmp[Mnew > 0] = colors_2[z]
            Imnew[:,:, z] = tmp

        I = cv2.imread(dir_input + '/ensemble_' + id + '_mask.png')
        I[:, I.shape[1] / 3 * 1:I.shape[1] / 3 * 2,:] = Im
        I[:, I.shape[1] / 3 * 2:,:] = Imnew
        cv2.imwrite(dir_output+'/' + id + '_mask.png', I)
        np.save(dir_output + '/'+dir_mask[i].split('/')[-1], Mnew)

def rle_encoding(x):
    dots = np.where(x.T.flatten()==1)[0]
    run_lengths=[]
    prev=-2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1,0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rle(x,cutoff = 0.5):
    yield rle_encoding(x>cutoff)

def write_rle(dir_input):
    image_id = glob.glob(dir_input+'/*.npy')
    rles = []
    test_id = []
    for ids in image_id:
        idno = ids.split('/')[-1][:-9]
        M=np.load(ids)
        V=np.unique(M)
        V=V[1:]
        mask=np.zeros([M.shape[0],M.shape[1],len(V)])
        for n in range(len(V)):
            tmp = np.zeros([M.shape[0],M.shape[1]])
            tmp[M == V[n]]=1
            mask[:,:,n]=tmp
        print mask.shape, idno
        for n in range(mask.shape[2]):
            rle = list(prob_to_rle(mask[:,:,n]))
            rles.extend(rle)
            test_id.append(idno)

    sub=pd.DataFrame()
    sub['ID']=test_id
    sub['RLE']=pd.Series(rles).apply(lambda x:' '.join(str(y) for y in x))
    sub.to_csv('RLE-test.csv',index=False)

def main_ensemble(params):

    MASK_ENSEMBLE_SAVE_PATH = params['MASK_ENSEMBLE_SAVE_PATH']
    MASK_POSTPROCESS_SAVE_PATH = params['MASK_POSTPROCESS_SAVE_PATH']

    if not os.path.exists(MASK_POSTPROCESS_SAVE_PATH):
        os.makedirs(MASK_POSTPROCESS_SAVE_PATH)

    postprocess(MASK_ENSEMBLE_SAVE_PATH, MASK_POSTPROCESS_SAVE_PATH)
    write_rle(MASK_POSTPROCESS_SAVE_PATH)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--MASK_ENSEMBLE_SAVE_PATH', default='data/masks_test_ensemble',
                        help='directory of mask output after ensemble')
    parser.add_argument('--MASK_POSTPROCESS_SAVE_PATH', default='data/mask_test_postprocess',
                        help='directory of mask output after postprocess')

    args = parser.parse_args()
    params = vars(args)
    main_ensemble(params)