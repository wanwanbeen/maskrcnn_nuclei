__authors__="Jie Yang and Xinyang Feng"

###########################################
# mask2rle after postprocessing
###########################################

import numpy as np
import glob
import pandas as pd

def postprocess(dir_input, dir_output):

    dir_img = glob.glob(dir_input + '*.png')
    dir_mask = dir([dir1 '*.npy'])
    colors = [250, 200, 80]

    se = ones(3, 3)

    num_subfigures = 3

    for i= 1:size(dir_mask):
        id = dir_mask(i).name(1:end - 9)
        I = imread([dir_img(i).folder '/' dir_img(i).name])
        M = readNPY([dir_mask(i).folder '/' dir_mask(i).name])
        I = I(:, 1:size(I, 2) / num_subfigures,:)
        tmp1 = I(:,:, 1)
        tmp2 = I(:,:, 2)
        tmp3 = I(:,:, 3)
        if sum(tmp1(: ):
            -tmp2(:)) == 0 & sum(tmp2(:)-tmp2(:)) == 0
            gray_image = 1
        else:
            gray_image = 0

    V = unique(M(:))
    V(1) = []
    Mnew = zeros(size(M))

    mean_size = sum(M(:) > 0) / length(V)

    for n=1:length(V):
        tmp = zeros(size(M))
        tmp(M == V(n)) = 1

    if gray_image:
        tmp_d = imclose(tmp, se)
    else
        tmp_d = imdilate(tmp, se)


    tmp_f = imfill(tmp_d)
    tmp_f(1,:)=tmp_f(2,:)
    tmp_f(:, 1)=tmp_f(:, 2)
    tmp_f(end,:)=tmp_f(end - 1,:)
    tmp_f(:, end)=tmp_f(:, end - 1)

    tmp_f = bwareafilt(tmp_f > 0, 1)

    if sum(tmp_f(: ) > 0) / mean_size <= 10 % 3.5:
        Mnew(tmp_f > 0) = V(n)

    figure(1);
    subplot(1, 3, 1);
    imshow(I);
    Im = I;
    for z=1:3
    tmp = Im(:,:, z);
    tmp(M > 0) = colors(z);
    Im(:,:, z)=tmp;
    end
    subplot(1, 3, 2);
    imshow(Im);

    Im = I;
    for z=1:3
    tmp = Im(:,:, z);
    tmp(Mnew > 0) = colors(z);
    Im(:,:, z)=tmp;
    end
    subplot(1, 3, 3);
    imshow(Im);

    I = imread([dir_img(i).folder '/' dir_img(i).name]);
    I(:, size(I, 2) / num_subfigures * 2 + 1:end,:) = Im;
    % imwrite(I, [dir_img(i).folder '/' dir_img(i).name]);

    np.save(Mnew, [dir_out + dir_mask(i).name])

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
    image_id = glob.glob(dir_input+'*.npy')

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