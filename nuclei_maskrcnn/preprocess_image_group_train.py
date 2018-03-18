import numpy as np
import cv2
import pickle
from glob import glob

dir_in  = 'stage1_train'
dir_out = 'img_group_train'
dir_img = glob('./data/'+dir_in+'/*/images/*.png')
color   = np.zeros([len(dir_img),5])
i       = 0

class7  = ['8f27ebc74164eddfe989a98a754dcf5a9c85ef599a1321de24bcf097df1814ca',
           '57bd029b19c1b382bef9db3ac14f13ea85e36a6053b92e46caedee95c05847ab',
           '87ea72894f6534b28e740cc34cf5c9eb75d0d8902687fce5fcc08a92e9f41386',
           '4193474b2f1c72f735b13633b219d9cabdd43c21d9c2bb4dfc4809f104ba4c06']

image_group = {}
for nclass in range(1,9):
    image_group[nclass] = []

for dir_img_curr in dir_img:
    idno        = dir_img_curr.split('/')[-3]
    img         = cv2.imread(dir_img_curr)
    color[i, 0] = int(np.max(img[:, :, 0] - img[:, :, 1]))
    color[i, 1] = int(np.max(img[:, :, 1] - img[:, :, 2]))
    color[i, 2] = int(np.max(img[:, :, 0] - img[:, :, 2]))
    color[i, 3] = int(np.average(img))
    color[i, 4] = int(np.max(img))

    dir_mask = glob(('/').join(dir_img_curr.split('/')[:-2])+'/masks/*.png')
    mm  = np.zeros(len(dir_mask))
    j   = 0
    for dir_mask_curr in dir_mask:
        mask = cv2.imread(dir_mask_curr)
        mm[j] = np.sum(mask>=0)/np.sum(mask==255)
        j += 1

    color[i, 4] += int(np.min(mm))
    info = color[i,:]

    if info[0]==0 and info[1]==0 and info[2]==0 and info[3]<100:
        if info[4] < 150:
            image_group[1].append(idno)
            cv2.imwrite('./data/'+dir_out+'/1_' + idno + '.png', img)
        elif info[4] < 1200:
            image_group[2].append(idno)
            cv2.imwrite('./data/'+dir_out+'/2_' + idno + '.png', img)
        else:
            image_group[3].append(idno)
            cv2.imwrite('./data/'+dir_out+'/3_' + idno + '.png', img)
    elif info[0]==0 and info[1]==0 and info[2]==0 and info[3]>100:
        image_group[4].append(idno)
        cv2.imwrite('./data/'+dir_out+'/4_' + idno + '.png', img)
    elif info[0]==255 and info[1]==255 and info[2]==255:
        image_group[5].append(idno)
        cv2.imwrite('./data/'+dir_out+'/5_' + idno + '.png', img)
    elif idno in class7:
        image_group[7].append(idno)
        cv2.imwrite('./data/' + dir_out + '/7_' + idno + '.png', img)
    else:
        image_group[6].append(idno)
        cv2.imwrite('./data/'+dir_out+'/6_' + idno + '.png', img)

    i += 1

pickle_out = open("image_group_train.pickle","wb")
pickle.dump(image_group, pickle_out)
pickle_out.close()






