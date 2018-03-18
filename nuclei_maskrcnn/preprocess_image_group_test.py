import numpy as np
import cv2
import pickle
from glob import glob

dir_in  = 'stage1_test'
dir_out = 'img_group_test'
dir_img = glob('./data/'+dir_in+'/*/images/*.png')
color   = np.zeros([len(dir_img),5])
i       = 0

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

    info = color[i, :]

    if info[0]==0 and info[1]==0 and info[2]==0:
        if idno == '4f949bd8d914bbfa06f40d6a0e2b5b75c38bf53dbcbafc48c97f105bee4f8fac':
            image_group[3].append(idno)
            cv2.imwrite('./data/' + dir_out + '/3_' + idno + '.png', img)
        else:
            image_group[2].append(idno)
            cv2.imwrite('./data/'+dir_out+'/2_' + idno + '.png', img)
    elif info[0]==255 and info[1]==255 and info[2]==255 and info[3] < 190:
        image_group[7].append(idno)
        cv2.imwrite('./data/'+dir_out+'/7_' + idno + '.png', img)
    else:
        image_group[8].append(idno)
        cv2.imwrite('./data/'+dir_out+'/8_' + idno + '.png', img)

    i += 1

pickle_out = open("image_group_test.pickle","wb")
pickle.dump(image_group, pickle_out)
pickle_out.close()

pickle_in  = open("image_group_train.pickle")
image_group_train = pickle.load(pickle_in)

for nclass in range(1,9):
    print nclass,len(image_group_train[nclass]),len(image_group[nclass])
    image_group[nclass] += image_group_train[nclass]
    print nclass,len(image_group_train[nclass]), len(image_group[nclass])

pickle_out = open("image_group_all.pickle","wb")
pickle.dump(image_group, pickle_out)
pickle_out.close()




