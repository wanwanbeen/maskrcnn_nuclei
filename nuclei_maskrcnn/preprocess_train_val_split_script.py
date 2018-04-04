from sklearn.model_selection import train_test_split
import os
import pickle
import pandas as pd
import numpy as np

ROOT_DIR=os.getcwd()
h = open(os.path.join(ROOT_DIR, "image_group_train.pickle"))
d = pickle.load(h)
train_ids = []
val_ids = []
rep_id = [2,2,8,6,4,4,8]
for k in range(1,8):
    train_id, val_id = train_test_split(d[k],train_size=0.8, random_state=1234)
    train_ids.extend(train_id)
    val_ids.extend(val_id)

# f_train = open(os.path.join(ROOT_DIR,"train_train_ids_1234.txt"),'w')
# for train_id in train_ids:
#     f_train.write("%s\n" % train_id)
# f_train.close()
#
# f_val = open(os.path.join(ROOT_DIR,"train_val_ids_1234.txt"),'w')
# for val_id in val_ids:
#     f_val.write("%s\n" % val_id)
# f_val.close()

df = pd.read_csv('image_group_train_pickle.csv')
ids = df['id']
groups = df['group']
istrain = df['istrain']
mosaic_ids = df['mosaic_id']
mosaic_train_ids = np.unique(mosaic_ids[istrain==1][1:])
