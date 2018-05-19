__authors__="Jie Yang and Xinyang Feng"

###########################################
# mosaic
###########################################
import pandas as pd
import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import cv2  # To read and manipulate images
import os  # For filepath, directory handling
import sys  # System-specific parameters and functions
import tqdm  # Use smart progress meter
import matplotlib.pyplot as plt  # Python 2D plotting library

import warnings
warnings.filterwarnings("ignore")
from skimage.color import label2rgb
import skimage.io
import networkx
from networkx.algorithms.components.connected import connected_components

# DIRECTORIES
IMG_DIR_NAME = 'images'  # Folder name including the image
MASK_DIR_NAME = 'masks'  # Folder name including the masks

###########################################
# utils
###########################################

def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None, space='bgr'):
    """Read an image from a file and resize it."""
    img = cv2.imread(filepath, color_mode)
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    if space == 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img

def read_train_data_properties(train_dir, img_dir_name, mask_dir_name):
    """Read basic properties of training images and masks"""
    tmp = []
    for i, dir_name in enumerate(next(os.walk(train_dir))[1]):
        img_dir = os.path.join(train_dir, dir_name, img_dir_name)
        mask_dir = os.path.join(train_dir, dir_name, mask_dir_name)
        num_masks = len(next(os.walk(mask_dir))[2])
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        # mask_path = os.path.join(train_dir,dir_name,FULL_MASK_DIR_NAME,img_name_id+'_mask.png')
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0] / img_shape[1], img_shape[2], num_masks,
                    img_path, mask_dir])

    train_df = pd.DataFrame(tmp, columns=['img_id', 'img_height', 'img_width',
                                          'img_ratio', 'num_channels',
                                          'num_masks', 'image_path', 'mask_dir'])
    return train_df

def read_test_data_properties(test_dir, img_dir_name):
    """Read basic properties of test images."""
    tmp = []
    for i, dir_name in enumerate(next(os.walk(test_dir))[1]):
        img_dir = os.path.join(test_dir, dir_name, img_dir_name)
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0] / img_shape[1], img_shape[2], img_path])

    test_df = pd.DataFrame(tmp, columns=['img_id', 'img_height', 'img_width',
                                         'img_ratio', 'num_channels', 'image_path'])
    return test_df

def load_raw_data(train_df, test_df, image_size=(256, 256), space='bgr', load_mask=True):
    """Load raw data."""
    # Python lists to store the training images/masks and test images.
    x_train, y_train, x_test = [], [], []

    # Read and resize train images/masks.
    print('Loading and resizing train images and masks ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(train_df['image_path']), total=len(train_df)):
        img = read_image(train_df['image_path'].loc[i], target_size=image_size, space=space)
        if load_mask:
            mask = read_image(train_df['mask_path'].loc[i],
                              color_mode=cv2.IMREAD_GRAYSCALE,
                              target_size=image_size)
            # mask = read_mask(train_df['mask_dir'].loc[i], target_size=image_size)
            y_train.append(mask)
        x_train.append(img)

    # Read and resize test images.
    print('Loading and resizing test images ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(test_df['image_path']), total=len(test_df)):
        img = read_image(test_df['image_path'].loc[i], target_size=image_size, space=space)
        x_test.append(img)

    # Transform lists into 4-dim numpy arrays.
    x_train = np.array(x_train)
    # if load_mask:
    y_train = np.array(y_train)
    # y_train = np.expand_dims(np.array(y_train), axis=4)
    x_test = np.array(x_test)
    print('Data loaded')
    if load_mask:
        return x_train, y_train, x_test
    else:
        return x_train, x_test

def get_domimant_colors(img, top_colors=1):
    """Return dominant image color"""
    img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    clt = KMeans(n_clusters=top_colors)
    clt.fit(img_l)
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return clt.cluster_centers_, hist

def cluster_images_by_hsv(train_df, test_df):
    """Clusterization based on hsv colors. Adds 'hsv_cluster' column to tables"""
    print('Loading data')
    x_train_hsv, x_test_hsv = load_raw_data(train_df, test_df, image_size=None, space='hsv', load_mask=False)
    x_hsv = np.concatenate([x_train_hsv, x_test_hsv])
    print('Calculating dominant hsv for each image')
    dominant_hsv = []
    for img in tqdm.tqdm(x_hsv):
        res1, res2 = get_domimant_colors(img, top_colors=1)
        dominant_hsv.append(res1.squeeze())
    print('Calculating clusters')
    kmeans = KMeans(n_clusters=3).fit(dominant_hsv)
    train_df['HSV_CLUSTER'] = kmeans.predict(dominant_hsv[:len(x_train_hsv)])
    test_df['HSV_CLUSTER'] = kmeans.predict(dominant_hsv[len(x_train_hsv):])
    print('Images clustered')
    return None

def plot_images(selected_images_df, images_rows=4, images_cols=8, plot_figsize=4):
    """Plot image_rows*image_cols of selected images. Used to visualy check clusterization"""
    f, axarr = plt.subplots(images_rows, images_cols, figsize=(plot_figsize * images_cols, images_rows * plot_figsize))
    for row in range(images_rows):
        for col in range(images_cols):
            if (row * images_cols + col) < selected_images_df.shape[0]:
                image_path = selected_images_df['image_path'].iloc[row * images_cols + col]
            else:
                continue
            img = read_image(image_path)
            # print image_path
            height, width, l = img.shape
            ax = axarr[row, col]
            ax.axis('off')
            ax.set_title("%dx%d" % (width, height))
            ax.imshow(img)

###########################################
# mosaic
###########################################
def combine_images(data, indexes):
    """ Combines img from data using indexes as follows:
        0 1
        2 3 
    """
    up = np.hstack([data[indexes[0]], data[indexes[1]]])
    down = np.hstack([data[indexes[2]], data[indexes[3]]])
    full = np.vstack([up, down])
    return full

def combine_images_4(data_ul, data_ur, data_ll, data_lr):
    """ Combines img from data using indexes as follows:
        0 1
        2 3 
    """
    up = np.hstack([data_ul, data_ur])
    down = np.hstack([data_ll, data_lr])
    full = np.vstack([up, down])
    return full

def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current

def make_mosaic(data, return_connectivity=False, plot_images=False, external_df=None):
    """Find images with similar borders and combine them to one big image"""
    if external_df is not None:
        external_df['mosaic_idx'] = np.nan
        external_df['mosaic_position'] = np.nan
        # print(external_df.head())

    # extract borders from images
    borders = []
    for x in data:
        borders.extend([x[0, :, :].flatten(), x[-1, :, :].flatten(),
                        x[:, 0, :].flatten(), x[:, -1, :].flatten()])
    borders = np.array(borders)

    # prepare df with all data
    lens = np.array([len(border) for border in borders])
    img_idx = list(range(len(data))) * 4
    img_idx.sort()
    position = ['up', 'down', 'left', 'right'] * len(data)
    nn = [None] * len(position)
    df = pd.DataFrame(data=np.vstack([img_idx, position, borders, lens, nn]).T,
                      columns=['img_idx', 'position', 'border', 'len', 'nn'])
    uniq_lens = df['len'].unique()

    for idx, l in enumerate(uniq_lens):
        # fit NN on borders of certain size with 1 neighbor
        nn = NearestNeighbors(n_neighbors=1).fit(np.stack(df[df.len == l]['border'].values))
        distances, neighbors = nn.kneighbors()
        real_neighbor = np.array([None] * len(neighbors))
        distances, neighbors = distances.flatten(), neighbors.flatten()

        # if many borders are close to one, we want to take only the closest
        uniq_neighbors = np.unique(neighbors)

        # difficult to understand but works :c
        for un_n in uniq_neighbors:
            # min distance for borders with same nn
            min_index = list(distances).index(distances[neighbors == un_n].min())
            # check that min is double-sided
            double_sided = distances[neighbors[min_index]] == distances[neighbors == un_n].min()
            if double_sided and distances[neighbors[min_index]] < 1000:
                real_neighbor[min_index] = neighbors[min_index]
                real_neighbor[neighbors[min_index]] = min_index
        indexes = df[df.len == l].index
        for idx2, r_n in enumerate(real_neighbor):
            if r_n is not None:
                df['nn'].iloc[indexes[idx2]] = indexes[r_n]

    # img connectivity graph.
    img_connectivity = {}
    for img in df.img_idx.unique():
        slc = df[df['img_idx'] == img]
        img_nn = {}

        # get near images_id & position
        for nn_border, position in zip(slc[slc['nn'].notnull()]['nn'],
                                       slc[slc['nn'].notnull()]['position']):

            # filter obvious errors when we try to connect bottom of one image to bottom of another
            # my hypotesis is that images were simply cut, without rotation
            if position == df.iloc[nn_border]['position']:
                continue
            img_nn[position] = df.iloc[nn_border]['img_idx']
        img_connectivity[img] = img_nn

    imgs = []
    indexes = set()
    mosaic_idx = 0

    # errors in connectivity are filtered
    good_img_connectivity = {}
    for k, v in img_connectivity.items():
        if v.get('down') is not None:
            if v.get('right') is not None:
                # need down right image
                # check if both right and down image are connected to the same image in the down right corner
                if (img_connectivity[v['right']].get('down') is not None) and img_connectivity[v['down']].get(
                        'right') is not None:
                    if img_connectivity[v['right']]['down'] == img_connectivity[v['down']]['right']:
                        v['down_right'] = img_connectivity[v['right']]['down']
                        temp_indexes = [k, v['right'], v['down'], v['down_right']]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        # It is necessary here to filter that they are not the same
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(data, temp_indexes))
                        if external_df is not None:
                            external_df['mosaic_idx'].iloc[temp_indexes] = mosaic_idx
                            external_df['mosaic_position'].iloc[temp_indexes] = ['up_left', 'up_right', 'down_left',
                                                                                 'down_right']
                            mosaic_idx += 1
                        continue
            if v.get('left') is not None:
                # need down left image
                if img_connectivity[v['left']].get('down') is not None and img_connectivity[v['down']].get(
                        'left') is not None:
                    if img_connectivity[v['left']]['down'] == img_connectivity[v['down']]['left']:
                        v['down_left'] = img_connectivity[v['left']]['down']
                        temp_indexes = [v['left'], k, v['down_left'], v['down']]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(data, temp_indexes))

                        if external_df is not None:
                            external_df['mosaic_idx'].iloc[temp_indexes] = mosaic_idx
                            external_df['mosaic_position'].iloc[temp_indexes] = ['up_left', 'up_right', 'down_left',
                                                                                 'down_right']

                            mosaic_idx += 1
                        continue
        if v.get('up') is not None:
            if v.get('right') is not None:
                # need up right image
                if img_connectivity[v['right']].get('up') is not None and img_connectivity[v['up']].get(
                        'right') is not None:
                    if img_connectivity[v['right']]['up'] == img_connectivity[v['up']]['right']:
                        v['up_right'] = img_connectivity[v['right']]['up']
                        temp_indexes = [v['up'], v['up_right'], k, v['right']]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(data, temp_indexes))

                        if external_df is not None:
                            external_df['mosaic_idx'].iloc[temp_indexes] = mosaic_idx
                            external_df['mosaic_position'].iloc[temp_indexes] = ['up_left', 'up_right', 'down_left',
                                                                                 'down_right']

                            mosaic_idx += 1
                        continue
            if v.get('left') is not None:
                # need up left image
                if img_connectivity[v['left']].get('up') is not None and img_connectivity[v['up']].get(
                        'left') is not None:
                    if img_connectivity[v['left']]['up'] == img_connectivity[v['up']]['left']:
                        v['up_left'] = img_connectivity[v['left']]['up']
                        temp_indexes = [v['up_left'], v['up'], v['left'], k]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(data, temp_indexes))

                        if external_df is not None:
                            external_df['mosaic_idx'].iloc[temp_indexes] = mosaic_idx
                            external_df['mosaic_position'].iloc[temp_indexes] = ['up_left', 'up_right', 'down_left',
                                                                                 'down_right']

                            mosaic_idx += 1
                        continue

    # same images are present 4 times (one for every piece) so we need to filter them
    print('Images before filtering: {}'.format(np.shape(imgs)))

    # can use np. unique only on images of one size, flatten first, then select
    flattened = np.array([i.flatten() for i in imgs])
    uniq_lens = np.unique([i.shape for i in flattened])
    filtered_imgs = []
    for un_l in uniq_lens:
        filtered_imgs.extend(np.unique(np.array([i for i in imgs if i.flatten().shape == un_l])))

    filtered_imgs = np.array(filtered_imgs)
    print('Images after filtering: {}'.format(np.shape(filtered_imgs)))

    if return_connectivity:
        print(good_img_connectivity)

    if plot_images:
        for i in filtered_imgs:
            plt.imshow(i)
            plt.show()

    # list of not combined images. return if you need
    not_combined = list(set(range(len(data))) - indexes)

    if external_df is not None:
        # un_mos_id = external_df[external_df.mosaic_idx.notnull()].mosaic_idx.unique()
        # mos_dict = {k:v for k,v in zip(un_mos_id,range(len(un_mos_id)))}
        # external_df.mosaic_idx = external_df.mosaic_idx.map(mos_dict)
        external_df.loc[external_df[external_df['mosaic_idx'].isnull()].index, 'mosaic_idx'] = range(
            int(np.nanmax(external_df.mosaic_idx.unique())) + 1,
            int(np.nanmax(external_df.mosaic_idx.unique())) + 1 + len(
                external_df.mosaic_idx[external_df.mosaic_idx.isnull()]))
        external_df['mosaic_idx'] = external_df['mosaic_idx'].astype(np.int32)
        if return_connectivity:
            return filtered_imgs, external_df, good_img_connectivity
        else:
            return filtered_imgs, external_df
    if return_connectivity:
        return filtered_imgs, good_img_connectivity
    else:
        return filtered_imgs
###########################################
# main
###########################################

def main(params):

    TRAIN_DIR = params['TRAIN_DIR']
    TEST_DIR = params['TEST_DIR']
    MOSAIC_TRAIN_DIR = params['MOSAIC_TRAIN_DIR']
    MOSAIC_TEST_DIR = params['MOSAIC_TEST_DIR']
    TRAIN_ONLY = params['TRAIN_ONLY']

    # Basic properties of images/masks.
    if not os.path.exists('./train_df.csv'):
        train_df = read_train_data_properties(TRAIN_DIR, IMG_DIR_NAME, MASK_DIR_NAME)
        test_df = read_test_data_properties(TEST_DIR, IMG_DIR_NAME)
        cluster_images_by_hsv(train_df,test_df)
        train_df.to_csv('./train_df.csv',index=False)
        test_df.to_csv('./test_df.csv',index=False)
    else:
        # after generated, simply read
        train_df = pd.read_csv('./train_df.csv')
        test_df = pd.read_csv('./test_df.csv')

    train_change_filepath = lambda x: TRAIN_DIR + '/{0}/images/{0}.png'.format(x.split('/')[-1][:-4])
    test_change_filepath = lambda x: TEST_DIR + '/{0}/images/{0}.png'.format(x.split('/')[-1][:-4])
    train_df.image_path = train_df.image_path.map(train_change_filepath)
    # train_df.drop(['mask_dir'],inplace=True,axis = 1)
    test_df.image_path = test_df.image_path.map(test_change_filepath)

    for idx in range(3):
        print("Images in cluster {}: {}".format(idx,train_df[train_df['HSV_CLUSTER'] == idx].shape[0]))
    plot_images(train_df[train_df['HSV_CLUSTER'] == 0],2,4)
    plot_images(train_df[train_df['HSV_CLUSTER'] == 1],2,4)
    plot_images(train_df[train_df['HSV_CLUSTER'] == 2],2,4)

    for idx in range(3):
        print("Images in cluster {}: {}".format(idx,test_df[test_df['HSV_CLUSTER'] == idx].shape[0]))
    plot_images(test_df[test_df['HSV_CLUSTER'] == 0],2,4)
    plot_images(test_df[test_df['HSV_CLUSTER'] == 2],2,4)
    plot_images(test_df[test_df['HSV_CLUSTER'] == 1],2,4)


    # Read images/masks from files and resize them. Each image and mask
    # is stored as a 3-dim array where the number of channels is 3 and 1, respectively.
    x_train, x_test = load_raw_data(train_df, test_df, load_mask=False,image_size=None)

    if not TRAIN_ONLY:
        _, data_frame_test, conn_test = make_mosaic(x_test,return_connectivity=True,plot_images=False,external_df=test_df)
        # code which makes csv with clusters and mosaic ids for test data
        data_frame_test[['img_id','HSV_CLUSTER','mosaic_idx','mosaic_position']].head(20)
        data_frame_test.to_csv('./test_mosaic.csv',index=False)

        for m in conn_test.keys():
            ids = conn_test[m]
            img_dir_ul = data_frame_test['image_path'][ids[0]]
            img_dir_ur = data_frame_test['image_path'][ids[1]]
            img_dir_ll = data_frame_test['image_path'][ids[2]]
            img_dir_lr = data_frame_test['image_path'][ids[3]]

            img_ul = skimage.io.imread(img_dir_ul)
            img_ur = skimage.io.imread(img_dir_ur)
            img_ll = skimage.io.imread(img_dir_ll)
            img_lr = skimage.io.imread(img_dir_lr)

            img_all = combine_images_4(img_ul, img_ur, img_ll, img_lr)

            mosaic_savedir = os.path.join(MOSAIC_TEST_DIR, 'mosaic_test_' + '%.3i' % m)
            mosaic_savedir_img = os.path.join(mosaic_savedir, 'images')

            if not os.path.exists(mosaic_savedir_img):
                os.makedirs(mosaic_savedir_img)

            skimage.io.imsave(os.path.join(mosaic_savedir_img, 'mosaic_test_' + '%.3i' % m + '_image.png'), img_all)

            f = open(os.path.join(mosaic_savedir, 'mosaic_test_' + '%.3i' % m + '_list.txt'), 'w')
            f.write(img_dir_ul + '\n' + img_dir_ur + '\n' + img_dir_ll + '\n' + img_dir_lr)
            f.close()
            np.save(os.path.join(mosaic_savedir, 'mosaic_test_' + '%.3i' % m + '_ul_shappe.npy'), img_ul.shape)


    _, data_frame_train, conn_train = make_mosaic(x_train,return_connectivity=True, plot_images=False,external_df=train_df)
    # code which makes csv with clusters and mosaic ids for test data
    data_frame_train[['img_id','HSV_CLUSTER','mosaic_idx','mosaic_position']].head(20)
    data_frame_train.to_csv('./train_mosaic.csv',index=False)

    for m in conn_train.keys():
        print m
        ids = conn_train[m]
        img_dir_ul = data_frame_train['image_path'][ids[0]]
        img_dir_ur = data_frame_train['image_path'][ids[1]]
        img_dir_ll = data_frame_train['image_path'][ids[2]]
        img_dir_lr = data_frame_train['image_path'][ids[3]]

        mask_dir_ul = img_dir_ul.split('images')[0] + 'masks/'
        mask_dir_ur = img_dir_ur.split('images')[0] + 'masks/'
        mask_dir_ll = img_dir_ll.split('images')[0] + 'masks/'
        mask_dir_lr = img_dir_lr.split('images')[0] + 'masks/'

        img_ul = skimage.io.imread(img_dir_ul)
        img_ur = skimage.io.imread(img_dir_ur)
        img_ll = skimage.io.imread(img_dir_ll)
        img_lr = skimage.io.imread(img_dir_lr)

        img_all = combine_images_4(img_ul, img_ur, img_ll, img_lr)

        mask_name_ul = next(os.walk(mask_dir_ul))[2]
        mask_name_ur = next(os.walk(mask_dir_ur))[2]
        mask_name_ll = next(os.walk(mask_dir_ll))[2]
        mask_name_lr = next(os.walk(mask_dir_lr))[2]

        masks = []
        qds = []
        for k in mask_name_ul:
            masks.append(combine_images_4(
                skimage.io.imread(mask_dir_ul + k),
                np.zeros((img_ur.shape[0:2])),
                np.zeros((img_ll.shape[0:2])),
                np.zeros((img_lr.shape[0:2]))).astype(np.uint16))
            qds.append(0)

        for k in mask_name_ur:
            masks.append(combine_images_4(
                np.zeros((img_ul.shape[0:2])),
                skimage.io.imread(mask_dir_ur + k),
                np.zeros((img_ll.shape[0:2])),
                np.zeros((img_lr.shape[0:2]))).astype(np.uint16))
            qds.append(1)

        for k in mask_name_ll:
            masks.append(combine_images_4(
                np.zeros((img_ul.shape[0:2])),
                np.zeros((img_ur.shape[0:2])),
                skimage.io.imread(mask_dir_ll + k),
                np.zeros((img_lr.shape[0:2]))).astype(np.uint16))
            qds.append(2)

        for k in mask_name_lr:
            masks.append(combine_images_4(
                np.zeros((img_ul.shape[0:2])),
                np.zeros((img_ur.shape[0:2])),
                np.zeros((img_ll.shape[0:2])),
                skimage.io.imread(mask_dir_lr + k)).astype(np.uint16))
            qds.append(3)

        qds = np.array(qds)

        mosaic_savedir = os.path.join(MOSAIC_TRAIN_DIR, 'mosaic_' + '%.3i' % m)
        mosaic_savedir_img = os.path.join(mosaic_savedir, 'images')
        mosaic_savedir_mask = os.path.join(mosaic_savedir, 'masks')

        if not os.path.exists(mosaic_savedir_img):
            os.makedirs(mosaic_savedir_img)
        if not os.path.exists(mosaic_savedir_mask):
            os.makedirs(mosaic_savedir_mask)

        skimage.io.imsave(os.path.join(mosaic_savedir_img, 'mosaic_' + '%.3i' % m + '_image.png'), img_all)
        mask_collapse = np.zeros(masks[0].shape)
        for i, mask in enumerate(masks):
            mask_collapse = mask_collapse + (i + 1) * (mask / 255.)
        mask_collapse = np.uint16(mask_collapse)

        b_row_u = mask_collapse[img_ul.shape[0] - 1, :]
        b_row_l = mask_collapse[img_ul.shape[0], :]
        b_col_l = mask_collapse[:, img_ul.shape[1] - 1]
        b_col_r = mask_collapse[:, img_ul.shape[1]]

        merge_list = []
        for k in list(np.unique(b_row_u[(b_row_u * b_row_l) > 0])):
            nb_labels = list(np.unique(b_row_l[((b_row_u == k) * b_row_l) > 0]))
            nb_labels.append(k)
            merge_list.append(nb_labels)

        for k in list(np.unique(b_col_l[(b_col_l * b_col_r) > 0])):
            nb_labels = list(np.unique(b_col_r[((b_col_l == k) * b_col_r) > 0]))
            nb_labels.append(k)
            merge_list.append(nb_labels)

        del_ind = np.unique(np.array([item for sublist in merge_list for item in sublist])) - 1

        G = to_graph(merge_list)
        merge_list_new = [list(c) for c in connected_components(G)]

        masks_merged = []
        for ml in merge_list_new:
            mask_tmp = np.zeros((masks[0].shape[0:2]))
            for ch in ml:
                mask_tmp += masks[ch - 1]
            masks_merged.append(mask_tmp.astype(np.uint16))

        for i in sorted(del_ind, reverse=True):
            del masks[i]

        masks.extend(masks_merged)
        mask_collapse = np.zeros(masks[0].shape)
        for i, mask in enumerate(masks):
            skimage.io.imsave(
                os.path.join(mosaic_savedir_mask, 'mosaic_' + '%.3i' % m + '_mask' + '%.3i' % i + '.png'), mask)
            mask_collapse = mask_collapse + (i + 1) * (mask / 255.)
        mask_collapse = np.uint16(mask_collapse)

        ## imsave
        skimage.io.imsave(os.path.join(mosaic_savedir, 'mosaic_' + '%.3i' % m + '_allmasks_scalar.png'),
                          mask_collapse)

        skimage.io.imsave(os.path.join(mosaic_savedir, 'mosaic_' + '%.3i' % m + '_allmasks.png'),
                          label2rgb(mask_collapse, bg_label=0))

        skimage.io.imsave(os.path.join(mosaic_savedir, 'mosaic_' + '%.3i' % m + '_allmasks_overlaid.png'),
                          label2rgb(mask_collapse, img_all, alpha=0.1, bg_label=0))

        f = open(os.path.join(mosaic_savedir, 'mosaic_' + '%.3i' % m + '_list.txt'), 'w')
        f.write(img_dir_ul + '\n' + img_dir_ur + '\n' + img_dir_ll + '\n' + img_dir_lr)
        f.close()

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument('--TRAIN_DIR', default='data/train', help='directory of training image')
  parser.add_argument('--TEST_DIR', default='data/test', help='directory of test image')
  parser.add_argument('--MOSAIC_TRAIN_DIR', default='data/mosaic_train', help='directory of mosaic training image')
  parser.add_argument('--MOSAIC_TEST_DIR', default='data/mosaic_test', help='directory of mosaic test image')
  parser.add_argument('--TRAIN_ONLY', default=True, help='apply only on training image; if false, apply on train & test')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  main(params)
