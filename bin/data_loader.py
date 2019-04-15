import glob
import os
import uuid

import numpy as np
import pandas as pd
from skimage import io
from skimage.io import imsave
from skimage.transform import resize

user_images = glob.glob('../dataset_2')
directory_common = '../dataset_2/common/'
directory_original = '../data/'
train_prefix = 'train/'
val_prefix = 'valid/'

def load_dataset_2(imageset):
    if not os.path.exists(directory_common):
        os.mkdir(directory_common)
    else:
        print('Common directory already exists, exiting')
        return

    for user in ["user_3", "user_4", "user_5", "user_6", "user_7", "user_9", "user_10"]:
        boundingbox_df = pd.read_csv('../dataset_2/' + user + '/' + user + '_loc.csv')
        print('Saving data in common dir')

        for rows in boundingbox_df.iterrows():
            cropped_img = crop(imageset[rows[1]['image']],
                               rows[1]['top_left_x'],
                               rows[1]['bottom_right_x'],
                               rows[1]['top_left_y'],
                               rows[1]['bottom_right_y'],
                               128
                               )
            random_postfix = uuid.uuid4().hex
            lowercase_name = rows[1]['image'].split('/')[1].lower().split('.')[0] + '_' + random_postfix + '.jpg'
            image_name_with_extention = lowercase_name.replace('.jpg', '.png')

            imsave(directory_common + image_name_with_extention, cropped_img)

    print('Data saved in common dir')
    print('Dividing into train and val set')
    shuffle = np.random.permutation(len(os.listdir(directory_common)))
    # todo change 1330 number for ratio
    load_train_data(os.listdir(directory_common), shuffle, directory_common, 1330)
    load_val_data(os.listdir(directory_common), shuffle, directory_common)

def crop(img, x1, x2, y1, y2, scale):
    crp=img[y1:y2,x1:x2]
    crp=resize(crp,((scale, scale)))
    return crp

def getfiles(filenames, directory):
    dir_files = {}
    for x in filenames:
        dir_files[x]=io.imread(directory + x)
    return dir_files

def train_binary(train_list, data_directory):

    list_ = []
    for user in train_list:
        directory = data_directory + user + '/' + user + '_loc.csv'
        list_.append(pd.read_csv(directory, index_col=None, header=0))
    frame = pd.concat(list_)
    frame['side'] = frame['bottom_right_x']-frame['top_left_x']
    frame['hand'] = 1

    imageset = getfiles(frame.image.unique(), data_directory)

    return imageset, frame

def load_train_data(list_dir, path_array, from_directory, lower_bound):
    print('length list dir ', len(list_dir))
    print('length path array ', len(path_array))
    for i in range(0 , lower_bound):
        file = list_dir[path_array[i]]
        os.rename(from_directory + file, directory_original + train_prefix + file[0] + '/' + file)

def load_val_data(list_dir, path_array, from_directory):
    print('length list dir ', len(list_dir))
    print('length path array ', len(path_array))
    for i in range(0, len(list_dir)):
        file = list_dir[i]
        os.rename(from_directory + file, directory_original + val_prefix + file[0] + '/' + file)

def load_second_dataset():
    imageset, frame = train_binary(["user_3", "user_4", "user_5", "user_6", "user_7", "user_9", "user_10"],
                                   '../dataset_2/')
    load_dataset_2(imageset)