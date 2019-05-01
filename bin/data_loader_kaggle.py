import glob
import os
import uuid

import numpy as np
import pandas as pd
from skimage import io
from skimage.io import imsave
from skimage.transform import resize
import shutil as sh
from torchvision.transforms import Compose,ToTensor,Resize,Normalize,RandomHorizontalFlip,RandomRotation
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class data_loader_kaggle():
    
    def __init__(self,):
        self.user_images = glob.glob('../dataset_2')
        self.directory_common = '../dataset_2/common/'
        self.directory_original = '../data/'
        self.train_prefix = 'train/'
        self.val_prefix = 'valid/'
        self.load_second_dataset()
        
        
        # preparing dataset-train dataset/ validation datadset
        self.transform = Compose([Resize([128,128]),RandomHorizontalFlip(0.4),RandomRotation(0.2),ToTensor(),Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
        self.train_dataset = ImageFolder(os.path.join(self.directory_original,'train'),transform=self.transform)
        self.valid_dataset = ImageFolder(os.path.join(self.directory_original,'valid'),transform=self.transform)
        #self.test_dataset=ImageFolder(self.pathTestDataTarget,transform=self.transform)


        # preparing dataloader - train dataloader /validation dataloader
        
        self.train_dataloader = DataLoader(self.train_dataset,batch_size=50)
        self.valid_dataloader = DataLoader(self. valid_dataset,batch_size=50)
        #self.test_dataloader = DataLoader(self. test_dataset,batch_size=50)
        
        ################ Removing temporary paths #######################
        
        sh.rmtree(self.directory_common)
        
        
        

    def load_dataset_2(self,imageset):
        if not os.path.exists(self.directory_common):
            os.mkdir(self.directory_common)
        else:
            print("Directory exists, deleting and recreating ....")
            sh.rmtree(self.directory_common)
            os.mkdir(self.directory_common)
        
        if os.path.exists(os.path.join(self.directory_original,'valid')):
            sh.rmtree(os.path.join(self.directory_original,'valid'))
            
        if os.path.exists(os.path.join(self.directory_original,'train')):
            sh.rmtree(os.path.join(self.directory_original,'train'))
            
        os.mkdir(os.path.join(self.directory_original,'valid'))
        os.mkdir(os.path.join(self.directory_original,'train'))
    
        for user in ["user_3", "user_4", "user_5", "user_6", "user_7", "user_9", "user_10"]:
            boundingbox_df = pd.read_csv('../dataset_2/' + user + '/' + user + '_loc.csv')
            print('Saving data in common dir')
    
            for rows in boundingbox_df.iterrows():
                cropped_img = self.crop(imageset[rows[1]['image']],
                                   rows[1]['top_left_x'],
                                   rows[1]['bottom_right_x'],
                                   rows[1]['top_left_y'],
                                   rows[1]['bottom_right_y'],
                                   128
                                   )
                random_postfix = uuid.uuid4().hex
                lowercase_name = rows[1]['image'].split('/')[1].lower().split('.')[0] + '_' + random_postfix + '.jpg'
                image_name_with_extention = lowercase_name.replace('.jpg', '.png')
    
                imsave(self.directory_common + image_name_with_extention, cropped_img)
    
        print('Data saved in common dir')
        print('Dividing into train and val set')
        shuffle = np.random.permutation(len(os.listdir(self.directory_common)))
        # todo change 1330 number for ratio
        self.load_train_data(os.listdir(self.directory_common), shuffle, self.directory_common, 1330)
        self.load_val_data(os.listdir(self.directory_common), shuffle, self.directory_common)
    
    def crop(self,img, x1, x2, y1, y2, scale):
        crp=img[y1:y2,x1:x2]
        crp=resize(crp,((scale, scale)))
        return crp
    
    def getfiles(self,filenames, directory):
        dir_files = {}
        for x in filenames:
            dir_files[x]=io.imread(directory + x)
        return dir_files
    
    def train_binary(self,train_list, data_directory):
    
        list_ = []
        for user in train_list:
            directory = data_directory + user + '/' + user + '_loc.csv'
            list_.append(pd.read_csv(directory, index_col=None, header=0))
        frame = pd.concat(list_)
        frame['side'] = frame['bottom_right_x']-frame['top_left_x']
        frame['hand'] = 1
    
        imageset = self.getfiles(frame.image.unique(), data_directory)
    
        return imageset, frame
    
    def load_train_data(self,list_dir, path_array, from_directory, lower_bound):
        print('length list dir ', len(list_dir))
        print('length path array ', len(path_array))
        for i in range(0 , lower_bound):
            file = list_dir[path_array[i]]
            
            if os.path.exists(os.path.join(self.directory_original,self.train_prefix,file[0])):
                os.rename(from_directory + file, self.directory_original + self.train_prefix + file[0] + '/' + file)
            else:
                os.mkdir(os.path.join(self.directory_original,self.train_prefix,file[0]))
                os.rename(from_directory + file, self.directory_original + self.train_prefix + file[0] + '/' + file)
                
    
    def load_val_data(self,list_dir, path_array, from_directory):
        print('length list dir ', len(list_dir))
        print('length path array ', len(path_array))
        for i in range(0, len(list_dir)):
            file = list_dir[i]
            
            if os.path.exists(os.path.join(self.directory_original,self.val_prefix,file[0])):
                os.rename(from_directory + file, self.directory_original + self.val_prefix + file[0] + '/' + file)
            else:
                os.mkdir(os.path.join(self.directory_original,self.val_prefix,file[0]))
                os.rename(from_directory + file, self.directory_original + self.val_prefix + file[0] + '/' + file)
                    
    def load_second_dataset(self,):
        imageset, frame = self.train_binary(["user_3", "user_4", "user_5", "user_6", "user_7", "user_9", "user_10"],
                                       '../dataset_2/')
        self.load_dataset_2(imageset)
    