import glob
import os

from torchvision.transforms import Compose,ToTensor,Resize,Normalize,RandomHorizontalFlip,RandomRotation
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class data_loader_asl():

    def __init__(self,):
        self.user_images = glob.glob('../asl_dataset/')
        self.asl_dataset = '../asl_dataset/'
        self.directory_original = '../data/'
        self.train_prefix = 'train1/'
        self.val_prefix = 'valid1/'
        self.test_prefix = 'test1/'
        self.asl_train_prefix = 'asl_alphabet_train/'
        self.asl_test_prefix = 'asl_alphabet_test/'
        self.test_dir = '../test/testdata1'

        # preparing dataset-train dataset / validation dataset
        self.transform = Compose([Resize([128, 128]), ToTensor(),
                                  Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.load_train_data(self.asl_dataset + self.asl_train_prefix, self.directory_original + self.train_prefix,
                             self.directory_original + self.val_prefix, 300)
        self.load_test_data(self.asl_dataset + self.asl_test_prefix, self.test_dir)

        self.train_dataset = ImageFolder(self.directory_original + self.train_prefix, transform=self.transform)
        self.valid_dataset = ImageFolder(self.directory_original + self.val_prefix, transform=self.transform)
        self.test_dataset = ImageFolder(self.test_dir, transform=self.transform)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=50)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=50)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=50)


    def load_train_data(self, from_directory, to_train_directory, to_val_directory, amount):

        root_dir_train = os.listdir(from_directory)

        if not os.path.exists(to_train_directory):
            os.mkdir(to_train_directory)
        if not os.path.exists(to_val_directory):
            os.mkdir(to_val_directory)

        for d in range(len(root_dir_train)):
            inner_directory_train = from_directory + root_dir_train[d]
            destination_letter_lower = root_dir_train[d].lower()
            files_names = os.listdir(inner_directory_train)
            files_names_train = files_names[amount:]
            files_names_val = files_names[:amount]

            # move train files
            if not os.path.exists(to_train_directory + destination_letter_lower):
                os.mkdir(to_train_directory + destination_letter_lower)

            for i in range(len(files_names_train)):
                pic = files_names_train[i]
                os.rename(inner_directory_train + '/' + pic, to_train_directory + destination_letter_lower + '/' + pic)

            # move validation files
            if not os.path.exists(to_val_directory + destination_letter_lower):
                os.mkdir(to_val_directory + destination_letter_lower)

            for i in range(len(files_names_val)):
                pic = files_names_val[i]
                os.rename(inner_directory_train + '/' + pic, to_val_directory + destination_letter_lower + '/' + pic)

    def load_test_data(self, from_directory, to_directory):

        root_dir_test = os.listdir(from_directory)

        if not os.path.exists(to_directory):
            os.mkdir(to_directory)

        for d in range(len(root_dir_test)):
            pic = root_dir_test[d]

            destination_letter_lower = pic[0].lower()

            # move train files
            if not os.path.exists(to_directory + '/' + destination_letter_lower):
                os.mkdir(to_directory + '/' + destination_letter_lower)
                os.rename(from_directory + pic, to_directory + '/' + destination_letter_lower + '/' + pic)



