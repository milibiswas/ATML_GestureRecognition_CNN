# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:36:46 2019

@author: Mili Biswas (MSc - Computer Sc.)

"""
import os
import numpy as np
import torch
from torchvision.transforms import Compose,ToTensor,Resize,Normalize,RandomHorizontalFlip,RandomRotation
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

##############################
# Type of dataset

DATASET_MESSEY=False
DATASET_KAGGLE=False

##############################

path = "../data/dataset"
path1 = "../data"
path_valid = "../data/valid"
path_train = "../data/train"


####################   Test    #########################

pathTestDataSource="../test/dataset"
pathTestDataTarget="../test/testdata"

########################################################

if os.path.exists(os.path.join(path1,'valid')) and os.path.exists(os.path.join(path1,'train')) :
    pass
else:
    os.mkdir(os.path.join(path1,'valid'))
    os.mkdir(os.path.join(path1,'train'))

# The shuffle will hold image files' indexes in random order

shuffle = np.random.permutation(len(os.listdir(path)))

ls = []

# preparing valid folder test/valid
for i in os.listdir(path):
    ls.append((i.split('_')[1],i,))

def prepare_valid_data(index,ls):
    for i in shuffle[:index]:
        if os.path.exists(os.path.join(path_valid,ls[i][0])):
            os.rename(os.path.join(path,ls[i][1]),os.path.join(path_valid,ls[i][0],ls[i][1]))
        else:
            print('ok')
            os.mkdir(os.path.join(path_valid,ls[i][0]))
            os.rename(os.path.join(path,ls[i][1]),os.path.join(path_valid,ls[i][0],ls[i][1]))

def prepare_train_data(index,ls):
    for i in shuffle[index:]:
        if os.path.exists(os.path.join(path_train,ls[i][0])):
            os.rename(os.path.join(path,ls[i][1]),os.path.join(path_train,ls[i][0],ls[i][1]))
        else:
            os.mkdir(os.path.join(path_train,ls[i][0]))
            os.rename(os.path.join(path,ls[i][1]),os.path.join(path_train,ls[i][0],ls[i][1]))

def prepare_test_data(pathTestDataSource,pathTestDataTarget):
    ls=[]
    for i in os.listdir(pathTestDataSource):
        ls.append((i.split('_')[1],i,))

    for i,j in enumerate(ls):
        if os.path.exists(os.path.join(pathTestDataTarget,ls[i][0])):
            os.rename(os.path.join(pathTestDataSource,ls[i][1]),os.path.join(pathTestDataTarget,ls[i][0],ls[i][1]))
        else:
            os.mkdir(os.path.join(pathTestDataTarget,ls[i][0]))
            os.rename(os.path.join(pathTestDataSource,ls[i][1]),os.path.join(pathTestDataTarget,ls[i][0],ls[i][1]))

prepare_valid_data(800,ls)
prepare_train_data(800,ls)
prepare_test_data(pathTestDataSource,pathTestDataTarget)

# preparing dataset-train dataset/ validation datadset
transform = Compose([Resize([128,128]),RandomHorizontalFlip(0.4),RandomRotation(0.2),ToTensor(),Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
train_dataset = ImageFolder(path_train,transform=transform)
valid_dataset = ImageFolder(path_valid,transform=transform)
#print(valid_dataset[0])
# preparing dataloader - train dataloader /validation dataloader

train_dataloader = DataLoader(train_dataset,batch_size=50)
valid_dataloader = DataLoader(valid_dataset,batch_size=50)

# preparing the model

class MLPModel(nn.Module):
    def __init__(self,):
        super(MLPModel, self).__init__()
        self.layers = nn.Sequential(nn.Linear(128*128*3,128),
        nn.LeakyReLU(0.2),
        nn.Linear(128,36))

    def forward(self,input):
        input = input.view(input.size(0),-1)
        ret=self.layers(input)
        return ret



class ConvModel(nn.Module):
    def __init__(self,):
        super(ConvModel,self).__init__()
        self.conv_layer=nn.Sequential(
                nn.Conv2d(3,64,kernel_size=3,padding=0),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25),
                nn.Conv2d(64,64,kernel_size=3,padding=0),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25),
                nn.Conv2d(64,64,kernel_size=3,padding=0),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25)
                )
        self.linear_layer=nn.Sequential(
                nn.Linear(64*12*12,128),
                nn.Dropout(0.5),
                nn.Linear(128,128),
                nn.Dropout(0.5),
                nn.Linear(128,36),
                )
    def forward(self,input):
        conv_out = self.conv_layer(input)
        conv_out_flat = conv_out.view(conv_out.size(0),-1)
        out=self.linear_layer(conv_out_flat)
        return out



#........................................
model = ConvModel()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
#--------------------------------------------------

# train/ validate the model

def train(model,train_dataloader,loss_fn,optimizer):
    train_loss = 0
    n_correct = 0
    for (images,labels) in train_dataloader:
        images=images.to(device)
        labels=labels.to(device)
        out = model(images)
        loss = loss_fn(out,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_correct += torch.sum(out.argmax(1)==labels).item()
    accuracy = (100*n_correct)/len(train_dataloader.dataset)
    average_loss = train_loss/len(train_dataloader)
    return average_loss,accuracy

def valid(model,valid_dataloader,loss_fn):
    valid_loss = 0
    n_correct = 0
    with torch.no_grad():
        for (images,labels) in valid_dataloader:
            images=images.to(device)
            labels=labels.to(device)
            out = model(images)
            loss = loss_fn(out,labels)
            valid_loss +=loss.item()
            n_correct += torch.sum(out.argmax(1)==labels).item()
        accuracy = 100*n_correct / len(valid_dataloader.dataset)
        average_loss = valid_loss/len(valid_dataloader)
        return average_loss,accuracy

def test(model,test_dataloader):
    n_correct = 0
    with torch.no_grad():
        for (images,labels) in test_dataloader:
            images=images.to(device)
            labels=labels.to(device)
            out = model(images)
            n_correct += torch.sum(out.argmax(1)==labels).item()
        accuracy = 100*n_correct / len(test_dataloader.dataset)
        return accuracy

def fit(model,train_dataloader,valid_dataloader,loss_fn,optimizer,n_epoch):
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    for epoch in range(n_epoch):
        train_loss,train_accuracy = train(model,train_dataloader,loss_fn,optimizer)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_loss,valid_accuracy = valid(model,valid_dataloader,loss_fn)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        print("Epoch:{}, train loss:{}, train accuracy:{}, validation loss:{}, validation accuracy:{}, loss diff:{}".format(epoch,train_loss,train_accuracy,valid_loss,valid_accuracy,abs(train_loss-valid_loss)))
        #print("Epoch:{} the validation loss is {} and the validation accuracy is {}".format(epoch,valid_loss,valid_accuracy))
    return train_losses , train_accuracies,valid_losses,valid_accuracies




t_losses , t_accuracies,v_losses,v_accuracies = fit(model,train_dataloader,valid_dataloader,loss_fn,optimizer,25)


##########################
#   plot function
##########################
'''
def plot (x_axis,y_axis,plotName=None):
    if plotName=="scatter":
        plt.scatter(x_axis,y_axis)
        plt.show()
    else:
        plt.plot(x_axis,y_axis)
        plt.show()

plot(range(len(t_losses)),t_losses)
plot(range(len(t_accuracies)),t_accuracies)

plot(range(len(v_losses)),t_losses)
plot(range(len(v_accuracies)),t_accuracies)
'''
#######################################
#     Test phase
#######################################

#  Data Loader (Test)
path_test = '../test/testdata'

test_dataset = ImageFolder(path_test,transform=transform)
test_dataloader = DataLoader(test_dataset,batch_size=100,shuffle=True)

test_accuracy=test(model,test_dataloader)

print("======================== Test Accurracy Score ========================")
print("Accuracy Score : ",test_accuracy)
print("========================        End           ========================")