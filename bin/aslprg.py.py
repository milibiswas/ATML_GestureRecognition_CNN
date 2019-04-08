# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:36:46 2019

@author: somde
"""
import os
import numpy as np
import torch
from torchvision.transforms import Compose,ToTensor,Resize,Normalize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt


path = "D:\\project\\ATML\\try\\dataset"
path1 = "D:\\project\\ATML\\try"
path_valid = "D:\\project\\ATML\\try\\valid"
path_train = "D:\\project\\ATML\\try\\train"

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

prepare_valid_data(400,ls)   
prepare_train_data(400,ls)  

# preparing dataset-train dataset/ validation datadset
transform = Compose([Resize([128,128]),ToTensor(),Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
train_dataset = ImageFolder(path_train,transform=transform)
valid_dataset = ImageFolder(path_valid,transform=transform)
#print(valid_dataset[0])
# preparing dataloader - train dataloader /validation dataloader
train_dataloader = DataLoader(train_dataset,batch_size=100)
valid_dataloader = DataLoader(valid_dataset,batch_size=100)

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
#........................................
model = MLPModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
#--------------------------------------------------

# train/ validate the model 

def train(model,train_dataloader,loss_fn,optimizer):
    train_loss = 0
    n_correct = 0
    for (images,labels) in train_dataloader:
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
            out = model(images)
            loss = loss_fn(out,labels)
            valid_loss +=loss.item()
            n_correct += torch.sum(out.argmax(1)==labels).item()
        accuracy = 100*n_correct / len(valid_dataloader.dataset)
        average_loss = valid_loss/len(valid_dataloader)
        return average_loss,accuracy

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
        print("Epoch:{} the train loss is {} and the train accuracy is {}".format(epoch,train_loss,train_accuracy))
        print("Epoch:{} the validation loss is {} and the validation accuracy is {}".format(epoch,valid_loss,valid_accuracy))
    return train_losses , train_accuracies,valid_losses,valid_accuracies




t_losses , t_accuracies,v_losses,v_accuracies = fit(model,train_dataloader,valid_dataloader,loss_fn,optimizer,25)


##########################
#   plot function
##########################

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