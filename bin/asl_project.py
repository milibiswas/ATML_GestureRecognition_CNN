# -*- coding: utf-8 -*-
"""
Name : asl_project.py
Created By: Mili Biswas
Date: 18-May-2019

Dept of Computer Scienece, University of Bern, Switzerland

"""

from torch.autograd import Variable
import numpy as np
from torchvision import models
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import data_loader_massey as dlm
import data_loader_kaggle as dlk

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#####################################################################################
#  Activate the dataset you want to use & Type of model (e.g. Pretrained or not)
#

DATASET_MESSEY=True
VGG_PRETRAINED=False


'''
    DO NOT Change the below parameters. Otherwise, dimension mismatch could occur!
'''

if not DATASET_MESSEY:
  LINEAR_LAYER_FINAL_DIM=26
else:
  LINEAR_LAYER_FINAL_DIM=36
  
if not VGG_PRETRAINED:
  LINEAR_LAYER_INITIAL_DIM=64*12*12
else:
  LINEAR_LAYER_INITIAL_DIM=256*32*32


batchNorm=[]

def getBatchNorm(e,model):
  total_norm=0
  for p in model.parameters():
    param_norm = p.grad.data.norm(2)
  total_norm += param_norm.item() ** 2
  total_norm = total_norm ** (1. / 2)
  batchNorm.append(total_norm)
  print("Epoch{} the batch norm is : {}".format(e,total_norm))

# preparing the model


class __dataset(object):
    def __init__(self,feat,labels):
        self.conv_feat=feat
        self.labels=labels
    def __len__(self,):
        return len(self.conv_feat)
    def __getitem__(self,idx):
        return self.conv_feat[idx],self.labels[idx]

def preconvfeat(dataset,model):
    conv_features=[]
    labels_lst=[]
    avgPool=nn.AdaptiveAvgPool2d((7,7))
    for data in dataset:
        images,labels = data
        images=images.to(device)
        labels=labels.to(device)
        images=Variable(images)
        labels=Variable(labels)
        output=model(images)
        output=avgPool(output)
        conv_features.extend(output)
        labels_lst.extend(labels)
    conv_features=torch.stack(conv_features)
    labels=torch.stack(labels_lst)
    return (conv_features,labels_lst)


class ConvModel(nn.Module):
    def __init__(self,):
        super(ConvModel,self).__init__()
        self.conv_layer=nn.Sequential(
                nn.Conv2d(3,16,kernel_size=3,padding=0),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
                nn.Conv2d(16,16,kernel_size=3,padding=0),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25),
                nn.Conv2d(16,32,kernel_size=3,padding=0),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.Conv2d(32,32,kernel_size=3,padding=0),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25),
                nn.Conv2d(32,64,kernel_size=3,padding=0),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,padding=0),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25)
                )
        self.linear_layer=nn.Sequential(
                nn.Linear(LINEAR_LAYER_INITIAL_DIM,128),
                nn.Dropout(0.5),
                nn.Linear(128,128),
                nn.Dropout(0.5),
                nn.Linear(128,LINEAR_LAYER_FINAL_DIM)
                )
    def forward(self,input):
        conv_out = self.conv_layer(input)
        conv_out_flat = conv_out.view(conv_out.size(0),-1)
        out=self.linear_layer(conv_out_flat)
        return out

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

def fit(model,train_dataloader,valid_dataloader,loss_fn,optimizer,n_epoch,dataObj):
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    for kfold in range(1):
        for epoch in range(n_epoch):
            train_loss,train_accuracy = train(model,train_dataloader,loss_fn,optimizer)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            valid_loss,valid_accuracy = valid(model,valid_dataloader,loss_fn)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)
            print("Epoch:{}, train loss:{}, train accuracy:{}, validation loss:{}, validation accuracy:{}, loss diff:{}".format(epoch,train_loss,train_accuracy,valid_loss,valid_accuracy,abs(train_loss-valid_loss)))
            #print("Epoch:{} the validation loss is {} and the validation accuracy is {}".format(epoch,valid_loss,valid_accuracy))
            getBatchNorm(epoch,model)
    return train_losses , train_accuracies,valid_losses,valid_accuracies
  
  
  
def plot(n_epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure()
    plt.plot(np.arange(n_epochs), train_losses)
    plt.plot(np.arange(n_epochs), val_losses)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('Train/val loss');

    plt.figure()
    plt.plot(np.arange(n_epochs), train_accuracies)
    plt.plot(np.arange(n_epochs), val_accuracies)
    plt.legend(['train_acc', 'val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Train/val accuracy')
    plt.savefig("train_valid.png")
    
def plot_batch_norm(n_epochs, batch_norm):
    plt.figure()
    plt.scatter(range(n_epochs), batch_norm)
    plt.legend(['Batch Norm Value'])
    plt.xlabel('epoch')
    plt.ylabel('batch norm value')
    plt.title('Batch norm changes')
    plt.savefig("batch_norm.png")

# %matplotlib inline
if not DATASET_MESSEY:
  
  if not VGG_PRETRAINED:
        d=dlk.data_loader_kaggle()
        train_dataloader=d.train_dataloader
        valid_dataloader=d.valid_dataloader
        
        #........................................
        #  Model, loss function & optimizer
        #----------------------------------------
        
        model = ConvModel()
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)
        optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
        
        #----------------------------------------
        #  Fit function calls
        #----------------------------------------
        
        t_losses , t_accuracies,v_losses,v_accuracies = fit(model,train_dataloader,valid_dataloader,loss_fn,optimizer,200,d)
        
        #----------------------------------------
        #  Testing & Accuracy Calculation
        #----------------------------------------
        
        test_dataloader = d.test_dataloader
        test_accuracy=test(model,test_dataloader)
        print("======================== Test Accurracy Score ========================")
        print("Accuracy Score : ",test_accuracy)
        print("========================        End           ========================")
        
        plot(200, t_losses, v_losses, t_accuracies, v_accuracies)
        plot_batch_norm(200,batchNorm)
        
  if VGG_PRETRAINED:
    
    d=dlk.data_loader_kaggle()
    train_dataloader=d.train_dataloader
    valid_dataloader=d.valid_dataloader
    
    #----------------------------------------
    #  Model, loss function & optimizer
    #----------------------------------------
    
    vgg=models.vgg16(pretrained=True)

    vggConvNet=nn.Sequential(*list(vgg.features.children())[0:14])
    for params in vgg.parameters():
        params.requires_grad=True
    model = ConvModel()
    model.conv_layer=vggConvNet
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)
    optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
    
    #----------------------------------------
    #  Fit function calls
    #----------------------------------------
    
    t_losses , t_accuracies,v_losses,v_accuracies = fit(model,train_dataloader,valid_dataloader,loss_fn,optimizer,100,d)
    
    #----------------------------------------
    #  Testing & Accuracy Calculation
    #----------------------------------------
   
    test_dataloader = d.test_dataloader
    test_accuracy=test(model,test_dataloader)
    print("======================== Test Accurracy Score ========================")
    print("Accuracy Score : ",test_accuracy)
    print("========================        End           ========================")
    
    plot(100, t_losses, v_losses, t_accuracies, v_accuracies)
    plot_batch_norm(100,batchNorm)
    
if DATASET_MESSEY:
  
  if not VGG_PRETRAINED:
        d=dlm.data_loader_messey()
        train_dataloader=d.train_dataloader
        valid_dataloader=d.valid_dataloader
        
        #----------------------------------------
        #  Model, loss function & optimizer
        #----------------------------------------
        model = ConvModel()
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)
        optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
        #----------------------------------------
        #  Fit function calls
        #----------------------------------------
        t_losses , t_accuracies,v_losses,v_accuracies = fit(model,train_dataloader,valid_dataloader,loss_fn,optimizer,200,d)
        
        #----------------------------------------
        #  Testing & Accuracy Calculation
        #----------------------------------------
        
        test_dataloader = d.test_dataloader
        test_accuracy=test(model,test_dataloader)
        print("======================== Test Accurracy Score ========================")
        print("Accuracy Score : ",test_accuracy)
        print("========================        End           ========================")
        
        plot(200, t_losses, v_losses, t_accuracies, v_accuracies)
        plot_batch_norm(200,batchNorm)
        
  if VGG_PRETRAINED:
    
    d=dlm.data_loader_messey()
    train_dataloader=d.train_dataloader
    valid_dataloader=d.valid_dataloader
    
    #----------------------------------------
    #  Model, loss function & optimizer
    #----------------------------------------
    
    vgg=models.vgg16(pretrained=True)

    vggConvNet=nn.Sequential(*list(vgg.features.children())[0:14])
    for params in vgg.parameters():
        params.requires_grad=True
    model = ConvModel()
    model.conv_layer=vggConvNet
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)
    optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
    
    #----------------------------------------
    #  Fit function calls
    #----------------------------------------
    
    t_losses , t_accuracies,v_losses,v_accuracies = fit(model,train_dataloader,valid_dataloader,loss_fn,optimizer,100,d)
    
    #----------------------------------------
    #  Testing & Accuracy Calculation
    #----------------------------------------
   
    test_dataloader = d.test_dataloader
    test_accuracy=test(model,test_dataloader)
    print("======================== Test Accurracy Score ========================")
    print("Accuracy Score : ",test_accuracy)
    print("========================        End           ========================")
    
    plot(100, t_losses, v_losses, t_accuracies, v_accuracies)
    plot_batch_norm(100,batchNorm)

#########################  Visualization of Intermediate Layer #########################
d=None
if DATASET_MESSEY:
  d=dlm.data_loader_messey()
else:
  d=dlk.data_loader_kaggle()
plt.figure()  
plt.imshow(transforms.ToPILImage()(d.train_dataset[0][0].cpu()), cmap='gray')  
train_dataloader=d.train_dataloader
class LayerActivations():
  features=None
  def __init__(self, model, layer_num):
    self.hook=model[layer_num].register_forward_hook(self.hook_fn)
    
  def hook_fn(self,module,input,output):
    self.features=output.cpu()
  def remove(self):
    self.hook.remove()
  
  
conv_out= LayerActivations(model.conv_layer,0)

for (images,labels) in train_dataloader:
  images=images.to(device)
  pic=images[0:1,:,:,:]
  model(pic)
  plt.figure() 
  plt.imshow(transforms.ToPILImage()(pic[0].cpu()), cmap='gray')
  break
       
conv_out.remove()
act=conv_out.features

from torchvision import transforms

fig=plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0,right=1,bottom=0,top=0.8,hspace=0,wspace=0.2)
for i in range(16):
  ax=fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
  ax.imshow(transforms.ToPILImage()(act[0][i]))

################################## Weight Visualization ######################################

model.state_dict().keys()
cnn_weights=model.state_dict()['conv_layer.0.weight'].cpu()


fig=plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0,right=1,bottom=0,top=0.8,hspace=0,wspace=0.2)
for i in range(16):
  for j in range(3):
    ax=fig.add_subplot(8,12,i+j+1,xticks=[],yticks=[])
    ax.imshow(transforms.ToPILImage()(cnn_weights[i][j]))

