# ATML_GestureRecognition_CNN


## Project description

Gesture Recognition in American Sign Language using Deep Convolution Networks. In this project, we implement a method for using deep convolutional networks to classify images of both the the letters and digits​ ​in​ ​American​ ​Sign​ ​Language. We have followed the paper "Using​ ​Deep​ ​Convolutional​ ​Networks​ ​for  Gesture​ ​Recognition​ ​
in​ ​American​ ​Sign​ ​Language" by Vivek​ ​Bheda​​ ​and​​ ​​N.​ ​Dianna​ ​Radpour, Department​ ​of​ ​Computer​ ​Science,​ ​Department​ ​of​ ​Linguistics State​ ​University​ ​of​ ​New​ ​York​ ​at​ ​Buffalo.  

## Getting the dataset

"Using​ ​Deep​ ​Convolutional​ ​Networks​ ​for  Gesture​ ​Recognition​ ​in​ ​American​ ​Sign​ ​Language"
- Download dataset from the following link and unpack everything into a folder /dataset
http://www.massey.ac.nz/~albarcza/gesture_dataset2012.html

"sign language and static-gesture recognition using scikit-learn"
- Download dataset from the following link and unpack everything into a folder /dataset_2
https://medium.freecodecamp.org/weekend-projects-sign-language-and-static-gesture-recognition-using-scikit-learn-60813d600e79

## Folder Structure

1. bin --> This folder contains all the python scripts related to the project.
2. data --> This folder contains all the datasets as well as container for train and validation dynamically.
3. lib --> This folder contains external libraries (Open source APIs etc.)
4. test --> This folder contains test data.
5. doc --> This folder contains documents related to the project or relevant research papers used.


## Installation Procedure

### Step 1: First clone this reporistory
### Step 2: run the below statement from command line
            python ./bin/asl_project.py
            
### Otherwise, just use the below jupiter notebook and trigger it.
            report.ipynb


### *Note: We ran the original model by commenting the batch norm layers in ConvModel & changed the optimizer to SGD. The original model is not stable with both the datasets (Massey & Kaggle) that we used*
