from msilib.schema import Class
import cv2
import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler

import os


# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 40
# percentage of training set to use as validation
valid_size = 0.2

num_of_detectors = 32
num_classes = 7
epoches = 100

path='fer_images.zip'

arr = os.listdir()

if 'fer_images' not in arr:

    zip_obj = zipfile.ZipFile(file= path, mode= 'r')
    zip_obj.extractall('./')
    zip_obj.close()

train_path = 'fer2013/train'
test_path = 'fer2013/validation'

h,w = 48,48

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device={}'.format(device))

# FIRST WE LOAD THE DATA AND DEFINE THE TRANSFORMS
train_transform = transforms.Compose([transforms.Resize((h,w)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor()])


test_transform = transforms.Compose([transforms.Resize((h,w)),
                                    transforms.ToTensor()])

train_dataset = datasets.ImageFolder(root= train_path,transform=train_transform)



test_dataset = datasets.ImageFolder(root= test_path,transform=train_transform)


# obtain training indices that will be used for validation
num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], (indices[:split])

# define samplers for obtaining training and validation batches
train_sampler =SubsetRandomSampler(train_idx) #train_dataset[train_idx]#SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)#train_dataset[valid_idx]#SubsetRandomSampler(valid_idx)


train_loader = DataLoader(dataset=train_dataset,batch_size=32 ,sampler=train_sampler)
test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)

valid_loader  = DataLoader(dataset=train_dataset,batch_size=32, sampler=valid_sampler)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3,num_of_detectors,3,padding=1)
    self.conv1_5 = nn.Conv2d(num_of_detectors,num_of_detectors,3,padding=1)
    self.bn1 = nn.BatchNorm2d(num_of_detectors)
    self.conv2 = nn.Conv2d(num_of_detectors,2*num_of_detectors,3,padding=1)
    self.conv2_5 = nn.Conv2d(2*num_of_detectors,2*num_of_detectors,3,padding=1)
    self.bn2 = nn.BatchNorm2d(2*num_of_detectors)
    self.conv3 = nn.Conv2d(2*num_of_detectors,2*4*num_of_detectors,3,padding=1)
    self.conv3_5 = nn.Conv2d(2*4*num_of_detectors,2*4*num_of_detectors,3,padding=1)
    self.bn3 = nn.BatchNorm2d(2*2*2*num_of_detectors)

    self.dropout = nn.Dropout(0.2)
    self.pool = nn.MaxPool2d(2, 2)
    self.relu = nn.ReLU(inplace=True)

    self.fc1 = nn.Linear(256*6*6,2*num_of_detectors)
    self.fc2 = nn.Linear(2*num_of_detectors,num_of_detectors)
    self.fc3 = nn.Linear(num_of_detectors,num_classes)

  def forward(self, x):
    #this is CNN layer stracture
    temp= x.shape

    x= (self.relu(self.conv1(x)))
    x=self.bn1(x)
    
    x= (self.relu(self.conv1_5(x)))
    x= self.pool(x)
    x=self.dropout(x)

    temp= x.shape

    x= (self.relu(self.conv2(x)))
    x=self.bn2(x)
    x= (self.relu(self.conv2_5(x)))
    x= self.pool(x)
    x= self.dropout(x)

    # shuff for git
    x= (self.relu(self.conv3(x)))
    x=self.bn3(x)
    x= (self.relu(self.conv3_5(x)))
    temp= x.shape
    x=self.pool(x)
    x= self.dropout(x)

    temp= x.shape

    #print('befor the view x is {}'.format(temp))

    x=  x.view(x.size(0), -1)
    temp= x.shape
    x= self.relu(self.fc1(x)) 
    temp= x.shape 
    
    x= self.relu(self.fc2(x))
    temp= x.shape
    x= self.relu(self.fc3(x))
    temp= x.shape
    #x= torch.nn.Softmax(dim=0)
    return x



model = Net()
model.to(device=device)
print(model)

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)#,weight_decay=0.001)
#################################################################
epochs = 50
steps = 0
train = 0

# number of epochs to train the model
train_losses, test_losses = [], []

train_accuracyA, accuracyA  = [], []
xaxis =[]

train_accuracy=0
for e in range(epochs):
    running_loss = 0

    train_accuracy=0

    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        log_ps = model(inputs)
        temp = log_ps.shape
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        train_accuracy += torch.mean(equals.type(torch.FloatTensor))
        
    else:
        print('reached test')
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                log_ps = model(inputs)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
                
        
        
        train_accuracy = train_accuracy/len(train_loader)
        accuracy = accuracy/len(test_loader)   

        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))
        
        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(test_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy))#/len(test_loader)))
        print("Train Accuracy: {:.3f}".format(train_accuracy))#train_accuracy/len(train_loader)))

        train_accuracyA.append(train_accuracy)
        accuracyA.append(accuracy)
        xaxis.append(e+1)

        model.train()

model_save_name = 'emotion_classification.pt'
path = F"save/{model_save_name}"

torch.save(model.state_dict(),path )
print('finished , model saved')



'''

trainning_generator = ImageDataGenerator(rescale=1./255, rotation_range= 7 , horizontal_flip= True, zoom_range= 0.2)


train_dataset = trainning_generator.flow_from_directory(directory= 'fer2013/train', target_size=(h,w),
                                                        batch_size=16, class_mode='categorical', shuffle= True)

test_generator = ImageDataGenerator(rescale=1./255)
test_dataset = test_generator.flow_from_directory(directory= 'fer2013/validation', target_size=(h,w),
                                                        batch_size=1, class_mode='categorical', shuffle= False)


#NOW WE BUILD THE NET

num_of_detectors = 32
num_classes = 7
epoches = 100

net = Sequential()
net.add((Conv2D(num_of_detectors,(3,3),activation= 'relu', padding= 'same',input_shape= (w,h,3))))
net.add(BatchNormalization())
net.add((Conv2D(num_of_detectors,(3,3),activation= 'relu', padding= 'same')))
net.add(BatchNormalization())
net.add(MaxPool2D(pool_size=(2,2)))
net.add(Dropout(0.2))

net.add((Conv2D(2*num_of_detectors,(3,3),activation= 'relu', padding= 'same')))
net.add(BatchNormalization())
net.add((Conv2D(2*num_of_detectors,(3,3),activation= 'relu', padding= 'same')))
net.add(BatchNormalization())
net.add(MaxPool2D(pool_size=(2,2)))
net.add(Dropout(0.2))

net.add((Conv2D(2*2*num_of_detectors,(3,3),activation= 'relu', padding= 'same')))
net.add(BatchNormalization())
net.add((Conv2D(2*2*num_of_detectors,(3,3),activation= 'relu', padding= 'same')))
net.add(BatchNormalization())
net.add(MaxPool2D(pool_size=(2,2)))
net.add(Dropout(0.2))

net.add((Conv2D(2*2*2*num_of_detectors,(3,3),activation= 'relu', padding= 'same')))
net.add(BatchNormalization())
net.add((Conv2D(2*2*2*num_of_detectors,(3,3),activation= 'relu', padding= 'same')))
net.add(BatchNormalization())
net.add(MaxPool2D(pool_size=(2,2)))
net.add(Dropout(0.2))


net.add(Flatten())

net.add(Dense(1156,activation= 'relu'))
net.add(BatchNormalization())
net.add(Dropout(0.2))

net.add(Dense(2*num_of_detectors,activation= 'relu'))
net.add(BatchNormalization())
net.add(Dropout(0.2))

net.add(Dense(num_classes,activation='softmax'))

net.compile(optimizer='Adam', loss= tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])

print(net.summary())
history = net.fit(train_dataset,epochs=100)


mode_json= net.to_json()
with open('net.json','w') as json_file:
  json_file.write(mode_json)

from keras.models import save_model
net_saves = save_model(net,'weights.hdf5')

'''