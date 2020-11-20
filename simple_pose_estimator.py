#import cv2 # openCV for python
#import dlib # dlib library => facial landmark library
import numpy as np
import pickle as pkl
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the data from the pkl file
x, y = pkl.load(open('data/samples.pkl', 'rb'))

# train and validation set
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.025, random_state=42)

# scale the madness
std = StandardScaler()
std.fit(x_train)
x_train = std.transform(x_train)
x_val = std.transform(x_val)

# Hyperparameters
batch_size = 64
epochs = 1000000

class Model(nn.Module):
    def __init__(self, input_dim=int(67*68/2),output_dim=3):
        super(Model,self).__init__()
        self.init_fc = nn.Linear(input_dim,int(input_dim/2))
        self.conv1 = nn.Conv2d(1,6,3,padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6,16,3,padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,8,4,padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8,1,3,padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(1)
        self.fc2 = nn.Linear(18,10)
        self.fc3 = nn.Linear(10,output_dim)
        self.d = nn.Dropout(0.2)
        self.rels = nn.LeakyReLU(0.3)
        self.input_dim = input_dim

    def forward(self,x):
        x = self.rels(self.init_fc(x))
        x = x.view(x.size()[0],1,int(x.size()[1]/67),67)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        #x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.rels(self.conv4(x))
        x = self.bn5(x)
        #print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        #print(x.size())
        x = self.d(self.rels(self.fc2(x)))
        #print(x.size())
        x = self.fc3(x)
        return x.to(device)


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def criterion(a,b):
    loss = nn.MSELoss()
    return loss(a,b)

def train(x_train,y_train,batch_size):
    e = 0 # number of epochs to count
    model = Model().to(device) # call the model
    opt = optim.Adam(model.parameters(),lr=0.01) # optimizer
    log_path = "./logs/logs.csv"
    if os.path.exists(log_path):
        os.remove(log_path)
    f = open(log_path,"w+")
    step = 0
    f.write("step , loss \n")

    while e < epochs:
        x = torch.FloatTensor(x_train)
        y = torch.FloatTensor(y_train)
        x = torch.split(x,batch_size)
        y = torch.split(y,batch_size)
        z = zip(x,y)
        loss_val = 0.0

        if e == 100:
            scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.33)

        for (x_comp, y_comp) in z:
            x_comp = x_comp.to(device)
            y_comp = y_comp.to(device)
            y_pred = model(x_comp)
            loss_val = criterion(y_pred, y_comp)
            opt.zero_grad()
            loss_val.backward()
            opt.step()
            if e > 100:
                scheduler.step()
            step += 1
            if e%10==0:
                string_save = "%i , %.2f\n" %(step,loss_val)
                f.write(string_save)

        # VERBOSE....
        if e%50==0:
            string_out = "After %i epochs, the loss is found as  %.2f" %(e,loss_val)
            print(string_out)
            # now onto the validation
            validation_x = torch.split(torch.FloatTensor(x_val),batch_size)
            validation_y = torch.split(torch.FloatTensor(y_val),batch_size)
            val_z = zip(validation_x, validation_y)
            val_loss = 0.0
            counter = 0.0
            for (xv, yv) in val_z:
                xv = xv.to(device)
                y_pred_val = model(xv).detach()
                yv = yv.to(device)
                #loss = criterion()
                validation_loss = criterion(y_pred_val,yv)
                val_loss += validation_loss
                counter += 1.0
            # now we have the validation loss
            val_loss = val_loss / counter
            print("The validation loss is: ", format(validation_loss, ".2f"))
            # now to save the model...
            if e%100 == 0:
                path = "./models/epoch_%i.pt" %(e)
                torch.save({
                    'epoch' : e,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': loss_val},path)

        e += 1

# run the main train module!
if __name__ == "__main__":
    train(x_train,y_train, batch_size)
