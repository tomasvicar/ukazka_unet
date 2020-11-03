import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim
import torchvision
from torchvision import transforms

import glob
import os
from skimage.io import imread

from torch.utils import data

from unet import Unet

batch=16

loader = DataLoader(split='train')
trainloader= torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=4, shuffle=True,drop_last=True)

  
loader = DataLoader(split='test')
testloader= torch.utils.data.DataLoader(loader,batch_size=1, num_workers=1, shuffle=True,drop_last=False)



net=Unet().cuda()

optimizer = optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-8)



train_loss=[]
test_loss=[]
train_acc=[]
test_acc=[]
position=[]
train_acc_tmp=[]
train_loss_tmp=[]
test_acc_tmp=[]
test_loss_tmp=[]

it=-1
for epoch in range(1000):
  for k,(data,lbl) in enumerate(trainloader):
    it+=1
    print(it)
    
    data=data.cuda()
    lbl=lbl.cuda()
    
    
    data.requires_grad=True
    lbl.requires_grad=True
    
    optimizer.zero_grad()   # zero the gradient buffers
    
    net.train()
    
    output=net(data)
    output=F.sigmoid(output)
    
    loss=dice_loss(output, lbl)   ### tady spočíta MSE pro denoising....
    
    loss.backward()  ## claculate gradients
    optimizer.step() ## update parametrs
    
    clas=(output>0.5).float()
    
    acc=torch.mean((clas==lbl).float())
    
    train_acc_tmp.append(acc.detach().cpu().numpy())
    train_loss_tmp.append(loss.detach().cpu().numpy())
    
    
    if it%50==0:

      d=data[0,0,:,:].detach().cpu().numpy()
      r=output[0,0,:,:].detach().cpu().numpy()
      g=lbl[0,0,:,:].detach().cpu().numpy()
      plt.imshow(np.concatenate((d,r,g),axis=1),cmap='gray',vmin=0,vmax=1)
      plt.show()

      for kk,(data,lbl) in enumerate(testloader):
        data=data.cuda()
        lbl=lbl.cuda()


        data.requires_grad=True
        lbl.requires_grad=True

        optimizer.zero_grad()   # zero the gradient buffers

        net.eval()

        output=net(data)
        output=F.sigmoid(output)

        loss=dice_loss(output, lbl)


        clas=(output>0.5).float()

        acc=torch.mean((clas==lbl).float())

        test_acc_tmp.append(acc.detach().cpu().numpy())
        test_loss_tmp.append(loss.detach().cpu().numpy())
        
        d=data[0,0,:,:].detach().cpu().numpy()
        r=output[0,0,:,:].detach().cpu().numpy()
        g=lbl[0,0,:,:].detach().cpu().numpy()
        plt.imshow(np.concatenate((d,r,g),axis=1),cmap='gray',vmin=0,vmax=1)
        plt.show()
        
      
      train_loss.append(np.mean(train_loss_tmp))
      test_loss.append(np.mean(test_loss_tmp))
      train_acc.append(np.mean(train_acc_tmp))
      test_acc.append(np.mean(test_acc_tmp))
      position.append(it)
      
      train_acc_tmp=[]
      train_loss_tmp=[]
      test_acc_tmp=[]
      test_loss_tmp=[]

      plt.plot(position,train_loss)
      plt.plot(position,test_loss)
      plt.show()
      
      plt.plot(position,train_acc)
      plt.plot(position,test_acc)
      plt.show()