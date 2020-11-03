import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim

import glob
import os
from skimage.io import imread

from torch.utils import data


class DataLoader(data.Dataset):
    def __init__(self, split="training",path_to_data='/content/drive/My Drive/tmp/mnist_png_small_200_20'):
      self.split=split
      self.path=path_to_data + '/' + split 
      
      self.file_list=[]
      self.lbls=[]
#       set_trace()
      for k in range(10):
        files=glob.glob(self.path + '/'+ str(k) +'/*png')
        self.file_list.extend(files)
        self.lbls.extend([k]*len(files))
        
      self.num_of_imgs=len(self.file_list)
    
    def __len__(self):
        return self.num_of_imgs
    
    
    
    def __getitem__(self, index): 
#       set_trace()
      img=imread(self.file_list[index])
      img=torch.Tensor(img.astype(np.float32)/255-0.5)
      lbl=torch.Tensor(np.array(self.lbls[index]).astype(np.float32))
      return img,lbl
    #load and preprocess one image - with number index
    #torchvision.transforms  contains several preprocessing functions for images

    
    

loader = DataLoader(split='training')
trainloader= data.DataLoader(loader,batch_size=2, num_workers=0, shuffle=True,drop_last=True)


for it,(batch,lbls) in enumerate(trainloader): ### you can iterate over dataset (one epoch)
  print(batch)
  print(batch.size())
  print(lbls)
  plt.imshow(batch[0,:,:].detach().cpu().numpy())
  break