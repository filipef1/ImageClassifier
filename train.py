import argparse
import os, random
import numpy as np
import time
import json
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
import utils as u


parser = argparse.ArgumentParser(description="Training process")
parser.add_argument('--save_dir', action='store',  default='checkpoint.pth', help='Name and location to save checkpoint in.')
parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'vgg19'], help='Pretrained model to use, default is VGG-16.')
parser.add_argument('--learning_rate', dest='learning_rate', default='0.001', type=float, help='Learning rate for training, default is 0.001')
parser.add_argument('--dropout', action='store', default = 0.05, type=float, help='Dropout for training, default is 0.05')
parser.add_argument('--hidden_units', dest='hidden_units', default='500', type=int, help='Number of hidden units in classifier.')
parser.add_argument('--epochs', dest='epochs', default='3', type=int, help='Number of epochs in tranning.')
parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json', help='Enter file with category names.')
parser.add_argument('--gpu', action='store_true', default=True, help='Turn GPU mode, default is on.')

results = parser.parse_args()

save_dir = results.save_dir
arch = results.arch
learning_rate = results.learning_rate
dropout = results.dropout
hidden_units = results.hidden_units
epochs = results.epochs
cat_to_name = u.cat_names(results.category_names)
categories = len(cat_to_name)

gpu_mode = results.gpu
if gpu_mode is True and torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'

train_dir = 'flowers/train'
valid_dir = 'flowers/valid'
test_dir =  'flowers/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data  = datasets.ImageFolder(test_dir,  transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=32)

image_datasets = {'train_data': train_data,
                  'test_data' : test_data,
                  'valid_data': valid_data}
         

# Creating classifier with function
model = u.classifier(arch, hidden_units, dropout, categories)

# Using NLLLoss with Softmax
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0008)

# Using function to training and validation
u.train(model, epochs, train_loader, valid_loader, optimizer, criterion, device)

model.class_to_idx =  image_datasets['train_data'].class_to_idx

# Saving Checkpoint
u.save_cp(arch, model, hidden_units, dropout, epochs, optimizer, categories, save_dir)

