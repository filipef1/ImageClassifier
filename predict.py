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

parser = argparse.ArgumentParser(description="Predict process")
parser.add_argument('--save_dir', action='store',  default='checkpoint.pth', help='Name and location to save checkpoint in.')
parser.add_argument('--top_k', dest='top_k', default='5', type=int, help='Number of top most likely classes, default is 5.')
parser.add_argument('--image_path', dest='image_path',default = 'flowers/test/22/image_05366.jpg', help='Path of image to predict')
parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json', help='Enter file with category names.')
parser.add_argument('--gpu', action='store_true', default=True, help='Turn GPU mode, default is on.')

results = parser.parse_args()

save_dir = results.save_dir
top_k = int(results.top_k)
cat_to_name = u.cat_names(results.category_names)
categories = len(cat_to_name)

gpu_mode = results.gpu
if gpu_mode is True and torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'

# Loading model
model, optimizer = u.load_cp(save_dir, device)
  
image_test = results.image_path

# TODO: Display an image along with the top 5 classes
prob, classes = u.predict(image_test, model, top_k, device)
max_index = np.argmax(prob)
max_probability = prob[max_index]
label = classes[max_index]
image = Image.open(image_test)

i=0
for cl in classes:
    print("Probability to be a {}: {}".format(cat_to_name[cl], prob[i]))
    i =+ 1

