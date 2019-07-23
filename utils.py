import torch
from torchvision import transforms, datasets
import json
import copy
import os, random
import argparse
import numpy as np
import time
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict

        
def save_cp(arch, model, hidden_units, dropout, epochs, optimizer, categories, save_dir):
    print("\nSaving checkpoint...")
    try:
        checkpoint = {'structure': arch,
                    'hidden_units' : hidden_units,
                    'dropout' : dropout,
                    'epochs': epochs,
                    'optimizer': optimizer.state_dict(),
                    'categories': categories,
                    'state_dict': model.state_dict(),
                    'class_to_idx': model.class_to_idx}

        torch.save(checkpoint, save_dir)
        print("Saved successfully")
    except:
        print("Failed to save checkpoint")

    
    
def load_cp(filename, device):
    print("Loading checkpoint...")
    try:
        checkpoint = torch.load(filename, map_location=device)
        arch = checkpoint['structure']
        hidden_units = checkpoint['hidden_units']
        dropout  = checkpoint['dropout']
        categories = checkpoint['categories']
        model = classifier(arch, hidden_units, dropout, categories)
        model.load_state_dict(checkpoint['state_dict'])
        model.epochs = checkpoint['epochs']
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = checkpoint['optimizer']
        print("Loaded successfully\n")
            
        return model, optimizer
    
    
    except:
        print("Failed to load checkpoint")


def cat_names(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name 


def classifier(arch, hidden_units, dropout, categories):
    if arch=='vgg16':
        model = models.vgg16(pretrained=True)
    elif arch=='vgg19':
        model = models.vgg19(pretrained=True)

    imp = model.classifier[0].in_features
    hid = hidden_units
    drp = dropout
    cat = categories

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(imp, hid)),
                                            ('relu', nn.ReLU()),
                                            ('dropout1', nn.Dropout(drp)),
                                            ('fc2', nn.Linear(hid, cat)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    
    return model


def validation(model, test_loader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy
        
def train(model, epochs, train_loader, valid_loader, optimizer, criterion, device):
    print('Training...\n')
    t_since = time.time()
    steps = 0
    print_every = 40
    model.to(device)

    for e in range(int(epochs)):
        since = time.time()
        running_loss = 0
        
        # Iterating over data to carry out training step
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
                      
            # zeroing parameter gradients
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Carrying out validation step
            if steps % print_every == 0:
                # setting model to evaluation mode during validation
                model.eval()
                
                # Gradients are turned off as no longer in training
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion, device)
                
                print("Epoch: {}/{} | ".format(e+1, epochs),
                    "Training Loss: {:.4f} | ".format(running_loss/print_every),
                    "Validation Loss: {:.4f} | ".format(valid_loss/len(valid_loader)),
                    "Validation Accuracy: {:.4f}".format(accuracy/len(valid_loader)))
                
                running_loss = 0
                
                # Turning training back on
                model.train()
                
        time_taken = time.time() - since
        print("Time for epoch: {:.0f}m {:.0f}s\n".format(time_taken // 60, time_taken % 60))
        
    t_time = time.time() - t_since
    print("Total time: {:.0f}m {:.0f}s".format(t_time // 60, t_time % 60))


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
   
    process = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ])
    
    img_p = process(img)
    return img_p


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()

    if device == 'cpu':
        model.cpu()
    
    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        output = model.forward(Variable(image))
    
    prob, labels = torch.topk(output, topk)
    prob = prob.exp()
    prob_array = prob.data.numpy()[0]
    
    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    
    labels_data = labels.data.numpy()
    labels_list = labels_data[0].tolist()  
    
    classes = [inv_class_to_idx[x] for x in labels_list]
    
    return prob_array, classes    