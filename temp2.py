# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:15:22 2024

@author: gozde
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import seaborn as sn
import pandas as pd
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time
import copy
from tabulate import tabulate
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, data_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = set()  # Initialize an empty set to store unique class names
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        data = []
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path = os.path.join(self.root_dir, line.strip())
                label = self.get_label(img_path)
                data.append((img_path, label))
        return data

    def get_label(self, img_path):
        class_name = None
        current_dir = os.path.dirname(img_path)  # Get the directory of the image
        while current_dir != self.root_dir:  # Loop until reaching the root directory
            current_dir = os.path.dirname(current_dir)  
            class_name = os.path.basename(current_dir)  
            if class_name in ["fog", "rain", "night", "clear_from_cityscapes"]:  
                self.classes.add(class_name)  
                break 
        class_to_label = {"fog": 0, "rain": 1, "night": 2, "clear_from_cityscapes": 3}
        return class_to_label.get(class_name, -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

root_dir =r'/nfsd/lttm4/tesisti/koca/datasets/ACDC/'

train_data_file = r'/nfsd/lttm4/tesisti/koca/datasets/ACDC/train.txt'
test_data_file = r'/nfsd/lttm4/tesisti/koca/datasets/ACDC/test.txt'

transforms_train = Compose([
    Resize((540, 960)), 
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transforms_test = Compose([
    Resize((540, 960)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(root_dir, train_data_file, transform=transforms_train)
test_dataset = CustomDataset(root_dir, test_data_file, transform=transforms_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))
class_names = train_dataset.classes
print('Class names:', class_names)

plt.rcParams['figure.figsize'] = [25, 20]
plt.rcParams['figure.dpi'] = 60
plt.rcParams.update({'font.size': 20})

def imshow(input, title):
    # torch.Tensor --> numpy
    input = input.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)
    plt.title(title)
    plt.show()

iterator = iter(train_dataloader)
inputs, classes = next(iterator)
out = torchvision.utils.make_grid(inputs[:4])
class_names = list(train_dataset.classes)  # Convert the set to a list
imshow(out, title=[class_names[x] for x in classes[:4]])


# Define the network architecture
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)  # 4 output classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Adding a fully-connected layer for classification
model.fc = nn.Linear(num_features, 4)
model = model.to(device)

# loss Function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def class_accuracy(model, dataloader, num_classes):
    class_correct = list(0. for _ in range(num_classes))
    class_total = list(0. for _ in range(num_classes))

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    class_accuracy = [class_correct[i] / class_total[i] * 100. if class_total[i] != 0 else 0 for i in range(num_classes)]
    return class_accuracy

train_loss = []
train_accuary = []
test_loss = []
test_accuary = []
y_pred = []
y_true = []

num_epochs = 1
start_time = time.time()

for epoch in range(num_epochs):
    print("Epoch {} running".format(epoch))
    # Training
    model.train()
    running_loss = 0.
    running_corrects = 0

    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset) * 100.
    train_loss.append(epoch_loss)
    train_accuary.append(epoch_acc)
    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time() -start_time))

    # Testing
    model.eval()
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()

            # Save predictions and true labels
            outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(outputs) # Save Prediction
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / len(test_dataset) * 100.
        test_loss.append(epoch_loss)
        test_accuary.append(epoch_acc)
        print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time()- start_time))

###
# printing accuracy and loss values for training and test
df = pd.DataFrame({'Training Accuracy': train_accuary, 'Test Accuracy': test_accuary, 'Training Loss': train_loss, 'Test Loss':test_loss})
print("Training: Selma (800), Test: Selma (400)")
print(df)

####
# class accuracies
num_classes = len(train_dataset.classes)
class_names = ['day', 'fog', 'night', 'rain']

class_acc_train = class_accuracy(model, train_dataloader, num_classes)
class_acc_test = class_accuracy(model, test_dataloader, num_classes)

train_acc_df = pd.DataFrame({'Class': class_names, 'Training Accuracy': class_acc_train, 'Testing Accuracy': class_acc_test})

for i, acc in enumerate(class_acc_train):
    print(f'Training Accuracy for {class_names[i]}: {acc:.2f}%')

for i, acc in enumerate(class_acc_test):
    print(f'Testing Accuracy for {class_names[i]}: {acc:.2f}%')

#####
# Printing accuracy and loss plots

plt.figure(figsize=(12, 6))

# accuracy
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, num_epochs+1), train_accuary, '-o')
plt.plot(np.arange(1, num_epochs+1), test_accuary, '-o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.title('Train vs Test Accuracy over time')
plt.grid(True)

# loss
plt.subplot(1, 2, 2)
plt.plot(np.arange(1, num_epochs+1), train_loss, '-o')
plt.plot(np.arange(1, num_epochs+1), test_loss, '-o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.title('Train vs Test Loss over time')
plt.grid(True)

plt.tight_layout()
plt.show()

#####
# Confusion Matrix

classes = test_dataset.classes
print("Accuracy on Training set: ",accuracy_score(y_true, y_pred))
print('Confusion matrix: \n', confusion_matrix(y_true, y_pred))
print('Classification report: \n', classification_report(y_true, y_pred))

cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
plt.figure(figsize = (7,7))
plt.title("Confusion matrix for Skin Cancer classification ")
sn.heatmap(df_cm, annot=True)



















