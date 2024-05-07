# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:31:39 2024

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

    def __len__(self):
        return len(self.data)
    
    def get_label(self, img_path):
        return 0  

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the root directory of your dataset
root_dir = r'koca/datasets/ACDC/'

# Define the paths to your train and test data files
train_data_file = r'koca/datasets/ACDC/train.txt'
test_data_file = r'koca/datasets/ACDC/test.txt'

transforms_train = transforms.Compose([
    transforms.Resize((540, 960)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])
transforms_test = transforms.Compose([
    transforms.Resize((540, 960)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets for training and testing
train_dataset = CustomDataset(root_dir, train_data_file, transform=transforms_train)
test_dataset = CustomDataset(root_dir, test_data_file, transform=transforms_test)

# Create data loaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))

# Define the network architecture
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)  # 4 output classes

# Move model to GPU if available
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

num_epochs = 10
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

# printing accuracy and loss values for training and test
df = pd.DataFrame({'Training Accuracy': train_accuary, 'Test Accuracy': test_accuary, 'Training Loss': train_loss, 'Test Loss':test_loss})
print("Training: ACDC, Test: ACDC")
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
















