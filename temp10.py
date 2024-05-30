# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:42:21 2024

@author: gozde
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sn
import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, data_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_label = { 
            "fog": 0,
            "night": 1,
            "rain": 2,
            "day": 3
        }
        self.label_to_class = {v: k for k, v in self.class_to_label.items()}
        self.classes = set(self.class_to_label.keys())
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        data = []
        print(f"Loading data from: {data_file}")
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path = os.path.join(self.root_dir, line.strip().replace('/', os.sep))
                if not os.path.isfile(img_path):
                    print(f"File not found: {img_path}")
                    continue
                label = self.get_label(img_path)
                if label is not None:
                    data.append((img_path, label))
                else:
                    print(f"Class name is None for path: {img_path}")
        return data

    def get_label(self, img_path):
        # Normalize the path for consistent parsing
        img_path = img_path.replace('\\', '/')
        
        # Split the path into parts
        parts = img_path.split('/')
        
        # Check if the directory structure matches the expected format
        if len(parts) >= 4:
            class_name = parts[-3]
            if class_name in self.class_to_label:
                return self.class_to_label[class_name]
            elif class_name in ["day", "night", "fog", "rain"]:
                if class_name == "day":
                    return self.class_to_label["day"]
                elif class_name == "night":
                    return self.class_to_label["night"]
                elif class_name == "fog":
                    return self.class_to_label["fog"]
                elif class_name == "rain":
                    return self.class_to_label["rain"]
        return None

    def print_class_distribution(self):
        class_counts = {class_name: 0 for class_name in self.classes}
        for _, label in self.data:
            class_name = self.label_to_class.get(label)
            if class_name in self.classes:  # Ensure only desired classes are counted
                class_counts[class_name] += 1
        print("Class Distribution:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count}")

    def change_class_name(self, old_name, new_name):
        if old_name in self.class_to_label:
            label = self.class_to_label.pop(old_name)
            self.class_to_label[new_name] = label
            self.label_to_class[label] = new_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the root directory of your dataset
root_dir_train = r'/nfsd/lttm4/tesisti/koca/datasets/UAVID'
root_dir_test = r'/nfsd/lttm4/tesisti/koca/datasets/UAVID'

# Define the paths to your train and test data files
train_data_file = r'/nfsd/lttm4/tesisti/koca/datasets/UAVID/train.txt'
test_data_file = r'/nfsd/lttm4/tesisti/koca/datasets/UAVID/test.txt'

# Define the transformations
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

# Create datasets for training and testing
try:
    train_dataset = CustomDataset(root_dir_train, train_data_file, transform=transforms_train)
    test_dataset = CustomDataset(root_dir_test, test_data_file, transform=transforms_test)
except Exception as e:
    print(f"Error initializing dataset: {e}")

# Create data loaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))
print('Training class names:', train_dataset.classes)
print('Test class names:', test_dataset.classes)

# Print class distribution
print("\nTraining Dataset:")
train_dataset.print_class_distribution()
print("\nTest Dataset:")
test_dataset.print_class_distribution()

# Helper function to unnormalize and display images
def imshow(input, title=None):
    input = input.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.show()

# Plot a batch of training images
iterator = iter(train_dataloader)
inputs, classes = next(iterator)
out = torchvision.utils.make_grid(inputs[:4])

# Print class names for debugging
class_names = list(train_dataset.classes)
print("Class names list:", class_names)

# Ensure the titles are correct
titles = [train_dataset.label_to_class[x] for x in classes[:4].tolist()]
print("Titles:", titles)

imshow(out, title=" | ".join(titles))

###### 2
# Define the network architecture
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)  # 4 output classes

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Adding a fully-connected layer for classification
model.fc = nn.Linear(num_features, 5)
model = model.to(device)

# loss Function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  #(lr changed from 0.0001 to 0.001)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Lists to store training and testing metrics
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []
y_pred = []
y_true = []

num_epochs = 10
start_time = time.time()

for epoch in range(num_epochs):
    print("Epoch {} running".format(epoch))
    # Training
    model.train()
    running_loss = 0.0
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
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset) * 100.0
    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_acc)
    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch + 1, epoch_loss, epoch_acc, time.time() - start_time))

    # Testing
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Save predictions and true labels
            outputs = (torch.max(torch.exp(outputs), 1)[1]).cpu().numpy()  # Transfer to CPU before converting to numpy
            y_pred.extend(outputs)  # Save Prediction
            labels = labels.cpu().numpy()  # Transfer to CPU before converting to numpy
            y_true.extend(labels)  # Save Truth

        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects.double() / len(test_dataset) * 100.0
        test_loss.append(epoch_loss)
        test_accuracy.append(epoch_acc)
        print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch + 1, epoch_loss, epoch_acc, time.time() - start_time))

# Print accuracy and loss values for training and test
df = pd.DataFrame({'Training Accuracy': train_accuracy, 'Test Accuracy': test_accuracy, 'Training Loss': train_loss, 'Test Loss': test_loss})
print(df)

# Compute accuracy, confusion matrix, and classification report
print("Accuracy on Test set: ", accuracy_score(y_true, y_pred))
print('Confusion matrix: \n', confusion_matrix(y_true, y_pred))
print('Classification report: \n', classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Calculate class accuracies
class_accuracies = cm.diagonal() / cm.sum(axis=1)
for i, class_name in enumerate(class_names):
    print(f'Accuracy for {class_name}: {class_accuracies[i] * 100:.2f}%')

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
plt.savefig('acc_loss_plot_uu-16_0_001.png')  # Save the loss plot

plt.tight_layout()
plt.show()

#####

classes = test_dataset.classes
# Compute accuracy, confusion matrix, and classification report
print("Accuracy on Training set: ", accuracy_score(y_true, y_pred))
print('Confusion matrix: \n', confusion_matrix(y_true, y_pred))
print('Classification report: \n', classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

class_accuracies = cm.diagonal() / cm.sum(axis=1)
for i, class_name in enumerate(class_names):
    print(f'Accuracy for {class_name}: {class_accuracies[i]*100:.2f}%')

# Plot confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 7))  # Increase figure size for better readability
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')  # Ensure annotations are integers
plt.title('Confusion Matrix', fontsize=18)
plt.show()
plt.savefig('confusion_matrix_uavid-uavid_16_0_001.png', bbox_inches='tight')
