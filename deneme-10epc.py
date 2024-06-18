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
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import copy
from tabulate import tabulate
from torch.utils.data import ConcatDataset

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, data_file, transform=None, subset_size=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_label = { 
            "fog": 0,
            "night": 1,
            "rain": 2,
            "day": 3
        }
        self.label_to_class = {v: k for k, v in self.class_to_label.items()}
        self.classes = list(self.class_to_label.keys())
        self.data = self.load_data(data_file)

        if subset_size is not None:
            self.data = self.data[:subset_size]
    
    def load_data(self, data_file):
        data = []
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
        img_path = img_path.replace('\\', '/')
        parts = img_path.split('/')
        class_name = None
        for part in parts:
            if 'Opt_' in part:
                class_name = part.split('_')[2]
                break
        if 'clear_from_cityscapes' in parts:
            class_name = 'clear_from_cityscapes'
        elif 'day' in parts or 'ClearNoon' in parts:
            class_name = 'day'
        elif 'fog' in parts or 'MidFoggyNoon' in parts:
            class_name = 'fog'
        elif 'night' in parts or 'ClearNight' in parts:
            class_name = 'night'
        elif 'rain' in parts or 'HardRainNoon' in parts:
            class_name = 'rain'
        if class_name in ["ClearNoon", "day", "clear_from_cityscapes"]:
            class_name = "day"
        elif class_name in ["MidFoggyNoon", "fog"]:
            class_name = "fog"
        elif class_name in ["ClearNight", "night"]:
            class_name = "night"
        elif class_name in ["HardRainNoon", "rain"]:
            class_name = "rain"
        if class_name is None:
            print(f"Class name is None for path: {img_path}")
        return self.class_to_label.get(class_name, None)

    def print_class_distribution(self):
        class_counts = {class_name: 0 for class_name in self.classes}
        for _, label in self.data:
            class_name = self.label_to_class.get(label)
            if class_name in self.classes:  # Ensure only desired classes are counted
                class_counts[class_name] += 1
        print("Class Distribution:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count}")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

oot_dirs_train = [r'/nfsd/lttm4/tesisti/koca/datasets/ACDC',
                   r'/nfsd/lttm4/tesisti/koca/datasets/UAVID',
                   r'/nfsd/lttm4/tesisti/koca/datasets/SELMA',
                   r'/nfsd/lttm4/tesisti/koca/datasets/Syndrone'
]
root_dirs_test = [r'/nfsd/lttm4/tesisti/koca/datasets/ACDC']

# Define the paths to your train and test data files
train_data_files = [r'/nfsd/lttm4/tesisti/koca/datasets/ACDC/train.txt', 
                    r'/nfsd/lttm4/tesisti/koca/datasets/UAVID/train.txt',
                    r'/nfsd/lttm4/tesisti/koca/datasets/SELMA/train.txt',
                    r'/nfsd/lttm4/tesisti/koca/datasets/Syndrone/train.txt'
]
test_data_files = [r'/nfsd/lttm4/tesisti/koca/datasets/ACDC/test.txt']

transform = transforms.Compose([
    transforms.Resize((540, 960)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

subset_size = 100  # Specify the number of samples you want to use

# Create datasets and data loaders
train_datasets = [CustomDataset(root_dir, data_file, transform=transform, subset_size=subset_size) for root_dir, data_file in zip(root_dirs_train, train_data_files)]
combined_train_dataset = ConcatDataset(train_datasets)

test_datasets = [CustomDataset(root_dir, data_file, transform=transform, subset_size=subset_size) for root_dir, data_file in zip(root_dirs_test, test_data_files)]
combined_test_dataset = ConcatDataset(test_datasets)

train_loader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(combined_test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Print dataset sizes
print('Train dataset size:', len(combined_train_dataset))
print('Test dataset size:', len(combined_test_dataset))

# Get classes from the first dataset and convert to lists
train_classes = list(combined_train_dataset.datasets[0].classes)
test_classes = list(combined_test_dataset.datasets[0].classes)

print('Training class names:', train_classes)
print('Test class names:', test_classes)

# Function to print class distribution for combined dataset
def print_combined_class_distribution(combined_dataset):
    combined_class_counts = {class_name: 0 for class_name in combined_dataset.datasets[0].classes}
    for dataset in combined_dataset.datasets:
        class_counts = {class_name: 0 for class_name in dataset.classes}
        for _, label in dataset:
            class_name = dataset.label_to_class[label]
            class_counts[class_name] += 1
        for class_name, count in class_counts.items():
            combined_class_counts[class_name] += count

    print("Class Distribution:")
    for class_name, count in combined_class_counts.items():
        print(f"{class_name}: {count}")

# Print class distribution for the combined train and test datasets
print("\nTraining Dataset:")
print_combined_class_distribution(combined_train_dataset)
print("\nTest Dataset:")
print_combined_class_distribution(combined_test_dataset)

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
iterator = iter(train_loader)
inputs, classes = next(iterator)

print("Inputs shape:", inputs.shape)
print("Classes shape:", classes.shape)
print("Classes:", classes.tolist())

out = torchvision.utils.make_grid(inputs[:4])

# Get the class names
train_classes = train_datasets[0].classes  # Assuming all datasets have the same classes

# Print class names for debugging
print("Class names list:", train_classes)

# Ensure the titles are correct
titles = [train_classes[x] for x in classes[:4].tolist()]
print("Titles:", titles)

imshow(out, title=" | ".join(titles))

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify the final layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

# Move the model to the appropriate device
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_losses.append(running_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}")
    
    # Validation loop
    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_losses.append(val_running_loss / len(test_loader))
    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)
    print(f"Validation Loss: {val_running_loss / len(test_loader)}, Accuracy: {accuracy}%")

print('Finished Training')

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('loss_plot_all-deneme.png')  # Save the loss plot

# Plot validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
plt.savefig('acc_plot_all-deneme.png')  # Save the loss plot

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[train_dataset.label_to_class[i] for i in range(4)])
disp.plot(cmap=plt.cm.Blues)
plt.show()
plt.savefig('confusion_matrix_all-deneme.png', bbox_inches='tight')
