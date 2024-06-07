import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sn
import pandas as pd
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import copy
from tabulate import tabulate
from torch.utils.data import ConcatDataset

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
        self.classes = list(self.class_to_label.keys())  # Store classes as list
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
        img_path = img_path.replace('\\', '/')
        parts = img_path.split('/')
        
        class_name = None
    
        # Extract class name based on known structure
        for part in parts:
            if 'Opt_' in part:
                class_name = part.split('_')[2]  # Extract the part after 'Opt_'
                break
    
        # Additional checks for specific class names
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
    
        # Normalize certain class names
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

root_dirs_train = [r'/nfsd/lttm4/tesisti/koca/datasets/ACDC',
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

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((540, 960)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets for training and testing
train_datasets = [CustomDataset(root_dir, data_file, transform=transform) for root_dir, data_file in zip(root_dirs_train, train_data_files)]
combined_train_dataset = ConcatDataset(train_datasets)

test_datasets = [CustomDataset(root_dir, data_file, transform=transform) for root_dir, data_file in zip(root_dirs_test, test_data_files)]
combined_test_dataset = ConcatDataset(test_datasets)

# Create data loaders for training and testing
train_dataloader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(combined_test_dataset, batch_size=32, shuffle=False)

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
iterator = iter(train_dataloader)
inputs, classes = next(iterator)
out = torchvision.utils.make_grid(inputs[:4])

# Print class names for debugging
print("Class names list:", train_classes)

# Ensure the titles are correct
titles = [train_classes[x] for x in classes[:4].tolist()]
print("Titles:", titles)

imshow(out, title=" | ".join(titles))

###  2

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  

# Modify the final layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

# Move the model to the appropriate device
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
start_time = time.time()

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_dataloader:
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
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_losses.append(running_loss / len(train_dataloader))
    train_accuracy = 100 * correct / total
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Training Loss: {running_loss / len(train_dataloader):.4f}, Training Accuracy: {train_accuracy:.2f}%, Time: {time.time() - start_time:.2f} seconds")
    
    # Validation loop
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Calculate accuracy
    correct_predictions = sum(np.array(y_true) == np.array(y_pred))
    total_predictions = len(y_true)
    epoch_acc = (correct_predictions / total_predictions) * 100.0
    
    val_losses.append(running_loss / len(test_dataloader))
    val_accuracies.append(epoch_acc)
    print(f"Test Loss: {running_loss / len(test_dataloader):.4f}, Test Accuracy: {epoch_acc:.2f}%, Time: {time.time() - start_time:.2f} seconds")
    
print('Finished Training')

# Print final accuracies
print(f"Final Training Accuracy: {train_accuracies[-1]}%")
print(f"Final Validation Accuracy: {val_accuracies[-1]}%")

# Create a table with epoch-wise accuracy and loss
epoch_data = {
    'Epoch': list(range(1, num_epochs + 1)),
    'Training Loss': train_losses,
    'Test Loss': val_losses,
    'Training Accuracy': train_accuracies,
    'Test Accuracy': val_accuracies
}

df = pd.DataFrame(epoch_data)
print(df)

# Plotting accuracy and loss curves
plt.figure(figsize=(12, 6))

# Training and Test accuracy
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, num_epochs+1), train_accuracies, '-o', label='Training Accuracy')
plt.plot(np.arange(1, num_epochs+1), val_accuracies, '-o', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Test Accuracy over Epochs')
plt.grid(True)

# Training and Test loss
plt.subplot(1, 2, 2)
plt.plot(np.arange(1, num_epochs+1), train_losses, '-o', label='Training Loss')
plt.plot(np.arange(1, num_epochs+1), val_losses, '-o', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss over Epochs')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('acc_loss_plot_all-syndrone.png')  # Save the loss plot

print('Confusion matrix: \n', confusion_matrix(y_true, y_pred))
print('Classification report: \n', classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[train_dataset.label_to_class[i] for i in range(4)])
disp.plot(cmap=plt.cm.Blues)
plt.show()
plt.savefig('confusion_matrix_all-syndrone.png', bbox_inches='tight')

class_accuracies = cm.diagonal() / cm.sum(axis=1)
for i, class_name in enumerate(class_names):
    print(f'Accuracy for {class_name}: {class_accuracies[i]*100:.2f}%')
