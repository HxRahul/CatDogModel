import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# 1. Setup
seed = 21
random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 2. Define Transforms
# Standard ImageNet normalization
norm_mean = (0.485, 0.456, 0.406)
norm_std  = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),        # Resize first to ensure we have enough image
    transforms.RandomResizedCrop(224),    # Then crop
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),        # deterministic resize for testing
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 3. Correct Dataset Splitting
root_dir = "/home/wahoo/Documents/CatDogModel/PetImages" 

# Load the dataset purely to get the length and indices
full_dataset_ref = datasets.ImageFolder(root=root_dir) 
n = len(full_dataset_ref)
indices = list(range(n))

# Shuffle indices manually so we can use them for both sets
random.shuffle(indices)
split = int(0.8 * n)
train_idx, val_idx = indices[:split], indices[split:]

# Create two separate dataset objects so they can have different transforms
train_data_full = datasets.ImageFolder(root=root_dir, transform=train_transform)
val_data_full   = datasets.ImageFolder(root=root_dir, transform=test_transform)

# Use Subset to apply the split indices
train_dataset = torch.utils.data.Subset(train_data_full, train_idx)
val_dataset   = torch.utils.data.Subset(val_data_full, val_idx)

print(f"Total: {n}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
classes = full_dataset_ref.classes

# 4. DataLoaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 5. Improved Model (Deeper + BatchNorm)
class BetterModel(nn.Module):
    def __init__(self):
        super(BetterModel, self).__init__()
        
        # Block 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2, 2) # Reduces 224 -> 112
        
        # Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        # Pool reduces 112 -> 56
        
        # Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        # Pool reduces 56 -> 28
        
        # Adaptive pool allows us to not worry about exact pixel math
        self.adapt = nn.AdaptiveAvgPool2d((4, 4)) # Outputs 128 * 4 * 4
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.drop = nn.Dropout(0.5) # Prevents overfitting
        self.fc2 = nn.Linear(512, len(classes)) 

    def forward(self, x):
        # Block 1
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        
        x = self.adapt(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

model = BetterModel().to(device)

# 6. Training Setup
criterion = nn.CrossEntropyLoss()
# Adam is generally faster/easier than SGD for beginners
optimizer = optim.Adam(model.parameters(), lr=0.001) 

num_epochs = 10 

print("Starting Training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f} | Acc: {epoch_acc:.4f}")

print("Finished Training")

# 7. Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

print(f"Final Validation Accuracy: {100.0 * correct / total:.2f}%")