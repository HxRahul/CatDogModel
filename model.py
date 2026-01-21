import os
import random
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

#seed random for reproducable results
seed = 21
random.seed(seed)
torch.manual_seed(seed)

#if cuda is available for the gpu then use it, if not just use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

#0 for now can change later for multiprocessing
num_workers = 0 

#train and test transforms
imagenet_mean = (0.485, 0.456, 0.406) # mean used to normalized
imagenet_std  = (0.229, 0.224, 0.225) # normalization deviation

#pre processing the images, doing different things to it
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

#preprocessing the testing images
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

#Split the dataset for training and testing
root_dir = r"C:\CatDogCNN\PetImages"  
full_dataset = datasets.ImageFolder(root=root_dir, transform=train_transform)


n = len(full_dataset)
train_len = int(0.8 * n) # 80% training 20% for testing
val_len = n - train_len
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_len, val_len],
                                                           generator=torch.Generator().manual_seed(seed))


val_dataset.dataset = datasets.ImageFolder(root=root_dir, transform=test_transform)

#just showing size of the cample and check the split size
print("Total samples:", n, " Train:", train_len, " Val:", val_len)
classes = full_dataset.classes
print("Classes:", classes)

#Dataloader 
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


#actual model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.adapt = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes))  

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.adapt(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Model().to(device)

#loss optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#training loop
num_epochs = 5
print_every_batches = 200 
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

        if i % print_every_batches == 0:
            print(f"[Epoch {epoch+1} Batch {i}]  Avg loss: {running_loss / i:.4f}  Acc: {running_correct / total:.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc  = running_correct / total
    print(f"Epoch {epoch+1}/{num_epochs}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

print("Finished Training")

#saving
PATH = './catdog_net.pth'
torch.save(model.state_dict(), PATH)
print("Saved model to", PATH)


model.eval()
correct = 0
total = 0
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        for label, pred in zip(labels, preds):
            if label == pred:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

print(f"Validation Accuracy: {100.0 * correct / total:.2f}%")
for classname in classes:
    if total_pred[classname] > 0:
        acc = 100.0 * correct_pred[classname] / total_pred[classname]
        print(f"Accuracy for class {classname:5s}: {acc:.1f}%")
    else:
        print(f"No samples for class {classname}")


model2 = Model()
model2.load_state_dict(torch.load(PATH))
model2 = model2.to(device)
model2.eval()


dataiter = iter(val_loader)
inputs, labels = next(dataiter)
inputs, labels = inputs.to(device), labels.to(device)
with torch.no_grad():
    outputs = model2(inputs)
    _, preds = torch.max(outputs, 1)

#truth tables and predictions
print("GT:   ", [classes[l] for l in labels[:8].cpu().numpy()])
print("Pred: ", [classes[p] for p in preds[:8].cpu().numpy()])
