import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 0. Config / EarlyStopping
# ---------------------------
seed = 21
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Training params
num_epochs = 55           # max epochs (early stopping may stop earlier)
patience = 5              # early stopping patience (epochs without improvement)
min_delta = 1e-4          # minimum change in val loss to count as improvement
save_path = "best_model.pth"

# ---------------------------
# EarlyStopping helper
# ---------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=True, path="best_model.pth", trace_func=print):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.trace_func = trace_func

        self.best_val_loss = np.inf
        self.num_bad_epochs = 0
        self.early_stop = False

    def step(self, val_loss, model):
        improved = val_loss < (self.best_val_loss - self.min_delta)
        if improved:
            self.best_val_loss = val_loss
            self.num_bad_epochs = 0
            self._save_checkpoint(val_loss, model)
            return False
        else:
            self.num_bad_epochs += 1
            if self.verbose:
                self.trace_func(f"No improvement in val loss for {self.num_bad_epochs}/{self.patience} epochs.")
            if self.num_bad_epochs >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.trace_func("Early stopping triggered.")
                return True
            return False

    def _save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f"Validation loss decreased -> saving model ({self.best_val_loss:.6f} -> {val_loss:.6f}) to {self.path}")
        torch.save(model.state_dict(), self.path)

    def load_best(self, model, device=None):
        if os.path.exists(self.path):
            state = torch.load(self.path, map_location=device if device is not None else "cpu")
            model.load_state_dict(state)
            if self.verbose:
                self.trace_func(f"Loaded best model from {self.path} (val_loss={self.best_val_loss:.6f})")
        else:
            if self.verbose:
                self.trace_func("No checkpoint found to load.")

early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True, path=save_path)

# ---------------------------
# 1. Transforms & Dataset
# ---------------------------
norm_mean = (0.485, 0.456, 0.406)
norm_std  = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

root_dir = "/home/wahoo/Documents/CatDogModel/PetImages"
full_dataset_ref = datasets.ImageFolder(root=root_dir)
n = len(full_dataset_ref)
indices = list(range(n))
random.shuffle(indices)
split = int(0.8 * n)
train_idx, val_idx = indices[:split], indices[split:]

train_data_full = datasets.ImageFolder(root=root_dir, transform=train_transform)
val_data_full   = datasets.ImageFolder(root=root_dir, transform=test_transform)
train_dataset = torch.utils.data.Subset(train_data_full, train_idx)
val_dataset   = torch.utils.data.Subset(val_data_full, val_idx)

print(f"Total: {n}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
classes = full_dataset_ref.classes

# ---------------------------
# 2. DataLoaders
# ---------------------------
batch_size = 32
num_workers = 4
pin_memory = True if torch.cuda.is_available() else False

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=pin_memory
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=pin_memory
)

# ---------------------------
# 3. Model
# ---------------------------
class BetterModel(nn.Module):
    def __init__(self):
        super(BetterModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, len(classes))

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.adapt(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

model = BetterModel().to(device)

# ---------------------------
# 4. Setup training
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# tracking containers
train_losses = []
val_losses = []
train_accs = []
val_accs = []
epoch_times = []

# ---------------------------
# 5. Training loop with Early Stopping
# ---------------------------
print("Starting Training (with Early Stopping)...")
for epoch in range(num_epochs):
    epoch_start = time.time()
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

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # validation
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_running_loss / val_total if val_total > 0 else 0.0
    val_acc = val_correct / val_total if val_total > 0 else 0.0
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")

    # early stopping check (monitors val_loss)
    if early_stopper.step(val_loss, model):
        print(f"Stopping early at epoch {epoch+1}")
        break

# restore best weights
early_stopper.load_best(model, device=device)

print("Finished Training (best model restored).")

# ---------------------------
# 6. Plot results
# ---------------------------
epochs_ran = len(train_losses)
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(range(1, epochs_ran+1), train_losses, label="train")
plt.plot(range(1, epochs_ran+1), val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, epochs_ran+1), train_accs, label="train")
plt.plot(range(1, epochs_ran+1), val_accs, label="val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()

plt.tight_layout()
plt.show()

# ---------------------------
# 7. Final validation print
# ---------------------------
if len(val_accs) > 0:
    print(f"Final Validation Accuracy: {100.0 * val_accs[-1]:.2f}%")
else:
    print("No validation results recorded.")
