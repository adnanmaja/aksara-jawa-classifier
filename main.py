import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


# Trying to use a simple CNN model (ended up performing poorly) 
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) 
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 512)  
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv → ReLU → Pool
        x = self.pool(torch.relu(self.conv1(x)))  
        x = self.pool(torch.relu(self.conv2(x)))    
        x = self.pool(torch.relu(self.conv3(x)))  
        x = self.pool(F.relu(self.conv3(x)))  
        x = self.gap(x)                       
        x = x.view(x.size(0), -1) 

        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Data Loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=(-20, 20)),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1), 
        scale=(0.8, 1.2),     
        shear=(-15, 15)       
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


traindir = "D:/Python/Aksara/data_splits/train"
valdir = "D:/Python/Aksara/data_splits/val"
testdir = "D:/Python/Aksara/data_splits/test"
train_dataset = datasets.ImageFolder(traindir, transform=train_transform)
test_dataset = datasets.ImageFolder(testdir, transform=transform)
val_dataset = datasets.ImageFolder(valdir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Setting up the model (ResNet18)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(train_dataset.classes)

model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 20)
model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

print(f"Device: {device}")
print(f"Number of classes: {num_classes}")


# Training Loop
num_epochs = 30  # Diminishing return after 30 epochs
val_correct = 0
val_total = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels = batch[0], batch[1] 
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()

    # Evaluation after each epochs
    model.eval()
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_preds = val_outputs.argmax(dim=1)
            val_correct += (val_preds == val_labels).sum().item()
            val_total += val_labels.size(0)

    val_acc = 100 * val_correct / val_total  
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')
    print(f"Validation Accuracy: {val_acc:.2f}%")




# Model Testing
model.eval()
correct = 0
total = 0


with torch.no_grad():
    for batch in test_loader:
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels = batch[0], batch[1]
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Saving the model
torch.save(model.state_dict(), "aksara_model.pth")
print("Model saved")

