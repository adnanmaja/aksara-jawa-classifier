import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# 1. Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 3 channels → 32 channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 32 → 64
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 64 → 128
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # 224/2/2/2 = 28
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv → ReLU → Pool
        x = self.pool(torch.relu(self.conv1(x)))  # 224 → 112
        x = self.pool(torch.relu(self.conv2(x)))  # 112 → 56  
        x = self.pool(torch.relu(self.conv3(x)))  # 56 → 28
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 2. Data loading (your existing code)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

datadir = "D:/Python/Aksara/aksara_data"
train_dataset = datasets.ImageFolder(datadir + "/train", transform=train_transform)
test_dataset = datasets.ImageFolder(datadir + "/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(train_dataset.classes)

model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Training on {device}")
print(f"Number of classes: {num_classes}")

# 4. Debug what's happening
print("Checking train_loader...")
sample_batch = next(iter(train_loader))
print(f"train_loader returns {len(sample_batch)} items")
print(f"First item type: {type(sample_batch[0])}")
print(f"First item: {sample_batch[0]}")
if len(sample_batch) > 1:
    print(f"Second item type: {type(sample_batch[1])}")
    print(f"Second item: {sample_batch[1]}")

# Let's also check the dataset directly
print("\nChecking dataset directly...")
sample_data = train_dataset[0]
print(f"Dataset returns {len(sample_data)} items")
print(f"First item type: {type(sample_data[0])}")
if hasattr(sample_data[0], 'shape'):
    print(f"First item shape: {sample_data[0].shape}")
else:
    print(f"First item: {sample_data[0]}")

# Check if images exist in the folder
import os
print(f"\nChecking folder structure...")
print(f"Directory exists: {os.path.exists(datadir)}")
if os.path.exists(datadir):
    print(f"Contents: {os.listdir(datadir)}")

# Try to check one more level down
try:
    subfolders = [f for f in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, f))]
    if subfolders:
        first_subfolder = os.path.join(datadir, subfolders[0])
        print(f"First subfolder contents: {os.listdir(first_subfolder)[:5]}")  # Show first 5 files
except:
    print("Could not check subfolder contents")

# 4. Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels = batch[0], batch[1]  # Take first 2 items
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

# 5. Test the model
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
print(f'Test Accuracy: {accuracy:.2f}%')

# 6. Save model
torch.save(model.state_dict(), 'simple_cnn.pth')
print('Model saved!')