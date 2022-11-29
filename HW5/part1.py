import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

# Load the dataset and train, val, test splits
print("Loading datasets...")
FASHION_transform = transforms.Compose([
    transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
    transforms.Normalize([0.2859], [0.3530]) # Normalize to zero mean and unit variance
])
FASHION_trainval = datasets.FashionMNIST('.', download=True, train=True, transform=FASHION_transform)
FASHION_train = Subset(FASHION_trainval, range(50000))
FASHION_val = Subset(FASHION_trainval, range(50000,60000))
FASHION_test = datasets.FashionMNIST('.', download=True, train=False, transform=FASHION_transform)
print("Done!")

# Create dataloaders
# TODO: Experiment with different batch sizes
trainloader = DataLoader(FASHION_train, batch_size=512, shuffle=True)
valloader = DataLoader(FASHION_val, batch_size=512, shuffle=True)
testloader = DataLoader(FASHION_test, batch_size=512, shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels = 6, kernel_size=5, padding=2), nn.ReLU(), 
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels = 6,out_channels = 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(28*28, 120, bias=True), nn.ReLU(), nn.Dropout(),
            nn.Linear(120, 84, bias=True), nn.ReLU(),nn.Dropout(),
            nn.Linear(84, 10, bias=True))

    def forward(self,x):
        x = self.net(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = Network().to(device)
criterion = nn.CrossEntropyLoss() # Specify the loss layer
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-10) 
num_epoch = 30 #

def train(model, loader, num_epoch = 10): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    total_loss = []
    total_acc = []
    for i in range(num_epoch):
        running_loss = []
        total = 0
        correct = 0
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
            
            _, predicted = pred.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            accu = 100. * correct/total
            total_acc.append(accu)
        total_loss += running_loss
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
    print("Done!")
    return total_loss, total_acc

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc
    
train(model, trainloader, 10)
print("Evaluate on validation set...")
evaluate(model, valloader)
print("Evaluate on test set")
evaluate(model, testloader)
