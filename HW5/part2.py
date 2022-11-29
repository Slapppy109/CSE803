import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

MNIST_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])
MNIST_train = datasets.MNIST('.', download=True, train = True, transform=MNIST_transform)
MNIST_test = datasets.MNIST('.', download=True, train = False, transform=MNIST_transform)
FASHION_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2859], [0.3530])
])
FASHION_train = datasets.FashionMNIST('.', download=True, train=True, transform=MNIST_transform)
FASHION_test = datasets.FashionMNIST('.', download=True, train=False, transform=FASHION_transform)

class GridDataset(Dataset):
    def __init__(self, MNIST_dataset, FASHION_dataset): # pass in dataset
        assert len(MNIST_dataset) == len(FASHION_dataset)
        self.MNIST_dataset, self.FASHION_dataset = MNIST_dataset, FASHION_dataset
        self.targets = FASHION_dataset.targets
        torch.manual_seed(442) # Fix random seed for reproducibility
        N = len(MNIST_dataset)
        self.randpos = torch.randint(low=0,high=4,size=(N,)) # position of the FASHION-MNIST image
        self.randidx = torch.randint(low=0,high=N,size=(N,3)) # indices of MNIST images
    
    def __len__(self):
        return len(self.MNIST_dataset)
    
    def __getitem__(self,idx): # Get one Fashion-MNIST image and three MNIST images to make a new image
        idx1, idx2, idx3 = self.randidx[idx]
        x = self.randpos[idx]%2
        y = self.randpos[idx]//2
        p1 = self.FASHION_dataset.__getitem__(idx)[0]
        p2 = self.MNIST_dataset.__getitem__(idx1)[0]
        p3 = self.MNIST_dataset.__getitem__(idx2)[0]
        p4 = self.MNIST_dataset.__getitem__(idx3)[0]
        combo = torch.cat((torch.cat((p1,p2),2),torch.cat((p3,p4),2)),1)
        combo = torch.roll(combo, (x*28,y*28), dims=(0,1))
        return (combo,self.targets[idx])
trainset = GridDataset(MNIST_train, FASHION_train)
testset = GridDataset(MNIST_test, FASHION_test)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels = 6, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels = 6,out_channels = 16, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels= 16, out_channels= 128, kernel_size= 5, padding= 2), nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels= 128, out_channels= 128, kernel_size= 5, padding= 2),
            nn.Conv2d(in_channels= 128, out_channels= 128, kernel_size= 5, padding= 2),

        )
        out_channel = 128 
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channel,10)
        self.conv = nn.Conv2d(out_channel,10,1) # 1x1 conv layer (substitutes fc)

    def transfer(self): # Copy weights of fc layer into 1x1 conv layer
        self.conv.weight = nn.Parameter(self.fc.weight.unsqueeze(2).unsqueeze(3))
        self.conv.bias = nn.Parameter(self.fc.bias)

    def visualize(self,x):
        x = self.base(x)
        x = self.conv(x)
        return x
        
    def forward(self,x):
        x = self.base(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
device = "cuda" if torch.cuda.is_available() else "cpu"
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=True)
model = Network().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
num_epoch = 10 

def train(model, loader, num_epoch = 10): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
    print("Done!")

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    i = 0
    cor_img = None
    cor_idx = -1
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
            if torch.argmax(pred[0]) == label[0] and cor_idx == -1:
                cor_img = batch[0]
                cor_idx = i
            i += 1
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc, cor_idx, cor_img

train(model, trainloader)
acc, idx, img = evaluate(model, testloader)

model.transfer() # Copy the weights from fc layer to 1x1 conv layer

img = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
act_layer = model.visualize(img)

plt.imshow(Tensor.cpu(img).numpy().reshape([img.shape[2],img.shape[3]]), cmap='gray')
plt.title('Input Image', fontsize=16)
plt.axis('off')

f, ax = plt.subplots(2, 5)
f.suptitle("Activation map for each class", fontsize=16)
ax[0,0].imshow(Tensor.cpu(act_layer.detach()).numpy()[0][0], cmap='gray')
ax[0,0].set_title('0', fontsize=16)
ax[0,0].axis('off')

ax[0,1].imshow(Tensor.cpu(act_layer.detach()).numpy()[0][1], cmap='gray')
ax[0,1].set_title('1', fontsize=16)
ax[0,1].axis('off')

ax[0,2].imshow(Tensor.cpu(act_layer.detach()).numpy()[0][2], cmap='gray')
ax[0,2].set_title('2', fontsize=16)
ax[0,2].axis('off')

ax[0,3].imshow(Tensor.cpu(act_layer.detach()).numpy()[0][3], cmap='gray')
ax[0,3].set_title('3', fontsize=16)
ax[0,3].axis('off')

ax[0,4].imshow(Tensor.cpu(act_layer.detach()).numpy()[0][4], cmap='gray')
ax[0,4].set_title('4', fontsize=16)
ax[0,4].axis('off')

ax[1,0].imshow(Tensor.cpu(act_layer.detach()).numpy()[0][5], cmap='gray')
ax[1,0].set_title('5', fontsize=16)
ax[1,0].axis('off')

ax[1,1].imshow(Tensor.cpu(act_layer.detach()).numpy()[0][6], cmap='gray')
ax[1,1].set_title('6', fontsize=16)
ax[1,1].axis('off')

ax[1,2].imshow(Tensor.cpu(act_layer.detach()).numpy()[0][7], cmap='gray')
ax[1,2].set_title('7', fontsize=16)
ax[1,2].axis('off')

ax[1,3].imshow(Tensor.cpu(act_layer.detach()).numpy()[0][8], cmap='gray')
ax[1,3].set_title('8', fontsize=16)
ax[1,3].axis('off')

ax[1,4].imshow(Tensor.cpu(act_layer.detach()).numpy()[0][9], cmap='gray')
ax[1,4].set_title('9', fontsize=16)
ax[1,4].axis('off')

plt.gcf().set_size_inches(18, 10)
plt.show()

print(f"Test set Index: {idx}")
