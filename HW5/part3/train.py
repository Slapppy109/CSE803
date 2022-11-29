import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from dataset import FacadeDataset

N_CLASS=5

class DownBlock(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            pooling: bool = True
            ):
        super(DownBlock, self).__init__()
        # Init Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = 1
        self.pooling = pooling

        # Convolutional Layers
        self.c1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size= 3, padding=1, bias=False)
        self.c2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size= 3, padding=1, bias=False)

        # Pooling Layer
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size= 2, stride= 2)

        # Normalization Layers
        self.n1 = nn.BatchNorm2d(num_features=self.out_channels)
        self.n2 = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, x):
        x = F.relu(self.n1(self.c1(x)))
        x = F.relu(self.n2(self.c2(x)))
        bp = x
        if self.pooling:
            x = self.pool(x)
        return x, bp

class UpBlock(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            ):
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Upsample Layer
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size= 2, stride=2, bias=False)

        # Convolutional Layers
        self.c1 = nn.Conv2d(in_channels= (2 * out_channels), out_channels= out_channels, kernel_size= 3, padding= 1, bias=False)
        self.c2 = nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= 3, padding=1, bias=False)

        # Normalization Layers
        self.n1 = nn.BatchNorm2d(num_features=self.out_channels)
        self.n2 = nn.BatchNorm2d(num_features=self.out_channels)
        self.n3 = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, down, up):
        up = self.n1(self.up(up))
        x = torch.cat((up,down), 1)
        x = F.relu(self.n2(self.c1(x)))
        x = F.relu(self.n3(self.c2(x)))
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_class = N_CLASS
        self.depth = 5
        self.down_blks = []
        self.up_blks = []
        self.in_channels = 3
        self.starting_filters = 32

        # Create Down blocks
        for i in range(self.depth):
            in_channels = self.in_channels if i == 0 else out_channels
            out_channels = self.starting_filters * (2**i)
            pooling = True if i < self.depth - 1 else False

            d_blk = DownBlock(in_channels, out_channels, pooling=pooling)
            self.down_blks.append(d_blk)

        # Create Up blocks
        for i in range(self.depth - 1):
            in_channels = out_channels 
            out_channels = in_channels // 2

            u_blk = UpBlock(in_channels, out_channels)
            self.up_blks.append(u_blk)

        self.down_blks = nn.ModuleList(self.down_blks)
        self.up_blks = nn.ModuleList(self.up_blks)

        self.output = nn.Conv2d(out_channels, self.n_class, kernel_size= 1)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        down_outs = []

        for blk in self.down_blks:
            x, bp = blk(x)
            down_outs.append(bp)

        for i, blk in enumerate(self.up_blks):
            # Connect with previous down blocks
            bp = down_outs[-(i+2)]
            x = blk(bp, x)

        x = self.output(x)
        return x


def save_label(label, path):
    '''
    Function for ploting labels.
    '''
    colormap = [
        '#000000',
        '#0080FF',
        '#80FF80',
        '#FF8000',
        '#FF0000',
    ]
    assert(np.max(label)<len(colormap))
    colors = [hex2rgb(color, normalise=False) for color in colormap]
    w = png.Writer(label.shape[1], label.shape[0], palette=colors, bitdepth=4)
    with open(path, 'wb') as f:
        w.write(f, label)

def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    net = net.train()
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
    end = time.time()
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, running_loss, end-start))

def test(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output, labels)
            losses += loss.item()
            cnt += 1
    print(losses / cnt)
    return (losses/cnt)


def cal_AP(testloader, net, criterion, device):
    '''
    Calculate Average Precision
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        preds = [[] for _ in range(5)]
        heatmaps = [[] for _ in range(5)]
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images).cpu().numpy()
            for c in range(5):
                preds[c].append(output[:, c].reshape(-1))
                heatmaps[c].append(labels[:, c].cpu().numpy().reshape(-1))

        aps = []
        for c in range(5):
            preds[c] = np.concatenate(preds[c])
            heatmaps[c] = np.concatenate(heatmaps[c])
            if heatmaps[c].max() == 0:
                ap = float('nan')
            else:
                ap = ap_score(heatmaps[c], preds[c])
                aps.append(ap)
            print("AP = {}".format(ap))
    print(f"Average AP: {np.average(aps)}")
    return None


def get_result(testloader, net, device, folder='output_train'):
    result = []
    cnt = 1
    with torch.no_grad():
        net = net.eval()
        cnt = 0
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)[0].cpu().numpy()
            c, h, w = output.shape
            assert(c == N_CLASS)
            y = np.zeros((h,w)).astype('uint8')
            for i in range(N_CLASS):
                mask = output[i]>0.5
                y[mask] = i
            gt = labels.cpu().data.numpy().squeeze(0).astype('uint8')
            save_label(y, './{}/y{}.png'.format(folder, cnt))
            save_label(gt, './{}/gt{}.png'.format(folder, cnt))
            plt.imsave(
                './{}/x{}.png'.format(folder, cnt),
                ((images[0].cpu().data.numpy()+1)*128).astype(np.uint8).transpose(1,2,0))

            cnt += 1

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data = FacadeDataset(flag='train', data_range=(0,724), onehot=False)
    train_loader = DataLoader(train_data, batch_size=32)
    test_data = FacadeDataset(flag='test_dev', data_range=(0,114), onehot=False)
    test_loader = DataLoader(test_data, batch_size=1)
    ap_data = FacadeDataset(flag='test_dev', data_range=(0,114), onehot=True)
    ap_loader = DataLoader(ap_data, batch_size=1)

    evaluation_data = FacadeDataset(flag='train', data_range=(724,906), onehot=False)
    evaluation_loader = DataLoader(evaluation_data, batch_size=32) 

    name = 'starter_net'
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss() #TODO decide loss
    optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)

    print('\nStart training')
    total_loss = []
    val_loss = []
    for epoch in range(10):
        print('-----------------Epoch = %d-----------------' % (epoch+1))
        train_loss = train(train_loader, net, criterion, optimizer, device, epoch+1)
        test_loss = test(evaluation_loader, net, criterion, device)
        total_loss += train_loss
        val_loss += test_loss

    print('\nFinished Training, Testing on test set')
    test(test_loader, net, criterion, device)
    print('\nGenerating Unlabeled Result')
    result = get_result(test_loader, net, device, folder='output_test')

    torch.save(net.state_dict(), './models/model_{}.pth'.format(name))

    cal_AP(ap_loader, net, criterion, device)

if __name__ == "__main__":
    main()
