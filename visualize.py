import torch
from dataset import CIFAR10,Adversarial_examples
from torch import nn,optim
from seed import setup_seed
from model import Covnet
import argparse
from train import test
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

setup_seed(1)

args = argparse.ArgumentParser()
args.add_argument('-d', '--device', dest='device', default='cpu', type=str)
args.add_argument('-aim',default='eps',dest='aim',type=str)
args = args.parse_args()

device = args.device
batch_size = 32
lr = 1e-3
loss = nn.CrossEntropyLoss()
net = Covnet(device=device)
optimizer = optim.Adam(net.parameters(), lr=lr)

net.load_state_dict(torch.load('checkpoint/checkpoint.pt'))

dataset = CIFAR10(is_transform=True, batch_size=batch_size)
test_loader = dataset.test_set
history = []
if args.aim == 'eps':
    for i in np.around(np.linspace(0.01, 0.1, 10), 2):
        ad_data_dataset = Adversarial_examples(
            test_loader, net, loss, device=device,eps = i)
        ad_data_loader = DataLoader(ad_data_dataset, batch_size=batch_size)
        history.append(test(ad_data_loader, net, loss,device=device,verbose=False))
if args.aim == 'alpha':
    for i in np.around(np.linspace(0.01, 0.1, 10), 2):
        ad_data_dataset = Adversarial_examples(
            test_loader, net, loss, device=device,alpha = i)
        ad_data_loader = DataLoader(ad_data_dataset, batch_size=batch_size)
        history.append(test(ad_data_loader, net, loss,device=device,verbose=False))
if args.aim == 'step':
    for i in np.arange(1,11):
        ad_data_dataset = Adversarial_examples(
            test_loader, net, loss, device=device,steps = i)
        ad_data_loader = DataLoader(ad_data_dataset, batch_size=batch_size)
        history.append(test(ad_data_loader, net, loss,device=device,verbose=False))

plt.plot(history)
if args.aim == 'step':
    plt.xticks(np.arange(len(history)),np.arange(1,11))
else:
    plt.xticks(np.arange(len(history)),np.around(np.linspace(0.01, 0.1, 10),2))
plt.title(args.aim)
plt.savefig(args.aim + '.png')