import torch
from torch import nn
from model import PGD, Conv, Linear
from dataset import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage
from model import Feature_Squeezing


def train(dataloader, model, loss_fn, optimizer, adversarial=False, regularization=None, beta=1e-4, device='cpu', verbose=True):
    """The training function

    Args:
        dataloader (DataLoader): The training DataLoader, which can be the return of DataLoader()
        model (nn.Module): The Module
        loss_fn (nn.function): The loss function
        optimizer (Optimizer): The Optimizer in training
        device (str, optional): The device to use of the training. Defaults to 'cpu'.
    """
    size = len(dataloader.dataset)
    pgd = PGD(model)
    for i, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        if regularization is None:
            loss = loss_fn(pred, y)
        else:
            loss = loss_fn(pred, y) + \
                model.regularizationTerm(regularization, beta=beta)
        optimizer.zero_grad()
        loss.backward()
        model.reconstruct()
        optimizer.step()
        if adversarial is True:
            X = pgd.forward(X, y, loss_fn, device)
            pred = model(X)
            if regularization is None:
                loss = loss_fn(pred, y)
            else:
                loss = loss_fn(
                    pred, y) + model.regularizationTerm(regularization, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            model.reconstruct()
            optimizer.step()
        if verbose is True:
            if i % 100 == 0:
                loss, current = loss.item(), i * len(X)
                print(f"loss: {loss:>7f}    [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device='cpu', feature_squeeze=None,verbose=True):
    """The test function

    Args:
        dataloader (DataLoader): The test DataLoader, which can be the return of DataLoader()
        model (nn.Module): The Module
        loss_fn (nn.function): The loss function
        device (str, optional): The device to use of the test. Defaults to 'cpu'.
    """
    size = len(dataloader.dataset)
    batch_num = len(dataloader)
    test_loss, correct = 0, 0
    if feature_squeeze is not None:
        fs = Feature_Squeezing(mode = feature_squeeze,device=device)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            if feature_squeeze is not None:
                X = fs.forward(X)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            model.reconstruct()
            correct += (pred.argmax(1) == y).type(torch.float32).sum().item()

        test_loss /= batch_num
        correct /= size
        if verbose is True:
            print(
                f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


def show_img(img, label, model, loss_fn, eps=8/255, alpha =2/255 ,steps=4,device='cpu',output='result'):
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    dataset = CIFAR10(is_transform=True, normalize=False)
    dataset = dataset.test_set
    for X, _ in dataset:
        or_img = X
        break
    ad_img = PGD(model,eps,alpha,steps).forward(img, label, loss_fn, device)
    or_result = model(img).argmax(1)
    ad_result = model(ad_img).argmax(1)
    or_pred = (or_result == label)
    ad_pred = (ad_result == label)
    imgs = []
    ad_imgs = []
    labels = []
    ad_labels = []
    for i in range(len(or_pred)):
        if or_pred[i] == True and ad_pred[i] == False:
            imgs.append(or_img[i])
            labels.append(classes[or_result[i]])
            ad_labels.append(classes[ad_result[i]])
            ad_imgs.append(ad_img[i])
        if len(imgs) >= 5:
            break
    fig, axs = plt.subplots(ncols=len(imgs), nrows=2)
    for i, img in enumerate(imgs):
        img = ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[],title=labels[i])
    for i, ad_img in enumerate(ad_imgs):
        ad_img = ToPILImage()(ad_img.to('cpu'))
        axs[1, i].imshow(np.asarray(ad_img))
        axs[1, i].set(xticklabels=[],
                             yticklabels=[], xticks=[], yticks=[],title=ad_labels[i])
    plt.savefig(output+'.png')
