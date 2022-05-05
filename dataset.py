import numpy as np
import torchvision
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model import PGD


class CIFAR10():
    """The CIFAR10 Dataset class
    """
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    NUM_CLASSES = 10
    IMAGE_SIZE = [32, 32]
    IMAGE_CHANNELS = 3

    def __init__(self, is_transform=False, normalize=True, batch_size=64):
        """The init function of the class

        Args:
            is_transform (bool, optional): Wether use transform to the dataset. Defaults to False.
        """
        if normalize == True:
            transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(self.MEAN, self.STD)])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        if is_transform == False:
            self.load_dataset(None, batch_size)
        else:
            self.load_dataset(transform, batch_size)

    def load_dataset(self, transform, batch_size):
        """The function to load the dataset, use torchvision.dataset

        Args:
            transform (Transforms object): The method to transform the dataset.
        """
        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", transform=transform, download=True)
        self.testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, transform=transform, download=True)
        self.train_set = DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_set = DataLoader(
            self.testset, batch_size=batch_size, shuffle=False, num_workers=4)


class Adversarial_examples(Dataset):
    """The Adversarial_examples Dataset.
    """

    def __init__(self, test_dataloader, model, loss_fn, eps=8 / 255, alpha=2 / 255, steps=4, device='cpu'):
        """The __init__ function of the Adversarial_examples Dataset.

        Args:
            test_dataloader (DataLoader): The dataloader of test dataset.
            model (nn.Module): The model to attack.
            loss_fn (_type_): The loss function.
            eps (float, optional): The limit of the disturbance. Defaults to 8/255.
            alpha (float, optional): The magnitude of the disturbance. Defaults to 2/255.
            steps (int, optional): Number of iterations of the disturbance. Defaults to 4.
            device (str, optional): The device to use. Defaults to 'cpu'.
        """
        self.adversarial_data = None
        self.lables = None
        pgd = PGD(model, eps=eps, alpha=alpha, steps=steps)
        for image, lable in test_dataloader:
            data = pgd.forward(images=image, labels=lable,
                               loss_fn=loss_fn, device=device)
            if self.adversarial_data is None:
                self.adversarial_data = data.cpu()
            else:
                self.adversarial_data = torch.concat(
                    (self.adversarial_data, data.cpu()), dim=0)
            if self.lables is None:
                self.lables = lable.cpu()
            else:
                self.lables = torch.concat((self.lables, lable.cpu()), dim=0)

    def __len__(self):
        return len(self.adversarial_data)

    def __getitem__(self, index):
        return self.adversarial_data[index], self.lables[index]
