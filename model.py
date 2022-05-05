from torch.autograd import Function
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import math
from torch import Tensor, nn
import torch
from torch.nn import quantized
from scipy import ndimage
import cv2 as cv


class Covnet(nn.Module):
    """A Simple net to test the code
    """

    def __init__(self, device='cpu', quant=False, dynamic=False) ->None:
        """The init function of the Covnet

        Args:
            device (str, optional): The device to use. Defaults to 'cpu'.
            quant (bool, optional): Whether to use quantlization. Defaults to False.
            dynamic (bool, optional): Whether to use dynamic active function . Defaults to False.
        """
        super(Covnet, self).__init__()
        self.device = device
        self.quant = quant
        self.dynamic = dynamic
        if quant == False and dynamic == False:
            self.net = nn.Sequential(
                nn.Conv2d(3, 15, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(15, 75, 4),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(75, 375, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(1500, 400),
                nn.ReLU(),
                nn.Linear(400, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10),).to(self.device)
        elif quant == True and dynamic == False:
            self.net = nn.Sequential(
                Conv(3, 15, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                Conv(15, 75, 4),
                nn.ReLU(),
                nn.MaxPool2d(2),
                Conv(75, 375, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                Linear(1500, 400),
                nn.ReLU(),
                Linear(400, 120),
                nn.ReLU(),
                Linear(120, 84),
                nn.ReLU(),
                Linear(84, 10),).to(self.device)
        elif quant == False and dynamic == True:
            self.net = nn.Sequential(
                nn.Conv2d(3, 15, 3),
                Dynamic_Relu(),
                nn.MaxPool2d(2),
                nn.Conv2d(15, 75, 4),
                Dynamic_Relu(),
                nn.MaxPool2d(2),
                nn.Conv2d(75, 375, 3),
                Dynamic_Relu(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(1500, 400),
                Dynamic_Relu(dimensions=1),
                nn.Linear(400, 120),
                Dynamic_Relu(dimensions=1),
                nn.Linear(120, 84),
                Dynamic_Relu(dimensions=1),
                nn.Linear(84, 10),).to(self.device)
        else:
            self.net = nn.Sequential(
                Conv(3, 15, 3),
                Dynamic_Relu(),
                nn.MaxPool2d(2),
                Conv(15, 75, 4),
                Dynamic_Relu(),
                nn.MaxPool2d(2),
                Conv(75, 375, 3),
                Dynamic_Relu(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                Linear(1500, 400),
                Dynamic_Relu(dimensions=1),
                Linear(400, 120),
                Dynamic_Relu(dimensions=1),
                Linear(120, 84),
                Dynamic_Relu(dimensions=1),
                Linear(84, 10),).to(self.device)

    def forward(self, x) ->Tensor:
        """The forward function

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor： The output tensor.
        """
        x = self.net(x)
        return x

    def reconstruct(self) -> None:
        """Reconstruct the weight of the quantized layer before backward
        """
        if self.quant == True:
            for layer in self.net:
                if type(layer) == Linear or type(layer) == Conv:
                    layer.reconstruct()

    def regularizationTerm(self, reg_type, beta=1e-4):
        """The regularzation of training

        Args:
            reg_type (str): The regularization type. Use "orthogonal" or "spectral"
            beta (float, optional): The hyperparameters of the regularzation. Defaults to 1e-4.

        Raises:
            NotImplementedError: reg_type must be "orthogonal" or "spectral".

        Returns:
            float: The regularization result.
        """
        term = 0.0
        if reg_type == "orthogonal":
            for layer in self.net:
                if type(layer) == Linear or type(layer) == nn.Linear:
                    w = layer._parameters['weight']
                    w = w @ w.T
                    term += torch.norm(w-torch.eye(w.shape[0]).to(self.device))
                if type(layer) == Conv or type(layer) == nn.Conv2d:
                    w = layer._parameters['weight']
                    N, C, H, W = w.shape
                    w = w.view(N*C, H, W)
                    w = torch.bmm(w, w.permute(0, 2, 1))
                    term += torch.norm(w-torch.eye(H).to(self.device))
        elif reg_type == "spectral":
            for layer in self.net:
                if type(layer) == Linear or type(layer) == nn.Linear:
                    term += SpectralNorm.apply(layer)

                if type(layer) == Conv or type(layer) == nn.Conv2d:
                    term += SpectralNorm.apply(layer)
        else:
            raise NotImplementedError
        return term * beta


class SpectralNorm:
    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.weight = 0
        self.u = 0
        self.v = 0

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = weight
        if self.dim != 0:
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    @staticmethod
    def apply(module: nn.Module, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(
                f'`SpectralNorm` cannot be applied as parameter `{name}` is None')
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                'The module passed to `SpectralNorm` can\'t have uninitialized parameters. '
                'Make sure to run the dummy forward before applying spectral normalization')

        weight_mat = fn.reshape_weight_to_matrix(weight)
        h, w = weight_mat.size()
        u = F.normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
        v = F.normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
        weight_mat = fn.reshape_weight_to_matrix(weight)
        do_power_iteration = True
        if do_power_iteration:
            for _ in range(fn.n_power_iterations):
                v = F.normalize(torch.mv(weight_mat.t(), u),
                                dim=0, eps=fn.eps)
                u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=fn.eps)
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        return sigma


class PGD():
    """The PGD attack class
    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=4):
        """The init function of the PGD attack class

        Args:
            model (nn.Model): The model.
            eps (float, optional): The limit of the disturbance. Defaults to 8/255.
            alpha (float, optional): The magnitude of the disturbance. Defaults to 2/255.
            steps (int, optional): Number of iterations of the disturbance. Defaults to 4.
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def forward(self, images, labels, loss_fn, device='cpu'):
        """The forward function of the PGD attack class

        Args:
            images (Tensor): The images.
            labels (Tensor): The labels.
            loss_fn (nn.function): The loss function.
            device (str, optional): The device to use. Defaults to 'cpu'.

        Returns:
            Tensor: Images that are used as attacks after adding scrambling.
        """
        self.images = images.to(device)
        self.labels = labels.to(device)
        ori_images = self.images.data
        for i in range(self.steps):
            self.images.requires_grad = True
            outputs = self.model(self.images)

            self.model.zero_grad()
            loss = loss_fn(outputs, self.labels).to(device)
            self.model.reconstruct()
            loss.backward()
            adv_images = self.images + self.alpha * self.images.grad.sign()
            eta = torch.clamp(adv_images - ori_images,
                              min=-self.eps, max=self.eps)
            self.images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        return self.images


class Conv(nn.Conv2d):
    """The quantized Conv2d, have quant method and reconstruct method
    """
    def __init__(self, *args, **kwargs):
        super(Conv, self).__init__(*args, **kwargs)

    def reconstruct(self):
        """The reconstruct function, to reconstruct the orignal_weight
        """
        self.weight.data = self.orignal_weight

    def quant(self, x):
        """The quant function, to quantize a tensor

        Args:
            x (Tensor): The tensor to reconstruct

        Returns:
            Tensor: The tensor after reconstruct
        """
        x_new = torch.abs(x)
        max_v = torch.max(x_new)
        scale = max_v/torch.tensor(127)
        input = x.detach()/scale
        input = torch.round(input)
        input = torch.clip(input, -128, 127)
        return input * scale

    def forward(self, x):
        x.data = self.quant(x)
        self.orignal_weight = self.weight.data
        self.weight.data = self.quant(self.weight)
        output = self._conv_forward(x, self.weight, self.bias)
        return output


class Linear(nn.Linear):
    """The quantized Liner, have quant method and reconstruct method
    """
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)

    def reconstruct(self):
        self.weight.data = self.orignal_weight

    def quant(self, x):
        x_new = torch.abs(x)
        max_v = torch.max(x_new)
        scale = max_v/torch.tensor(127)
        input = x.detach()/scale
        input = torch.round(input)
        input = torch.clip(input, -128, 127)
        return input*scale

    def forward(self, x):
        x.data = self.quant(x)
        self.orignal_weight = self.weight.data
        self.weight.data = self.quant(self.weight)
        return F.linear(x, self.weight, self.bias)


class Dynamic_Relu(torch.nn.Module):
    '''
    from paper: https://arxiv.org/pdf/2104.03693.pdf
    Args:
        N = int - number of intervals contained in function
        momentum = float - strength of momentum during the statistics collection phase
        (Phase I in paper)
    '''

    def __init__(self, N=16, momentum=0.9, dimensions=3):
        super(Dynamic_Relu, self).__init__()
        self.N = N
        self.momentum = momentum
        self.mode = 0
        self.dimensions = dimensions
        self.Br = torch.nn.Parameter(torch.tensor(10.))
        self.Bl = torch.nn.Parameter(torch.tensor(-10.))
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.Kl = torch.nn.Parameter(torch.tensor(0.))
        self.Kr = torch.nn.Parameter(torch.tensor(1.))
        self.Yidx = torch.nn.Parameter(nn.functional.relu(
            torch.linspace(self.Bl.item(), self.Br.item(), self.N+1)))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x):
        if self.mode == 1:
            d = (self.Br-self.Bl)/self.N  # Interval length
            '''{TODO} Refactor code'''
#             Bidx = torch.linspace(self.Bl.item(),self.Br.item(),self.N)#LEFT Interval boundaries
            # Number of corresponding interval for X
            DATAind = torch.clamp(torch.floor(
                (x-self.Bl.item())/d), 0, self.N-1)
            Bdata = self.Bl+DATAind*d  # LEFT Interval boundaries
            maskBl = x < self.Bl  # Mask for LEFT boundary
            maskBr = x >= self.Br  # Mask for RIGHT boundary
            maskOther = ~(maskBl+maskBr)  # Mask for INSIDE boundaries
            Ydata = self.Yidx[DATAind.type(torch.int64)]  # Y-value for data
            Kdata = (self.Yidx[(DATAind).type(
                torch.int64)+1]-self.Yidx[DATAind.type(torch.int64)])/d  # SLOPE for data
            return maskBl*((x-self.Bl)*self.Kl+self.Yidx[0]) + maskBr*((x-self.Br)*self.Kr + self.Yidx[-1]) + maskOther*((x-Bdata)*Kdata + Ydata)
        else:
            # {TODO}: Possibly split along channel axis
            if self.dimensions == 3:
                mean = x.detach().mean([0, 1, 2, -1])
            # {TODO}: Possibly split along channel axis
                var = x.detach().var([0, 1, 2, -1])
            elif self.dimensions == 1:
                mean = x.detach().mean([0, -1])
                var = x.detach().var([0, -1])
            self.running_mean = (self.momentum * self.running_mean) + \
                (1.0-self.momentum) * mean  # .to(input.device)
            self.running_var = (self.momentum * self.running_var) + \
                (1.0-self.momentum) * (x.shape[0]/(x.shape[0]-1)*var)
            return nn.functional.relu(x)

    def setparams(self, device='cpu'):
        self.Bl = torch.nn.Parameter(
            (self.running_mean.detach() - 3*self.running_var.detach()).to(device))
        self.Br = torch.nn.Parameter(
            (self.running_mean.detach() + 3*self.running_var.detach()).to(device))
        self.Yidx = torch.nn.Parameter(nn.functional.relu(
            torch.linspace(self.Bl.item(), self.Br.item(), self.N+1)).to(device))
    
class Feature_Squeezing():
    """The feature_squeeze class
    """
    def __init__(self,mode='b',device='cpu'):
        """The init function of the class

        Args:
            mode (str, optional): The mode of feature_squeeze. Defaults to 'b'.
            device (str, optional): The device to use. Defaults to 'cpu'.
        """
        self.mode=mode
        self.device=device
        
    def forward(self,X):
        if self.mode == 'b':
            return self.bit_depth_reduction(X)
        if self.mode == 'l':
            return self.median_filter(X)
        if self.mode == 'n':
            return self.non_local_filter(X)
        
    def bit_depth_reduction(self,X,bit = 2):
        npp_int = 2**bit-1
        X = X.detach().cpu().numpy()
        x_int = np.rint(X * npp_int)
        x_float = x_int / npp_int
        return torch.tensor(x_float).to(self.device)

    def median_filter(self,X,width=2,height=-1):
        X = X.detach().cpu().numpy()
        if height == -1:
            height = width
        return torch.tensor(ndimage.filters.median_filter(X, size=(1, 1, width, height), mode='reflect')).to(self.device)
    
    def non_local_filter(self,X):
        X = X.detach().cpu().numpy()
        X = X - np.min(X)
        X = X/np.max(X)
        X = (X * 255).astype(np.uint8)
        X = np.transpose(X,(0,2,3,1))
        for i in range(X.shape[0]):
            X[i] = cv.cvtColor(X[i],cv.COLOR_RGB2BGR)
            X[i] = cv.fastNlMeansDenoisingColored(X[i])
        X = X.astype(np.float32)/255
        X = X - np.mean(X)
        X = np.transpose(X,(0,3,1,2))
        return torch.tensor(X).to(self.device)
