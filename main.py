import torch
from dataset import CIFAR10, Adversarial_examples
from model import Covnet,Dynamic_Relu
from train import train, test,show_img
from torch import nn, optim
from torch.utils.data import DataLoader
import os
from seed import setup_seed
import argparse
import sys

if __name__ == '__main__':

   args = argparse.ArgumentParser()
   args.add_argument('-v', '--verbose', dest='verbose',
                     default=False, action='store_true')
   args.add_argument('-r', '--random', dest='random',
                     default=False, action='store_true')
   args.add_argument('-o', '--output', dest='output',
                     default='results.txt', type=str)
   args.add_argument('-no_report',default= False,dest='no_report', action='store_true')
   args.add_argument('-d', '--device', dest='device', default='cpu', type=str)
   args.add_argument('-b', '--batch_size',
                     dest='batch_size', default=32, type=int)
   args.add_argument('-lr', '--learning_rate',
                     dest='learning_rate', default=1e-3, type=float)
   args.add_argument('-n', '--num_of_epochs',
                     dest='num_of_epochs', default=10, type=int)
   args.add_argument('-ad', '--adversarial', dest='adversarial',
                     default=False, action='store_true')
   args.add_argument('-reg', '--regularization',
                     dest='regularization', default=None, type=str)
   args.add_argument('-beta', '--beta', dest='beta',default=1e-4,type=float)
   args.add_argument('-q', '--quantize', dest='quant',
                     default=False, action='store_true')
   args.add_argument('-dy', '--dynamic', dest='dynamic',
                     default=False, action='store_true')
   args.add_argument('-fs','--feature_squeeze',dest='feature_squeeze',default=None,type=str)
   args.add_argument('-s', '--save', dest='save',
                     default=False, action='store_true')
   args.add_argument('-l', '--load', dest='load',
                     default=False, action='store_true')
   args.add_argument('-eps', dest='eps', default=8/255, type=float)
   args.add_argument('-alpha', dest='alpha', default=2/255, type=float)
   args.add_argument('-step', dest='step', default=4, type=int)
   args.add_argument('-show', dest='show', default=False, action='store_true')
   args = args.parse_args()

   # set seed and initalize parameters
   if args.random is False:
      setup_seed(1)
   device = args.device
   batch_size = args.batch_size
   lr = args.learning_rate
   num_of_epochs = args.num_of_epochs
   loss = nn.CrossEntropyLoss()
   net = Covnet(device=device, quant=args.quant, dynamic=args.dynamic)
   optimizer = optim.Adam(net.parameters(), lr=lr)

   # set the file name and enviroument name
   file_name = 'checkpoint/checkpoint'
   evn = ''
   if args.adversarial is True:
      file_name += '_adversarial'
      evn += 'adversarial_'
   if args.regularization is not None:
      file_name += ('_' + args.regularization)
      evn += (args.regularization + '_')
   if args.quant is True:
      file_name += '_quant'
      evn += 'quant_'
   if args.dynamic is True:
      file_name += '_dynamic'
      evn += 'dynamic_'
   if args.feature_squeeze is not None:
      evn += (args.feature_squeeze + '_')
   file_name += '.pt'
   if evn == '':
      evn = 'default_'

   # load the dataset
   dataset = CIFAR10(is_transform=True, batch_size=batch_size)
   train_loader = dataset.train_set
   test_loader = dataset.test_set

   # traing
   if args.adversarial is False:
      if args.load is False:
         best_acc = 0.0
         for epoch in range(num_of_epochs):
            if args.dynamic is True and epoch == 5:
               for layer in net.net:
                  if type(layer) == Dynamic_Relu:
                     layer.setparams(device=args.device)
                     layer.mode = 1
            if args.verbose is True:
               print(f"Epoch {epoch+1}\n-------------------------------")
            train(train_loader, net, loss, optimizer,
                  regularization=args.regularization, device=device, verbose=args.verbose,beta=args.beta)
            acc = test(test_loader, net, loss, device=device, verbose=args.verbose)
            if args.save is True:
               if best_acc < acc:
                  best_acc = acc
                  torch.save(net.state_dict(), file_name)
      else:
         net.load_state_dict(torch.load(file_name))
   else:
      net.load_state_dict(torch.load(file_name.replace('_adversarial', '')))
      net = Covnet(device=device, quant=args.quant, dynamic=args.dynamic)
      optimizer = optim.Adam(net.parameters(), lr=lr)
      if args.load is False:
         best_acc = 0.0
         for epoch in range(num_of_epochs):
            if args.dynamic is True and epoch == 5:
               for layer in net.net:
                  if type(layer) == Dynamic_Relu:
                     layer.setparams(device=args.device)
                     layer.mode = 1
            if args.verbose is True:
               print(f"Epoch {epoch+1}\n-------------------------------")
            train(train_loader, net, loss, optimizer, adversarial=True,
                  regularization=args.regularization, device=device, verbose=args.verbose,beta=args.beta)
            acc = test(test_loader, net, loss, device=device, verbose=args.verbose)
            if args.save is True:
               if best_acc < acc:
                  best_acc = acc
                  torch.save(net.state_dict(), file_name)
      else:
            net.load_state_dict(torch.load(file_name))
   if os.path.exists(file_name) and args.save is True:
      net.load_state_dict(torch.load(file_name))
   
   # generate adversarial examples
   if args.dynamic is True:
       for layer in net.net:
           if type(layer) == Dynamic_Relu:
               layer.mode = 1
   ad_data_dataset = Adversarial_examples(
      test_loader, net, loss, device=device)
   ad_data_loader = DataLoader(ad_data_dataset, batch_size=batch_size)
   
   # test and report the results
   ori_stdout = sys.stdout
   if args.no_report is False:
      with open('results.txt', 'a') as f:
         sys.stdout = f
         print('----------' + evn + '-------------')
         if args.feature_squeeze is not None:
            test(test_loader, net, loss, feature_squeeze=args.feature_squeeze,device=device)
            test(ad_data_loader, net, loss, feature_squeeze=args.feature_squeeze,device=device)
         print('----------------------------------')
      sys.stdout = ori_stdout
      
   # show the image of attack results
   if args.show is True:
      for X,y in test_loader:
         X = X.to(device)
         y = y.to(device)
         show_img(X,y,net,loss,device=device,output=evn)
         break