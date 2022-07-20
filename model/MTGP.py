"""
Implementation of Multi-task GP (MTGP)

Version: 2022-05-19
Author: Quan-En, Li

Reference: 
- (1). Edwin V. Bonilla et al. (2007) Multi-task Gaussian Process Prediction
- (2). Kevin Swersky et al. (2013) Multi-Task Bayesian Optimization


In this program have: 
- class: InitialMTGP
- class: Model
- function: ModelTraining(model, optimizer, epoch: int, print_loss=True)


Environment and module version: 
- python : 3.7.4
- torch version: 1.10.2

"""

import numpy as np
import torch
import torch.nn as nn
from model import ModelBase


class InitialMTGP(nn.Module):
    def __init__(self, input, output_list: list, noise_term: bool =True):
        super(InitialMTGP, self).__init__()

        # list of dataset
        self.input = input
        self.quan_dim = input.shape[1]
        self.output_list = output_list
        self.obj_dim = len(output_list)
        self.output = torch.cat(output_list, dim=0)
        
        # device tool
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Parameters

        ## scale for distance: each factor (denoted x) have one
        self.labda = nn.Parameter(torch.Tensor([1] * self.quan_dim), requires_grad=True)

        ## sigma for noise term: sqrt(cov(epsilon_t(x), epsilon_t(x))), each objective function have one object
        self.noise_term = noise_term
        if noise_term: self.noise_sigma = nn.Parameter(torch.Tensor([0.05] * self.obj_dim), requires_grad=True)

        ## objective covariance matrix
        self.weight = nn.Parameter(torch.ones((self.obj_dim, self.obj_dim), device=self.device) * 0.5, requires_grad=True)
        
        ## mu: [mu_1, mu_2, ..., mu_T]
        self.mu = nn.Parameter(torch.Tensor([0.5] * len(output_list)), requires_grad=True)

        ## keyword arguments of objective covariance function
        self.obj_corr_kwargs = {"quan_dim":self.quan_dim, "labda":self.labda}

class Model(InitialMTGP, ModelBase.Func):
    
    def __init__(self, input, output_list: list, noise_term: bool = True):
        super(Model, self).__init__(input, output_list, noise_term)

    def ObjectiveCovarianve(self, input_pair_list):

        # correlation matrix
        corr_matrix = self.QuanCorr(x_pair_list=input_pair_list, labda=self.labda)

        # covariance matrix of each objective function
        cov_matrix = torch.kron(
            torch.matmul(
                torch.tril(self.weight),
                torch.tril(self.weight).t(),
            ),
            corr_matrix,
        )

        return cov_matrix

def ModelTraining(model, optimizer, epoch: int, print_loss=True):
    for i in range(epoch):
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # Output from model
        loss = model.NegativeMarginalLogLikelihood()
        
        # Print statement
        if print_loss and (i == 0 or (i + 1) % 50 == 0): 
            print("iter: ", i + 1, "loss: ", np.round(loss.item(), 4))
        
        # Update parameters
        loss.backward()
        optimizer.step()