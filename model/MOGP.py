"""
Implementation of MOGP (Multi-Output Gaussian Process)

Version: 2022-05-19
Author: Quan-En, Li

In this program have: 
- class: InitialMOGP
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

class InitialMOGP(nn.Module):
    def __init__(self, input, output_list: list, noise_term: bool = True):
        super(InitialMOGP, self).__init__()

        # list of dataset
        self.input = input
        self.quan_dim = input.shape[1]
        self.output_list = output_list
        self.obj_dim = len(output_list)
        self.output = torch.cat(output_list, dim=0)

        # device tools
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Parameters
        
        ## scale for distance : each objective function have one object
        self.labda_list = nn.ParameterList([
            nn.Parameter(torch.Tensor([0.1] * self.quan_dim), requires_grad=True)
            for _ in range(self.obj_dim)
        ])
        
        ## sigma: sqrt(cov(v_i(x), v_i(x))), each objective function have one object
        self.sigma_list = nn.ParameterList([
            nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
            for _ in range(self.obj_dim)
        ])

        ## sigma for noise term: sqrt(cov(epsilon_t(x), epsilon_t(x))), each objective function have one object
        self.noise_term = noise_term
        if noise_term: self.noise_sigma = nn.Parameter(torch.Tensor([0.05] * self.obj_dim), requires_grad=True)
        
        ## mu: [mu_1, mu_2, ..., mu_T]
        self.mu = nn.Parameter(torch.Tensor([0.5] * self.obj_dim), requires_grad=True)

class Model(InitialMOGP, ModelBase.Func):
    
    def __init__(self, input, output_list: list, noise_term: bool = True):
        super(Model, self).__init__(input, output_list, noise_term)

    def ObjectiveCovarianve(self, input_pair_list):

        # covariance matrix list
        cov_matrix_list=[
            (self.sigma_list[i] ** 2) * self.QuanCorr(x_pair_list=input_pair_list, labda=self.labda_list[i])
            for i in range(self.obj_dim)
        ]
        # correlation matrix
        cov_matrix = torch.block_diag(*cov_matrix_list)

        return cov_matrix

def ModelTraining(model, optimizer, epoch: int, print_loss=True):
    for i in range(epoch):
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # Output from model
        loss = model.NegativeMarginalLogLikelihood()
        
        # Print statement
        if print_loss and (i == 0 or (i + 1) % 100 == 0):
            print("iter: ", i + 1, "loss: ", np.round(loss.item(), 4))
        
        # Update parameters
        loss.backward()
        optimizer.step()