"""
Implementation of Collaborative Multi-Output Gaussian Process(CoMOGP)

Version: 2022-05-19
Author: Quan-En, Li

Reference: 
- (1). Haitao Liu et al. (2018) Remarks on multi-output Gaussian process regression

In this program have: 
- class: InitialCoMOGP
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


class InitialCoMOGP(nn.Module):
    def __init__(self, input, output_list: list, latent_process_num: int = 1, noise_term: bool = True):
        super(InitialCoMOGP, self).__init__()

        # list of dataset
        self.input = input
        self.quan_dim = input.shape[1]
        self.output_list = output_list
        self.obj_dim = len(output_list)
        self.output = torch.cat(output_list, dim=0)

        # device tools
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Parameters
        self.latent_process_num = latent_process_num
        
        ## scale for distance (1): each latent process have one object

        self.labda_l_list = nn.ParameterList([
            nn.Parameter(torch.Tensor([-0.1] * self.quan_dim), requires_grad=True)
            for _ in range(latent_process_num)
        ])
        
        ## scale for distance (2): each objective function have one object
        self.labda_o_list = nn.ParameterList([
            nn.Parameter(torch.Tensor([0.1] * self.quan_dim), requires_grad=True)
            for _ in range(self.obj_dim)
        ])

        ## sigma_l: sqrt(cov(u_i(x), u_i(x))), each latent process have one object
        self.sigma_l_list = nn.ParameterList([
            nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
            for _ in range(latent_process_num)
        ])
        
        ## sigma_o: sqrt(cov(v_i(x), v_i(x))), each objective function have one object
        self.sigma_o_list = nn.ParameterList([
            nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
            for _ in range(self.obj_dim)
        ])

        ## sigma for noise term: sqrt(cov(epsilon_t(x), epsilon_t(x))), each objective function have one object
        self.noise_term = noise_term
        if noise_term: self.noise_sigma = nn.Parameter(torch.Tensor([0.05] * self.obj_dim), requires_grad=True)

        ## weight of each latent process
        self.weight_list = nn.ParameterList([
            nn.Parameter(torch.ones((self.obj_dim, ),device=self.device) * 0.5, requires_grad=True)
            for _ in range(latent_process_num)
        ])
        
        ## mu: [mu_1, mu_2, ..., mu_T]
        self.mu = nn.Parameter(torch.Tensor([0.5] * self.obj_dim), requires_grad=True)

class Model(InitialCoMOGP, ModelBase.Func):
    
    def __init__(self, input, output_list: list, latent_process_num: int = 1, noise_term: bool = True):
        super(Model, self).__init__(input, output_list, latent_process_num, noise_term)

    def ObjectiveCovarianve(self, input_pair_list):

        # linear combination covariance matrix of each objective function
        latent_cov_matrix_list = [
            torch.kron(
                self.weight_list[i].reshape(-1,1) @ self.weight_list[i].reshape(1,-1),
                # covariance matrix of each latent process
                (self.sigma_l_list[i] ** 2) * self.QuanCorr(x_pair_list=input_pair_list, labda=self.labda_l_list[i]),
            )
            for i in range(self.latent_process_num)
        ]
        latent_cov_matrix = torch.stack(latent_cov_matrix_list, dim=0).sum(dim=0)

        # covariance matrix list
        special_cov_matrix_list=[
            (self.sigma_o_list[i] ** 2) * self.QuanCorr(x_pair_list=input_pair_list, labda=self.labda_o_list[i])
            for i in range(self.obj_dim)
        ]
        # special correlation matrix
        special_cov_matrix = torch.block_diag(*special_cov_matrix_list)

        cov_matrix = latent_cov_matrix + special_cov_matrix
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