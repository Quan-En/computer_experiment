"""
Implementation of Linear Model of Coregionalization (LMC) with Qualitative and Quantitative Factors

Version: 2022-05-19
Author: Quan-En, Li

Reference: 
- (1). Peter Z. G. QIAN et al. (2008) Gaussian Process Models for Computer Experiments With Qualitative and Quantitative Factors
- (2). Haitao Liu et al. (2018) Remarks on multi-output Gaussian process regression

The main idea in this program is combine:
    - (1). Linear Model of Coregionalization (LMC)
    - (2). Qualitative and Quantitative Factors GP (QQGP)
    replace the GP model to QQGP in LMC

In this program have: 
- class: InitialQQLMC
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


class InitialQQLMC(nn.Module):
    def __init__(self, input, levels_num: int, output_list: list, latent_process_num: int = 2, noise_term: bool = True):
        super(InitialQQLMC, self).__init__()

        # list of dataset
        self.input = input
        self.quan_dim = self.input.shape[1] - 1
        self.levels_num = levels_num

        self.output_list = output_list
        self.obj_dim = len(output_list)
        self.output = torch.cat(output_list, dim=0)

        # device tools
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Parameters
        self.latent_process_num = latent_process_num
        
        ## scale for distance
        ### each latent process (denoted f(.)) has a set of scale parameters
        ### each quantitative factor (denoted x) have one
        self.labda_list = nn.ParameterList([
            nn.Parameter(torch.Tensor([1] * self.quan_dim), requires_grad=True)
            for _ in range(latent_process_num)
        ])

        ## correlation of qualitative factors (denoted z)
        ### relative to `QualCorr` function
        ### each latent process (denoted f(.)) has a set of correlation parameters
        L_length = int(0.5 * self.levels_num * (self.levels_num + 1))
        self.L_list = nn.ParameterList([
            nn.Parameter(torch.Tensor([1] * L_length), requires_grad=True)
            for _ in range(latent_process_num)
        ])

        # ## sigma for latent process: sqrt(cov(f_i(x), f_i(x))), each process have one
        # self.sigma_list = nn.ParameterList([
        #     nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        #     for _ in range(latent_process_num)
        # ])

        ## sigma for noise term: sqrt(cov(epsilon_t(x), epsilon_t(x))), each objective function have one object
        self.noise_term = noise_term
        if noise_term: self.noise_sigma = nn.Parameter(torch.Tensor([0.05] * self.obj_dim), requires_grad=True)

        ## weight of each latent process
        self.weight_list = nn.ParameterList([
            nn.Parameter(torch.tril(torch.ones((self.obj_dim, self.obj_dim)) * 0.5), requires_grad=True)
            for _ in range(latent_process_num)
        ])
        
        ## mu: [mu_1, mu_2, ..., mu_T]
        self.mu = nn.Parameter(torch.Tensor([0.5] * self.obj_dim), requires_grad=True)

        ## indices of lower triangular matrix
        ### use for compute qualitative correlation matrix
        self.tril_indices = torch.tril_indices(row=self.levels_num, col=self.levels_num, offset=0)

class Model(InitialQQLMC, ModelBase.Func):
    
    def __init__(self, input, levels_num: int, output_list: list, latent_process_num: int = 2, noise_term: bool = True):
        super(Model, self).__init__(input, levels_num, output_list, latent_process_num, noise_term)

    def ObjectiveCovarianve(self, input_pair_list):

        # linear combination covariance matrix of each objective function
        cov_matrix_list = [
            torch.kron(
                torch.matmul(
                    torch.tril(self.weight_list[i]),
                    torch.tril(self.weight_list[i]).t(),
                ),
                # covariance matrix of each latent process
                self.QQCorr(input_pair_list, labda=self.labda_list[i], L=self.L_list[i]),
            )
            for i in range(self.latent_process_num)
        ]
        cov_matrix = torch.stack(cov_matrix_list, dim=0).sum(dim=0)

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