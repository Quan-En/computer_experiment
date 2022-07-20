"""
Implementation of Multi-task QQGP (MTQQGP)

Version: 2022-05-19
Author: Quan-En, Li

Reference: 
- (1). Edwin V. Bonilla et al. (2007) Multi-task Gaussian Process Prediction
- (2). Kevin Swersky et al. (2013) Multi-Task Bayesian Optimization
- (3). Peter Z. G. QIAN et al. (2008) Gaussian Process Models for Computer Experiments With Qualitative and Quantitative Factors
- (4). Haitao Liu et al. (2018) Remarks on multi-output Gaussian process regression


The main idea in this program is combine:
    - (1). Multi-task GP (MTGP)
    - (2). Qualitative and Quantitative Factors GP (QQGP)
    replace the GP model to QQGP in MTGP


In this program have: 
- class: InitialMTQQGP
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

class InitialMTQQGP(nn.Module):
    def __init__(self, input, levels_num: int, output_list: list, noise_term: bool = True):
        super(InitialMTQQGP, self).__init__()

         # list of dataset
        self.input = input
        self.quan_dim = self.input.shape[1] - 1
        self.levels_num = levels_num

        self.output_list = output_list
        self.obj_dim = len(output_list)
        self.output = torch.cat(output_list, dim=0)

        # device tool
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Parameters

        ## scale for distance: each quantitative factor (denoted x) have one
        self.labda = nn.Parameter(torch.Tensor([0.01] * self.quan_dim), requires_grad=True)

        ## correlation of qualitative factor (denoted z)
        ### relative to `QualFactor_corr` function
        L_length = int(0.5 * self.levels_num * (self.levels_num + 1))
        self.L = nn.Parameter(torch.Tensor([1] * L_length), requires_grad=True)

        ## indices of lower triangular matrix
        ### use for compute qualitative correlation matrix
        self.tril_indices = torch.tril_indices(row=levels_num, col=levels_num, offset=0)

        ## sigma for noise term: sqrt(cov(epsilon_t(x), epsilon_t(x))), each objective function have one object
        self.noise_term = noise_term
        if noise_term: self.noise_sigma = nn.Parameter(torch.Tensor([0.05] * self.obj_dim), requires_grad=True)

        ## mu: [mu_1, mu_2, ..., mu_T]
        self.mu = nn.Parameter(torch.Tensor([0.5] * self.obj_dim), requires_grad=True)
        
        ## objective covariance matrix
        self.weight = nn.Parameter(torch.tril(torch.ones((self.obj_dim, self.obj_dim)) * 0.2), requires_grad=True)


class Model(InitialMTQQGP, ModelBase.Func):
    
    def __init__(self, input, levels_num: int, output_list: list, noise_term: bool = True):
        super(Model, self).__init__(input, levels_num, output_list, noise_term)

    def ObjectiveCovarianve(self, input_pair_list):

        # correlation matrix
        corr_matrix = self.QQCorr(
            w_pair_list=input_pair_list,
            labda=self.labda,
            L=self.L,
        )

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