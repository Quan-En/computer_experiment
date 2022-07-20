
"""
Basic functions of Gaussian process model

Version: 2022-05-25
Author: Quan-En-Li

Reference: 
- (1). Haitao Liu et al. (2018) Remarks on multi-output Gaussian process regression
- (2). Peter Z. G. QIAN et al. (2008) Gaussian Process Models for Computer Experiments With Qualitative and Quantitative Factors
- (3). Edwin V. Bonilla et al. (2007) Multi-task Gaussian Process Prediction
- (4). Kevin Swersky et al. (2013) Multi-Task Bayesian Optimization
"""


import torch

class Func(object):
    def __init__(self, ):
        self.quan_dim = None
        self.levels_num = None

        self.input = None
        self.output = None
        self.output_list = None

        self.mu = None
        self.labda = None
        self.labda_list = None

        self.noise_term = None
        self.noise_sigma = None

        self.L = None
        self.L_list = None

        # self.indices_list = None
        self.tril_indices = None

        self.device = None

    def DataSplit(self, w):
        z, x = w[:, 0].reshape(-1), w[:, 1:].reshape(-1, self.quan_dim)
        return z, x

    # Qualitative correlation
    def QualCorr(self, z_pair_list, **kwargs):
        ## use matrix L be lower triangular matrix
        ## let `correlation matrix = LL^t` is positive define matrix with unit diagonal constrained
        ## `correlation matrix` with unit diagonal <=> L with row unit vector

        un_unit_L = torch.zeros((self.levels_num, self.levels_num), device=self.device)

        ### fill up lower triangular matrix
        un_unit_L[self.tril_indices[0], self.tril_indices[1]] = torch.exp(kwargs["L"])

        ### normalize
        un_unit_L = torch.nn.functional.normalize(un_unit_L, dim=1)
        
        ### correlation matrix
        qual_corr_matrix = torch.matmul(un_unit_L, un_unit_L.t())

        ### indices
        Z1, Z2 = torch.meshgrid(z_pair_list[0].long(), z_pair_list[1].long(), indexing="ij")
        Z1, Z2 = Z1-1, Z2-1 # since z isstart from 1,2,...,m

        return qual_corr_matrix[Z1, Z2]

    # Quantitative correlation
    def QuanCorr(self, x_pair_list, **kwargs):

        # observation size
        n1, n2 = x_pair_list[0].shape[0], x_pair_list[1].shape[0]

        X1, X2 = x_pair_list[0].repeat(1, n2).view(-1, self.quan_dim), x_pair_list[1].repeat(n1, 1)

        quan_corr_matrix = torch.matmul(
            ((X1 - X2) ** 2).float(),
            torch.exp(kwargs["labda"]).reshape(-1,1),
        )
        quan_corr_matrix = torch.exp(-0.5 * quan_corr_matrix)
        quan_corr_matrix = quan_corr_matrix.view(n1, n2)

        return quan_corr_matrix

    # Correlation: qualitative correlation times quantitative correlation
    def QQCorr(self, w_pair_list, **kwargs):

        # transfer to z (qualitative factor) and x (quantitative factor)
        z1, x1 = self.DataSplit(w_pair_list[0])
        z2, x2 = self.DataSplit(w_pair_list[1])
        
        # correlation matrix of x (quantitative factor)
        quan_corr_matrix = self.QuanCorr(x_pair_list=[x1, x2], **kwargs)

        # correlation matrix of z (qualitative factor)
        qual_corr_matrix = self.QualCorr(z_pair_list=[z1, z2], **kwargs)

        # correlation matrix of w
        corr_matrix = qual_corr_matrix * quan_corr_matrix

        return corr_matrix
    
    def AddNoise(self, size):
        noise_block_diag_matrix = torch.kron(
            torch.diag(self.noise_sigma ** 2),
            torch.eye(size, device=self.device)
        )
        return noise_block_diag_matrix

    # calculate negative marginal log-likelihood
    def NegativeMarginalLogLikelihood(self):

        cov_matrix = self.ObjectiveCovarianve([self.input, self.input])

        # add noise variance covariance matrix
        if self.noise_term:
            noise_block_diag_matrix = self.AddNoise(self.input.shape[0])
            cov_matrix = cov_matrix + noise_block_diag_matrix

        # add small value to prevent sigular
        cov_matrix = cov_matrix + torch.eye(cov_matrix.shape[0], device=self.device) * 1e-6
        
        inverse_cov_matrix = torch.inverse(cov_matrix)

        # mu vector
        vec_mu = torch.kron(
            self.mu.reshape(-1, 1),
            torch.ones((self.input.shape[0], 1), device=self.device),
        )

        # (y - mu)^t inv(Sigma) (y - mu)
        left_side = torch.matmul(
            self.output.view(-1, 1).t() - vec_mu.t(),
            torch.matmul(inverse_cov_matrix, self.output.view(-1, 1) - vec_mu),
        )

        right_side = torch.log(torch.det(cov_matrix) + 1)

        nmll = 0.5 * (left_side + right_side)

        return nmll.view(-1)

    def JointMean(self, pred_size):
        """Calculate mean vector"""

        ## concatenate of mean vector of each objective according training size
        vec_mu = torch.kron(
            self.mu.reshape(-1, 1),
            torch.ones((self.input.shape[0], 1), device=self.device),
        )

        ## concatenate of mean vector of each objective according prediction size
        new_vec_mu = torch.kron(
            self.mu.reshape(-1, 1),
            torch.ones((pred_size, 1), device=self.device),
        )
        return vec_mu, new_vec_mu

    def JointCovariance(self, new_input):
        """Calculate covariance matrix"""

        # shape(train_input, train_input)
        train_cov_matrix = self.ObjectiveCovarianve([self.input, self.input])

        # shape(train_input, new_input)
        input_to_new_input_cov_matrix = self.ObjectiveCovarianve([self.input, new_input])
        
        # shape(new_input, new_input)
        new_input_cov_matrix = self.ObjectiveCovarianve([new_input, new_input])

        # consider noise
        if self.noise_term:train_cov_matrix = train_cov_matrix + self.AddNoise(self.input.shape[0])

        # add small value to prevent singular
        train_cov_matrix = train_cov_matrix + torch.eye(train_cov_matrix.shape[0], device=self.device) * 1e-6
        new_input_cov_matrix = new_input_cov_matrix + torch.eye(new_input_cov_matrix.shape[0], device=self.device) * 1e-6

        return train_cov_matrix, input_to_new_input_cov_matrix, new_input_cov_matrix

    def ConditionalDistribution(self, new_input, with_noise=False):
        """Calculate conditional mean/variance based on Gaussian's formula"""

        # covariance matrix
        train_cov_matrix, input_to_new_input_cov_matrix, new_input_cov_matrix = self.JointCovariance(new_input)
        
        # mean vector
        vec_mu, new_vec_mu = self.JointMean(pred_size=new_input.shape[0])

        # inverse of covariance matrix
        inverse_train_cov_matrix = torch.inverse(train_cov_matrix)

        # conditional means of new_input
        cond_new_input_mu = new_vec_mu + torch.matmul(
            input_to_new_input_cov_matrix.t(),
            torch.matmul(
                inverse_train_cov_matrix,
                (self.output.view(-1, 1) - vec_mu),
            ),
        )
        cond_new_input_mu = cond_new_input_mu.view(-1)
        
        # coditional variances of new_input
        cond_new_input_cov_matrix = new_input_cov_matrix - torch.matmul(
            input_to_new_input_cov_matrix.t(),
            torch.matmul(
                inverse_train_cov_matrix,
                input_to_new_input_cov_matrix,
            ),
        )

        # consider noise
        if with_noise and self.noise_term:
            cond_new_input_cov_matrix = cond_new_input_cov_matrix + self.AddNoise(new_input.shape[0])

        return cond_new_input_mu, cond_new_input_cov_matrix

    def PredDistribution(self, new_input):
        """
        Decide the format of output
        
        if number of objective equal to 2, then return means, variances and covariances of objective pair
            variance=[
                variance of objective 1,
                variance of objective 2,
            ]
            covariance=[
                cov(objective 1, objective 2),
            ]
        
        else if number of objective equal to 3, then return means, variances and covariances of each objective pair
            variance=[
                variance of objective 1,
                variance of objective 2,
            ]
            covariance=[
                cov(objective 1, objective 2),
                cov(objective 1, objective 3),
                cov(objective 2, objective 3),
            ]
        else number of objective great than 3, then return  means, covariances
            covariance=matrix[
                cov_matrix(objective 1, objective 1),                   ...                 cov_matrix(objective 1, objective T),
                cov_matrix(objective 2, objective 1),           .
                            .                                       .                                       .
                            .                                           .                                   .
                            .                                                                               .
                cov_matrix(objective T, objective 1),                   ...                 cov_matrix(objective T, objective Y),
            ]
        """
        with torch.no_grad():

            new_point_mu, new_point_sigma_square = self.ConditionalDistribution(new_input)

            if len(self.output_list) == 2:

                index = int(0.5 * new_point_sigma_square.shape[0])
                new_point_covariance = new_point_sigma_square[index:, :index]
                new_point_covariance = torch.diag(new_point_covariance)
                # new_point_covariance[new_point_covariance < 1e-2] = 0

                new_point_sigma_square = torch.diag(new_point_sigma_square)
                new_point_sigma_square[new_point_sigma_square < 1e-6] = 0

                return new_point_mu, new_point_sigma_square, new_point_covariance
            
            elif len(self.output_list) == 3:
                index_1 = int(new_point_sigma_square.shape[0] / 3)
                index_2 = index_1 * 2

                cov_f1_f2 = new_point_sigma_square[:index_1, index_1:index_2].diag()
                cov_f1_f3 = new_point_sigma_square[:index_1, index_2:].diag()
                cov_f2_f3 = new_point_sigma_square[index_1:index_2, index_2:].diag()

                new_point_sigma_square = new_point_sigma_square.diag()
                new_point_sigma_square[new_point_sigma_square < 1e-6] = 0

                return new_point_mu, new_point_sigma_square, [cov_f1_f2, cov_f1_f3, cov_f2_f3]
            elif len(self.output_list) == 1:
                return new_point_mu, new_point_sigma_square.diag()
            else:
                return new_point_mu, new_point_sigma_square

