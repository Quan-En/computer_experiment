"""
Several functions

Version: 2022-03-05
Author: Quan-En, Li

In this program have: 
- function: 
    collect_efficient_solutions(candidate_set:list) -> list
    count_parameters(model) -> int
    print_model_parameters(model) -> null
    get_model_parameters_dict(model) -> dict
    get_sliced_response_f(w, function_list) -> list
    data_generate(seed, size, range_list=[[-4.0, 4.0], [-4.0, 4.0]]) -> ndarray
    sliced_data_generate(kwargs_list, function_list) -> ndarray, list
    real_data_generate_with_slhd(fname) -> ndarray, list
    update_data(model, new_w: torch.Tensor, new_y_list: list) -> torch.Tensor, list
    save_object(obj, name:str) -> null
    load_object(name:str) -> dict
    PasteFileName(lr:str, low_beta:str, up_beta='0999') -> str
    ReadFileGetMetrics(DataPath) -> ndarray
    UnzipMeanCovar(all_info: list) -> torch.Tensor, torch.Tensor
    add_random_noise(arr, seed=0, scale=0.05) -> ndarray

"""

from matplotlib.pyplot import grid
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize

from paretoset import paretoset
from collections import namedtuple

from smt.sampling_methods import LHS  # Latin Hyper Cube Design

import pickle

def collect_efficient_solutions(candidate_set):

    # Check dimension
    q_dim = candidate_set[0].reshape(-1).shape[0]

    # Create Solution objects holding the problem solution and objective values
    Solution = namedtuple("Solution", ["solution", "obj_value"])
    solutions = [
        Solution(solution=object, obj_value=solution.reshape(-1))
        for  solution in candidate_set
    ]

    # Create an array of shape (solutions, objectives) and compute the non-dominated set
    objective_values_array = np.vstack([s.obj_value for s in solutions])
    mask = paretoset(objective_values_array, sense=["min"] * q_dim)

    # Filter the list of solutions, keeping only the non-dominated solutions
    efficient_solutions = [solution for (solution, m) in zip(solutions, mask) if m]
    efficient_solutions = list(map(lambda x: x.obj_value, efficient_solutions))

    return efficient_solutions

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total num of parms: ", pytorch_total_params)
    print("\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "labda" in name:
                print(name, "---->", np.exp(param.data.detach().cpu().numpy()))
            elif "sigma" in name:
                print(name, "---->", param.data.detach().cpu().numpy() ** 2)
            elif "L" in name:
                print("QualFactor correlation", "---->", model.QualFactor_corr().detach().cpu().numpy())
            elif "weight" in name and "list" not in name:
                print("Task correlation", "---->", torch.matmul(torch.tril(model.weight_list), torch.tril(model.weight_list).t()).detach().cpu().numpy())
            else:
                print(name, "---->", param.data.detach().cpu().numpy())

def get_model_parameters_dict(model):
    output = dict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "labda" in name:
                output[name] = np.exp(param.data.detach().cpu().numpy())
            elif "sigma" in name:
                output[name] = param.data.detach().cpu().numpy() ** 2
            elif "weight" in name and "list" not in name:
                output["TaskCovar"] = torch.matmul(torch.tril(model.weight), torch.tril(model.weight).t()).detach().cpu().numpy()
            elif "L" in name:
                L = torch.exp(param.data.detach().cpu())
                L_len = len(L)
                levels_num = int((-1 + np.sqrt(1 + 8*L_len)) / 2)
                result = torch.zeros((levels_num, levels_num))
                tril_indices = torch.tril_indices(levels_num, levels_num)
                result[tril_indices[0], tril_indices[1]] = L
                output[name] = result.numpy()
            else:
                output[name] = param.data.detach().cpu().numpy()
    return output

def get_sliced_response_f(w, function_list):
    
    z = w[:, 0]
    x = w[:, 1:]
    uniq_z = np.unique(z)

    responses_list = [function(x) for function in function_list]

    sliced_list = [
        (y1 * (z == sub_z), y2 * (z == sub_z))
        for sub_z, (y1, y2) in zip(uniq_z, responses_list)
    ]

    y_list = [
        np.sum(np.column_stack([sub_sliced[0] for sub_sliced in sliced_list]), axis=1),
        np.sum(np.column_stack([sub_sliced[1] for sub_sliced in sliced_list]), axis=1),
    ]

    return y_list

def data_generate(seed, size, range_list=[[-4.0, 4.0], [-4.0, 4.0]]):

    xlimits = np.array(range_list)

    sampling = LHS(xlimits=xlimits, criterion="maximin", random_state=seed)

    generated_x = sampling(size)

    return generated_x

def sliced_data_generate(kwargs_list, function_list):

    w_list = []
    for index, kwargs in enumerate(kwargs_list):

        generated_x = data_generate(**kwargs)
        generated_z = np.array([index+1] * kwargs["size"])

        w_list.append(np.column_stack([generated_z, generated_x]))
    
    train_w = np.row_stack(w_list)
    train_y_list = get_sliced_response_f(train_w, function_list)
    return train_w, train_y_list

def real_data_generate(seed, size, algorithm_type_list=["Adam", "Adam_unweighted"]):

    w_list = []
    y_list = []

    for i, algorithm_type in enumerate(algorithm_type_list):
        # Generate w-data
        sub_x = data_generate(seed=seed+i, size=size, range_list=[[0.0005, 0.010], [0.80, 0.90]])
        sub_x = np.column_stack([0.0005 * np.around(sub_x[:,0] / 0.0005), np.around(sub_x[:,1], 2)])
        sub_z = np.array([i+1] * size)
        sub_w = np.column_stack([sub_z, sub_x])

        # Get all (learning rate, lower beta) to read Resnet metrics
        x_lr = [format(lr, "8f").replace(".", "") for lr in sub_x[:,0].tolist()]
        x_low_beta = [str(low_beta).replace(".", "") for low_beta in sub_x[:,1].tolist()]

        # Read Resnet metrics 
        sub_y = np.row_stack([
            ReadFileGetMetrics("resnet_metrics_TrainTest/" + algorithm_type + "/" + PasteFileName(lr, low_beta))
            for lr, low_beta in zip(x_lr, x_low_beta)
        ])

        w_list.append(sub_w)
        y_list.append(sub_y)

    w_arr = np.row_stack(w_list)
    y_arr = np.row_stack(y_list)
    
    return w_arr, [y_arr[:,0], y_arr[:,1]]

def real_data_generate_with_slhd(fname):
    rescale_range_list = [[0.0005, 0.01], [0.80, 0.90]]
    algorithm_type_list = ["Adam", "Adam_unweighted", "Adamax", "Adamax_unweighted"]
    standard_arr = np.loadtxt(fname)

    # Recale
    for i, sub_range in enumerate(rescale_range_list):
        standard_arr[:,i+1] = (sub_range[1] - sub_range[0]) * (standard_arr[:,i+1])  + sub_range[0]
    
    # Round to grid
    standard_arr[:,1] = 0.0005 * np.around(standard_arr[:,1] / 0.0005)
    standard_arr[:,2] = np.around(standard_arr[:,2], 2)

    # Read Resnet metrics
    info_list = [[algorithm_type_list[int(z)-1], format(lr, "8f").replace(".", ""), str(low_beta).replace(".", "")] for z, lr, low_beta in standard_arr]
    y_arr = np.row_stack([
        ReadFileGetMetrics("../resnet_metrics_TrainTest/" + algorithm_type + "/" + PasteFileName(lr, low_beta)) 
        for algorithm_type, lr, low_beta in info_list
    ])

    return standard_arr, [y_arr[:,0], y_arr[:,1]]

def real_data_generate_with_slhd_z5x1(fname):
    rescale_range_list = [[0.0005, 0.01], ]
    algorithm_type_list = ["Adam", "Adamax", "NAdam", "RMSprop", "SGD"]
    standard_arr = np.loadtxt(fname)

    # Recale
    for i, sub_range in enumerate(rescale_range_list):
        standard_arr[:,i+1] = (sub_range[1] - sub_range[0]) * (standard_arr[:,i+1])  + sub_range[0]
    
    # Round to grid
    standard_arr[:,1:] = 0.0005 * np.around(standard_arr[:,1:] / 0.0005)

    # Read Resnet metrics
    info_list = [
        [algorithm_type_list[int(z)-1], format(lr, "8f").replace(".", "")]
        for z, lr in standard_arr
    ]

    y_arr_list=[]
    for algorithm_type, lr in info_list:
        if algorithm_type in ["Adam", "Adamax", "NAdam"]:
            y_arr_list.append(ReadFileGetMetrics("../resnet_metrics_TrainTest/" + algorithm_type + "/" + PasteFileName(lr, "09")))
        elif algorithm_type == "RMSprop":
            y_arr_list.append(ReadFileGetMetrics("../resnet_metrics_TrainTest/RMSprop/lr_" + lr + "_metrics.csv"))
        else:
            y_arr_list.append(ReadFileGetMetrics("../resnet_metrics_TrainTest/SGD/lr_" + lr + "_momentum_09_metrics.csv"))

    y_arr = np.row_stack(y_arr_list)

    return standard_arr, [y_arr[:,0], y_arr[:,1]]

def Round2Closest(value, arr):
    return arr[np.argmin((arr - value) ** 2)]

def numerical_data_generate_with_slhd(fname, rescale_range_list=[[-4.0, 4.0], [-4.0, 4.0]], grid_size=40):
    
    standard_arr = np.loadtxt(fname)
    x_list = [np.linspace(l, u, num=grid_size) for (l, u) in rescale_range_list]

    # Recale
    for j, sub_range in enumerate(rescale_range_list):
        standard_arr[:,j+1] = (sub_range[1] - sub_range[0]) * (standard_arr[:,j+1])  + sub_range[0]
    
    # Round to grid
    for j, sub_arr in enumerate(x_list):
        for i, num in enumerate(standard_arr[:,j+1]):
            standard_arr[i,j+1] = Round2Closest(num, sub_arr)

    return standard_arr

def numerical_index_generate_with_slhd(fname, grid_size):

    standard_arr = np.loadtxt(fname)
    standard_arr[:,1:] = (grid_size - 1) * standard_arr[:,1:]
    standard_arr = np.round(standard_arr)

    return standard_arr.astype(int)

def update_data(model, new_w: torch.Tensor, new_y_list: list):

    new_train_w = torch.cat((model.input, new_w), dim=0).float()

    new_train_y_list = []
    for y, new_y in zip(model.output_list, new_y_list):
        new_train_y_list.append(torch.cat((y, new_y), dim=0).float())
    
    return new_train_w, new_train_y_list

def save_object(obj, name:str):
    """Save object to specific path (.pkl)"""
    with open(name, "wb+") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(name:str):
    """Load object from specific path (.pkl)"""
    with open(name, "rb") as f:
        return pickle.load(f)

def PasteFileName(lr:str, low_beta:str, up_beta='0999'):
    return "lr_" + lr + "_betas_" + low_beta + "_" + up_beta + "_metrics.csv"

def ReadFileGetMetrics(DataPath):
    
    DataFrame = pd.read_csv(DataPath)
    Cond = (DataFrame['data_type'] == "test") & (DataFrame['epochs'] == 49)
    
    return  1 - DataFrame[Cond][["macro_f1", "micro_f1"]].to_numpy().reshape(-1)

def UnzipMeanCovar(all_info: list):
    """For two objective"""
    Means, Vars, Covars = all_info
    size = int(Means.shape[0] / 2)
    
    # y_means (size, objective size)
    # objective size = 2
    y_means = torch.column_stack([Means[:size], Means[size:]])
    
    # y_covars (size, objective size, objective size)
    # objective size = 2
    y_covars = torch.column_stack([Vars[:size], Covars, Covars, Vars[size:]]).reshape(-1, 2, 2)
    
    return y_means, y_covars

def add_random_noise(arr, seed=0, scale=0.05):
    np.random.seed(seed)
    return arr + np.random.normal(loc=0, scale=scale, size=arr.shape)

