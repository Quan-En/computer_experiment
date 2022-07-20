"""
Real case2

Apply "SLHD" + "MOQQGP/MTQQGP" + "OEHVI/PEHVI" to real metrics data

Objective function: 1 - MacroF1, 1 - MicroF1
Factors:
    Qualitative (4-levels): Adam/Adamax, Weighted/Unweighted
    Quantitative: 
        (1) Learning rate: start=0.0005, end=0.0100, step=0.0005
        (2) Decay rate(lower): start=0.80, end=0.90, step=0.01

Version: 2022-06-03
Author: Quan-En, Li

""" 

# Add system path
import sys
sys.path.append("..")
import os
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from pygmo import hypervolume

from utils import utils, EHVI, experimental_procedure
from model import MOQQGP, MTQQGP, QQLMC
import argparse


def main():

    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--RandomSeed", "-rs", type=int, nargs='?', default=1, help="Random seed")
    parser.add_argument("--SampleSize", "-ss", type=int, nargs='?', default=4, help="Sample size")
    parser.add_argument("--ModelName", "-mn", type=str, nargs='?', help="Model name")
    parser.add_argument("--NoiseSigma", "-ns", type=float, nargs='?', default=0.05, help="Sigma of noise term")
    parser.add_argument("--PosteriorPateto", "-pp", type=int, nargs='?', default=0, choices=[0,1], help="Use posterior mean be pateto front")
    args = parser.parse_args()
    
    RandomSeed = args.RandomSeed
    SampleSize = args.SampleSize
    ModelName = args.ModelName.lower() # most in ["moqqgp", "mtqqgp", "qqlmc"]
    NoiseSigma = 1 if args.NoiseSigma != 0 else 0
    PosteriorPateto = args.PosteriorPateto
    IsIndependent = 1 if ModelName == "moqqgp" else 0
    ref_point = [1.0, 1.0]

    lrs_list = [num / 10000 for num in range(5, 105, 5)]
    low_betas_list = [num / 100 for num in range(80, 91)]

    lrs_arr = np.array(lrs_list)
    low_betas_arr = np.array(low_betas_list)

    file_name_list = [
        "lr_" + format(lr, "8f").replace(".", "") + "_betas_" + str(low_beta).replace(".", "") + "_0999_metrics.csv"
        for lr in lrs_list 
        for low_beta in low_betas_list
    ]

    lrs_grid, low_betas_grid = np.meshgrid(low_betas_arr, lrs_arr)[::-1]


    # Read data

    ## Adam, weighted
    all_adam_matrics_list = [
        utils.ReadFileGetMetrics("../resnet_metrics_TrainTest/Adam" + "/" + file_name).astype(np.float32)
        for file_name in file_name_list
    ]
    all_adam_matrics_arr = np.row_stack(all_adam_matrics_list)

    ## Adam, unweighted
    all_adam_unweight_matrics_list = [
        utils.ReadFileGetMetrics("../resnet_metrics_TrainTest/Adam_unweighted" + "/" + file_name).astype(np.float32)
        for file_name in file_name_list
    ]
    all_adam_unweight_matrics_arr = np.row_stack(all_adam_unweight_matrics_list)

    ## Adamax, weighted
    all_adamax_matrics_list = [
        utils.ReadFileGetMetrics("../resnet_metrics_TrainTest/Adamax" + "/" + file_name).astype(np.float32)
        for file_name in file_name_list
    ]
    all_adamax_matrics_arr = np.row_stack(all_adamax_matrics_list)

    ## Adamax, unweighted
    all_adamax_unweight_matrics_list = [
        utils.ReadFileGetMetrics("../resnet_metrics_TrainTest/Adamax_unweighted" + "/" + file_name).astype(np.float32)
        for file_name in file_name_list
    ]
    all_adamax_unweight_matrics_arr = np.row_stack(all_adamax_unweight_matrics_list)

    # Concatenate
    all_matrics_arr = np.row_stack([
        all_adam_matrics_arr, all_adam_unweight_matrics_arr,
        all_adamax_matrics_arr, all_adamax_unweight_matrics_arr,
    ]).astype(np.float32)

    y1 = all_matrics_arr[:,0]
    y2 = all_matrics_arr[:,1]

    all_w = np.row_stack([
        np.column_stack([[1] * lrs_grid.size, lrs_grid.reshape(-1), low_betas_grid.reshape(-1)]),
        np.column_stack([[2] * lrs_grid.size, lrs_grid.reshape(-1), low_betas_grid.reshape(-1)]),
        np.column_stack([[3] * lrs_grid.size, lrs_grid.reshape(-1), low_betas_grid.reshape(-1)]),
        np.column_stack([[4] * lrs_grid.size, lrs_grid.reshape(-1), low_betas_grid.reshape(-1)]),
    ]).astype(np.float32)

    # Get true pareto front
    true_pareto_set = utils.collect_efficient_solutions([
        *all_adam_matrics_list, *all_adam_unweight_matrics_list,
        *all_adamax_matrics_list, *all_adamax_unweight_matrics_list,
    ])

    hv = hypervolume(true_pareto_set)
    true_pareto_contribute = hv.compute(ref_point)

    # Data generate
    train_w, train_y_list = utils.real_data_generate_with_slhd("../SLHD_initial_data/z4x2/size" + str(SampleSize) + "/" + str(RandomSeed)+".txt")
    train_w = train_w.astype(np.float32)
    train_y_list = [y.astype(np.float32) for y in train_y_list]
    # Get pareto set
    train_y_array = np.column_stack(train_y_list)
    init_canditate_set = [arr for arr in train_y_array]
    init_pareto_set = utils.collect_efficient_solutions(init_canditate_set)

    # Define EI-criterion
    eic = EHVI.Criterion(pareto_set=init_pareto_set, boundary_point=np.array(ref_point))


    model_kwargs = {
        "input":torch.from_numpy(train_w).to(device),
        "levels_num":4,
        "output_list":[torch.from_numpy(y).to(device) for y in train_y_list],
        "noise_term":NoiseSigma,
    }
    # Define model
    my_model = eval(ModelName.upper() + ".Model(**model_kwargs).to(device)")
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.01)

    experimental_procedure.ModelTraining(my_model, optimizer, 200, False)

    procedure_kwargs = {
        "levels_num":4,
        "noise_term":True if NoiseSigma > 0 else False,
        "all_input":all_w,
        "y1":y1,
        "y2":y2,
        "true_pareto_contribute":true_pareto_contribute,
        "start_num":0,
        "add_num":30,
        "is_indep":IsIndependent,
        "ref_point":ref_point,
        "post_pareto":PosteriorPateto,
    }

    # Experiment process
    max_ei_value, ei_time_cost, ei_cr, model_parm = experimental_procedure.AddNewPoints(my_model, eic, **procedure_kwargs)

    # Save result
    output_object = {
        "seed":RandomSeed, 
        "ei_value":max_ei_value, 
        "time_cost":ei_time_cost, 
        "cr":ei_cr, 
        "parm":model_parm,
        "input":my_model.input.cpu().numpy(),
        "output_list":[tensor.cpu().numpy() for tensor in my_model.output_list],
    }

    save_folder_name = "../experiment_result/real/c2_" + ModelName + "_ss" + str(SampleSize) + "_ns" + str(NoiseSigma).replace(".", "-")
    if PosteriorPateto: save_folder_name = save_folder_name + "_pp"
    if not os.path.exists(save_folder_name): os.makedirs(save_folder_name)
    save_file_name = save_folder_name + "/" + str(RandomSeed) + ".pkl"

    utils.save_object(output_object, save_file_name)

if __name__ == "__main__":
    main()