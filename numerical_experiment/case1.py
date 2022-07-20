"""
Numerical case1: 
    - Moderately correlated (overall)
    - Slightly correlated (conditional)

Apply "SLHD" and "MTQQGP/MOQQGP/QQLMC" to numerical data

Factors:
    Qualitative: 2-levels
    Quantitative: 2-dimensions

Objective functions:
    z=1: Fonseca-Fleming function
    z=2: Combine Fonseca-Fleming function and Sphere function

Correlation:

    - Objective:
        corr(y_1, y_2): 0.6351

    - Qualitative level:
        corr(y(z=1), y(z=2)): 0.7794

Version: 2022-05-20
Author: Quan-En, Li
In thesis
""" 

# Add system path
import os
import sys
sys.path.append("..")

import numpy as np

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from pygmo import hypervolume

from utils import utils, test_functions, EHVI, experimental_procedure
from model import MTQQGP, MOQQGP, QQLMC

import argparse

def main():

    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--GridSize", "-gs", type=int, nargs='?', default=20, help="Grid size")
    parser.add_argument("--RandomSeed", "-rs", type=int, nargs='?', default=1, help="Random seed")
    parser.add_argument("--SampleSize", "-ss", type=int, nargs='?', default=4, help="Sample size")
    parser.add_argument("--ModelName", "-mn", type=str, nargs='?', help="Model name")
    parser.add_argument("--NoiseSigma", "-ns", type=float, nargs='?', default=0.05, help="Sigma of noise term")
    parser.add_argument("--PosteriorPateto", "-pp", type=int, nargs='?', default=0, choices=[0,1], help="Use posterior mean be pateto front")
    args = parser.parse_args()
    
    GridSize = args.GridSize
    RandomSeed = args.RandomSeed
    SampleSize = args.SampleSize
    ModelName = args.ModelName.lower() # most in ["moqqgp", "mtqqgp", "qqlmc"]
    NoiseSigma = args.NoiseSigma
    PosteriorPateto = args.PosteriorPateto
    IsIndependent = 1 if ModelName == "moqqgp" else 0
    ref_point = [1.1, 1.1]
    
    # Factors generate
    x_ = np.linspace(-4, 4, num=GridSize)
    all_x = np.column_stack([x.reshape(-1) for x in np.meshgrid(x_.reshape(-1,1), x_.reshape(-1,1))])

    all_w = np.row_stack([
        np.column_stack([np.array([1] * all_x.shape[0]), all_x]),
        np.column_stack([np.array([2] * all_x.shape[0]), all_x]),
    ]).astype(np.float32)

    # Calculate true surfaces

    ## True surfaces without noise
    pure_z1y1, pure_z1y2 = test_functions.FF_f(all_x)
    pure_z2y1, pure_z2y2 = test_functions.FFS_f(all_x)
    
    ## Add noise
    z1y1 = utils.add_random_noise(arr=pure_z1y1.astype(np.float32), seed=RandomSeed, scale=NoiseSigma)
    z1y2 = utils.add_random_noise(arr=pure_z1y2.astype(np.float32), seed=RandomSeed+1, scale=NoiseSigma)
    z2y1 = utils.add_random_noise(arr=pure_z2y1.astype(np.float32), seed=RandomSeed+2, scale=NoiseSigma)
    z2y2 = utils.add_random_noise(arr=pure_z2y2.astype(np.float32), seed=RandomSeed+3, scale=NoiseSigma)

    ## Clip to [0,1]
    z1y1 = np.clip(z1y1, 0, 1)
    z1y2 = np.clip(z1y2, 0, 1)
    z2y1 = np.clip(z2y1, 0, 1)
    z2y2 = np.clip(z2y2, 0, 1)

    y1 = np.concatenate([z1y1, z2y1])
    y2 = np.concatenate([z1y2, z2y2])

    # Get true pareto contribute
    all_y_pair = np.column_stack([y1, y2])
    full_pareto_front = utils.collect_efficient_solutions([y_pair for y_pair in all_y_pair])
    hv = hypervolume(full_pareto_front)
    true_pareto_contribute = hv.compute(ref_point)

    # TrainData generate

    ## Train input
    train_w_index = utils.numerical_index_generate_with_slhd("../SLHD_initial_data/z2x2/size"+str(SampleSize)+"/"+str(RandomSeed)+".txt", GridSize)
    train_w = np.column_stack([train_w_index[:,0].astype(float), x_[train_w_index[:,1]], x_[train_w_index[:,2]]]).astype(np.float32)
    
    ## Train output
    slice_index = train_w[:,0] == 1
    train_y_list = [
        # y1
        np.concatenate([
            z1y1.reshape(GridSize, GridSize)[train_w_index[slice_index,1], train_w_index[slice_index,2]].reshape(-1),
            z2y1.reshape(GridSize, GridSize)[train_w_index[~slice_index,1], train_w_index[~slice_index,2]].reshape(-1),
        ]),
        # y2
        np.concatenate([
            z1y2.reshape(GridSize, GridSize)[train_w_index[slice_index,1], train_w_index[slice_index,2]].reshape(-1),
            z2y2.reshape(GridSize, GridSize)[train_w_index[~slice_index,1], train_w_index[~slice_index,2]].reshape(-1),
        ]),
    ]
    
    # Get pareto set
    train_y_array = np.column_stack(train_y_list)
    init_canditate_set = np.split(train_y_array, indices_or_sections=train_y_array.shape[0], axis=0)
    init_pareto_set = utils.collect_efficient_solutions(list(map(lambda x: x.reshape(-1), init_canditate_set)))
    
    # Define EI-criterion
    eic = EHVI.Criterion(pareto_set=init_pareto_set, boundary_point=np.array(ref_point))

    # Define model input kwargs
    model_kwargs = {
        "input":torch.from_numpy(train_w.astype(np.float32)).to(device),
        "levels_num":2,
        "output_list":[torch.from_numpy(arr.astype(np.float32)).to(device) for arr in train_y_list],
        "noise_term":True if NoiseSigma > 0 else False,
    }
    
    # Define model
    my_model = eval(ModelName.upper() + ".Model(**model_kwargs).to(device)")
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.01)
    
    # Model training
    experimental_procedure.ModelTraining(my_model, optimizer, 200, print_loss=False)
    
    # Experiment process

    procedure_kwargs = {
        "levels_num":2,
        "noise_term":True if NoiseSigma > 0 else False,
        "all_input":all_w,
        "y1":y1,
        "y2":y2,
        "true_pareto_contribute":true_pareto_contribute,
        "start_num":0,
        "add_num":20,
        "is_indep":IsIndependent,
        "ref_point":ref_point,
        "post_pareto":PosteriorPateto,
    }

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
        "z1y1":z1y1, "z1y2":z1y2, "z2y1":z2y1, "z2y2":z2y2,
    }
    
    save_folder_name = "../experiment_result/numerical/c1_" + ModelName + "_ss" + str(SampleSize) + "_gs" + str(GridSize) + "_ns" + str(NoiseSigma).replace(".", "-")
    if PosteriorPateto: save_folder_name = save_folder_name + "_pp"
    if not os.path.exists(save_folder_name): os.makedirs(save_folder_name)
    save_file_name = save_folder_name + "/" + str(RandomSeed) + ".pkl"

    utils.save_object(output_object, save_file_name)
    
if __name__ == "__main__":
    main()