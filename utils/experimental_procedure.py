
import numpy as np

import torch
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from pygmo import hypervolume # python version==3.7.4

from tqdm import tqdm
from timeit import default_timer as time_stamp

from utils import utils, EHVI, NEHVI, SOEI

def unzip_means_vars_covs(all_info: list):
    """
    Only consider two objective
    """

    # y_means_list[i] (batch size, objective size)
    y_means_list = []
    for (mean, var, cov) in all_info:
        half_index = int(len(mean) / 2)
        y_means_list.append(torch.column_stack([mean[:half_index], mean[half_index:]]))

    y_means_list = torch.cat(y_means_list, dim=0)

    # y_covars_list[i] (batch size, objective size, objective size)
    y_covars_list = []
    for (mean, var, cov) in all_info:
        half_index = int(len(mean) / 2)
        y_covars_list.append(torch.column_stack([var[:half_index], cov, cov, var[half_index:]]).reshape(-1, 2, 2))

    y_covars_list = torch.cat(y_covars_list, dim=0)

    return y_means_list, y_covars_list

def unzip_means_vars(all_info: list):
    """
    Only consider single objective
    """
    y_means_list = torch.cat([mean for (mean, var) in all_info])
    y_vars_list = torch.cat([var for (mean, var) in all_info])
    return y_means_list, y_vars_list

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

def AddNewPoints(model, ei_criterion, **kwargs):
    """
    kwargs = {
        "levels_num": int,
        "noise_term": bool,
        "all_input": nd.array,
        "y1": nd.array,
        "y2": nd.array,
        "true_pareto_contribute": float,
        "start_num": int,
        "add_num": int,
        "is_indep": bool,
        "ref_point": list,
        "post_pareto": bool,
    }
    """
    max_ei_value = []
    ei_time_cost = []
    ei_cr = []
    model_parm = []

    for step in tqdm(range(kwargs["start_num"], kwargs["start_num"] + kwargs["add_num"])):

        dataloader = DataLoader(kwargs["all_input"], batch_size=400, shuffle=False)
        # Get model parameters
        model_parm.append(utils.get_model_parameters_dict(model))

        # Collect pareto set
        if kwargs["post_pareto"]:
            post_means = model.PredDistribution(model.input)[0]
            half_len_post_means = int(len(post_means) / 2)
            candidate_y_array = torch.column_stack([
                post_means[:half_len_post_means],
                post_means[half_len_post_means:],
            ]).to("cpu").numpy()
        else:
            candidate_y_array = torch.column_stack(model.output_list).to("cpu").numpy()

        pareto_set = utils.collect_efficient_solutions([arr for arr in candidate_y_array])
        ei_criterion.__init__(pareto_set=pareto_set, boundary_point=np.array(kwargs["ref_point"]))
        
        # Prediction
        means_vars_covs_list = [model.PredDistribution(data.to(device)) for data in dataloader]

        y_means_list, y_covars_list = unzip_means_vars_covs(means_vars_covs_list)
        y_means_array = y_means_list.to("cpu").numpy()
        y_covars_array = y_covars_list.to("cpu").numpy()

        # Calculate EI and add new point
        start_time_stamp = time_stamp()
        argmax_index, ei_values = EHVI.argmax_EHVI(ei_criterion, y_means_array, y_covars_array, is_indep=kwargs["is_indep"])
        end_time_stamp = time_stamp()

        all_candidate_input = np.copy(kwargs["all_input"]).astype(np.float32)
        all_candidate_y1 = np.copy(kwargs["y1"])
        all_candidate_y2 = np.copy(kwargs["y2"])
        model_current_input = np.copy(model.input.to("cpu").numpy().astype(np.float32))

        while np.any(np.all(model_current_input == all_candidate_input[argmax_index], axis=1)):
            all_candidate_input = np.delete(all_candidate_input, obj=argmax_index, axis=0)
            ei_values = np.delete(ei_values, obj=argmax_index)
            all_candidate_y1 = np.delete(all_candidate_y1, obj=argmax_index)
            all_candidate_y2 = np.delete(all_candidate_y2, obj=argmax_index)
            y_means_array = np.delete(y_means_array, obj=argmax_index, axis=0)
            y_covars_array = np.delete(y_covars_array, obj=argmax_index, axis=0)
            argmax_index = np.argmax(ei_values)

        # Save values
        ei_time_cost.append(round(end_time_stamp - start_time_stamp, 4))
        max_ei_value.append(ei_values[argmax_index].item())

        if kwargs["post_pareto"]:
            candidate_y_array = torch.column_stack(model.output_list).to("cpu").numpy()
            pareto_set = utils.collect_efficient_solutions([arr for arr in candidate_y_array])
            ei_criterion.__init__(pareto_set=pareto_set, boundary_point=np.array(kwargs["ref_point"]))
            
        ei_cr.append(round(ei_criterion.ei_contribute() / kwargs["true_pareto_contribute"], 6))

        # Update data
        new_train_w, new_train_y_list = utils.update_data(
            model=model,
            new_w=torch.from_numpy(all_candidate_input[argmax_index, :]).to(device).reshape(1, -1),
            new_y_list=[
                torch.from_numpy(np.array([all_candidate_y1[argmax_index]])).to(device),
                torch.from_numpy(np.array([all_candidate_y2[argmax_index]])).to(device),
            ],
        )

        # Model re-initialization
        model.__init__(input=new_train_w, levels_num=kwargs["levels_num"], output_list=new_train_y_list, noise_term=kwargs["noise_term"])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        ModelTraining(model, optimizer, 200, print_loss=False)

    # Collect pareto set
    candidate_y_array = torch.column_stack(model.output_list).to("cpu").numpy()
    candidate_set = [candidate_y for candidate_y in candidate_y_array]
    pareto_set = utils.collect_efficient_solutions(candidate_set)
    ei_criterion.__init__(pareto_set=pareto_set, boundary_point=np.array(kwargs["ref_point"]))
    ei_cr.append(round(ei_criterion.ei_contribute() / kwargs["true_pareto_contribute"], 6))

    return max_ei_value, ei_time_cost, ei_cr, model_parm