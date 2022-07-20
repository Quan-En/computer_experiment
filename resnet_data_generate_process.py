
"""
Generation: Matrics data

# DataSet: cassava on TensorFlow

# Referance:
## Mwebaze et al. (2019). iCassava 2019 fine-grained visual categorization challenge. arXiv preprint arXiv:1908.02900

# Classification problem
# Metrics: Macro-F1 score, Micro-F1 score

# Need check data path and makesure data have download already:
## cassava_data_folder_path = r"/home/r26094022/Research/tensorflow_datasets"

Author: Quan-En, Li
Version: 2022-02-24

"""

cassava_data_folder_path = r"/home/r26094022/Research/tensorflow_datasets"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from os import mkdir
from os.path import exists
import csv

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow_datasets as tfds

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn

from torchvision.transforms import RandomRotation
from model.ResNet import resnet18

import argparse

# Image tools
transform_f = lambda img, label: (tf.transpose(tf.image.resize(img, (224, 224)) / 255.0, perm=[2, 0, 1]), label)
ImageRotater = RandomRotation(degrees=(0, 360))

def Model_Training(DataSet, Model, Optimizer, Criterion, Epochs):

    # Set model as training mode
    Model.train()

    # Training...
    for images, labels in DataSet:
        for i in range(Epochs):
            
            # True labels
            tensor_labels = torch.from_numpy(labels.numpy()).to('cuda:0')

            # Set zero gradient
            Optimizer.zero_grad()

            # Calculate loss
            rotated_pred_prob = Model.forward(ImageRotater(torch.from_numpy(images.numpy()).to('cuda:0')))
            rotated_loss = Criterion(rotated_pred_prob, tensor_labels)

            # Print statement
            if (i + 1) % 10 == 0: print("Iter: ", i + 1, ", loss: ", np.round(rotated_loss.item(), 4))

            # Update model parameters
            rotated_loss.backward()
            Optimizer.step()

def Model_Evaluating(DataSet, Model):

    # Set model as evaluating mode
    Model.eval()

    # Prediction without gradiant
    with torch.no_grad():
        
        all_true_label = []
        all_pred_label = []
        
        # Prediction each batch size
        for images, labels in DataSet:
            
            pred_prob = Model.forward(torch.from_numpy(images.numpy()).to('cuda:0'))
            pred_class = torch.argmax(pred_prob, dim=1)
            
            all_true_label.append(torch.from_numpy(labels.numpy()).to('cuda:0'))
            all_pred_label.append(pred_class)
            
        all_true_label = torch.cat(all_true_label).detach().to("cpu").numpy()
        all_pred_label = torch.cat(all_pred_label).detach().to("cpu").numpy()
        
        # Calculation: Macro F1 score and Micro F1 score
        ma_f1 = f1_score(all_true_label, all_pred_label, average="macro")
        mi_f1 = f1_score(all_true_label, all_pred_label, average="micro")
        
        print("Macro F1-score: ", round(ma_f1, 4), "/  Micro F1-score: ", round(mi_f1, 4))
        return ma_f1, mi_f1

def Training_Process(TrainSet, TestSet, Model, Optimizer, Criterion, Epochs, **kwargs):
    print("Training...")
    
    # Folder name: check exist or make one
    folder_name = "resnet_metrics_TrainTest/" + type(Optimizer).__name__
    if not kwargs['weighted']:
        folder_name = folder_name + "_unweighted"
    if not exists(folder_name):
        mkdir(folder_name)

    # File name
    file_name = folder_name + "/" + "lr_" + format(kwargs["lr"], '8f') +  "_"
    
    if type(Optimizer).__name__ in ["Adam", "Adamax", "NAdam"]:
        file_name += "betas_" + str(kwargs["low_beta"]) + "_" + str(kwargs["up_beta"]) + "_"
    elif type(Optimizer).__name__ == "SGD":
        file_name += "momentum_" + str(kwargs["momentum"]) + "_"
    
    file_name = file_name.replace(".", "")
    file_name = file_name + "metrics.csv"
    
    # File header
    header = ["epochs", "data_type", "macro_f1", "micro_f1"]
    
    f = open(file_name, 'w')
    writer = csv.writer(f)
    writer.writerow(header)
    
    # Print optim information
    for key, value in Optimizer.defaults.items():
        print("[", key, "]", ": ", value)
    
    # Training model and save parameters
    for i in range(Epochs):
        print("=====================", "Epoch: ", str(i), "=====================")
        
        ## Training
        Model_Training(TrainSet, Model, Optimizer, Criterion, 1)
        
        ## Evaluating
        print("[ Training set ] ", end="")
        train_ma_f1, train_mi_f1 = Model_Evaluating(TrainSet, Model)
        
        print("[   Test set   ] ", end="")
        test_ma_f1, test_mi_f1 = Model_Evaluating(TestSet, Model)
        
        ## Writing metrics data to csv
        writer.writerows([
            [str(i), "train", str(train_ma_f1), str(train_mi_f1)],
            [str(i), "test", str(test_ma_f1), str(test_mi_f1)],
        ])
    
    f.close()
    
def main():
    
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", "-a", type=str, nargs='?', default="Adam", choices=["Adam", "Adamax", "NAdam", "SGD", "RMSprop"] ,help="Optimizer")
    parser.add_argument("--weighted", "-w", type=bool, nargs='?', default=True, help="Use weighted loss function")
    parser.add_argument("--lr", "-l", type=float, nargs='?', default=0.001, help="Learning rate")
    parser.add_argument("--low_beta", "-lb", type=float, nargs='?', default=0.9, help="Decay rate (lower)")
    parser.add_argument("--up_beta", "-ub", type=float, nargs='?', default=0.999, help="Decay rate (upper)")
    parser.add_argument("--momentum", "-m", type=float, nargs='?', default=0.9, help="Momentum in SGD")
    args = parser.parse_args()
    
    # Data loading
    (train_ds, test_ds), ds_info = tfds.load(
        name='cassava', 
        split=['train+validation', 'test'],
        as_supervised=True, 
        shuffle_files=False,
        with_info=True,
        data_dir=cassava_data_folder_path,
    )
    
    train_ds = train_ds.map(transform_f).batch(64)
    test_ds = test_ds.map(transform_f).batch(64)
    
    # Model
    my_model = resnet18(pretrained=True, num_classes=5).cuda()

    if args.algorithm in ["Adam", "Adamax", "NAdam"]:
        optimizer = eval("torch.optim." + args.algorithm + "(my_model.parameters(), lr=args.lr, betas=(args.low_beta, args.up_beta))")
        msg_str = args.algorithm + ": [" + str(args.lr) + ", " + str(args.low_beta) + ", " + str(args.up_beta) + "]"
    elif args.algorithm == "SGD":
        optimizer = eval("torch.optim." + args.algorithm + "(my_model.parameters(), lr=args.lr, momentum=args.momentum)")
        msg_str = args.algorithm + ": [" + str(args.lr) + ", " + str(args.momentum) + "]"
    elif args.algorithm == "RMSprop":
        optimizer = eval("torch.optim." + args.algorithm + "(my_model.parameters(), lr=args.lr)")
        msg_str = args.algorithm + ": [" + str(args.lr) + "]"
    else:
        print("Error in algorithm names")
        
    if args.weighted:
        weights = [
            (5656+1889)/(466+156), # cbb
            (5656+1889)/(1443+482), # cbsd
            (5656+1889)/(774+258), # cgm
            (5656+1889)/(2658+887), # cmd
            (5656+1889)/(316+106), # healthy
        ]
        class_weights = torch.FloatTensor(weights).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    Training_Process(train_ds, test_ds, my_model, optimizer, criterion, 100, **args.__dict__)
    my_model.to("cpu")
    print(msg_str + " Complete.")

if __name__ == "__main__":
    main()