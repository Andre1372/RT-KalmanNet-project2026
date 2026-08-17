#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import torch
from torch.nn.functional import mse_loss
import Simulations.config as config
import matplotlib.pyplot as plt
import math
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from Simulations.Synthetic_NL_model.parameters import Q_structure, R_structure,m,n,m1x_0,m2x_0, \
    f,h
from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen
from CODE.RT_KFNET.RobustKalmanPY.robust_kalman_original import RobustKalman
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
# KalmanNet
from Pipelines.Pipeline_EKF import Pipeline_EKF
from KNet.KalmanNet_nn import KalmanNetNN

print("Pipeline Start")
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()

### dataset parameters
# N is the number of sequences to be generated
args.N_E = 1 #length of training dataset
args.N_CV = 1 #length of validation dataset
args.N_T = 1 #length of test dataset
args.T = 100 #input sequence length 
args.T_test = 100 #input test sequence length 

### training parameters 
args.use_cuda = False # use GPU or not (True = use GPU)
args.n_steps =  200 #number of training steps (default: 1000)
args.n_batch = 30 #input batch size for training (default: 20)
args.lr = 1e-3 #learning rate (default: 1e-3)
args.wd = 1e-3 #weight decay (default: 1e-4)

path_results = 'KNet/'
DatafolderName = 'Simulations/Synthetic_NL_model/data' + '/'

# noise q and r
r2 = torch.tensor([0.1]) # [100, 10, 1, 0.1, 0.01]
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10) #vdb = 10*log(q2/r2)
q2 = torch.mul(v,r2) #see paper pag. 8 

Q = q2[0] * Q_structure #defining the transition matrix noise
R = r2[0] * R_structure #defining the output matrix noise

print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

traj_resultName = ['traj_synNL_T100.pt']
dataFileName = ['data_synNL_T100.pt'] #used to load data below
    
device = torch.device('cpu')

#########################################
###  Generate and load data DT case   ###
#########################################

sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)
sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

## IMPORTANT! To compute the hard coded jacobian the sys_model also must contain the parameters of f and h
sys_model.alpha = 0.9
sys_model.beta = 1.1
sys_model.phi = 0.1*math.pi
sys_model.delta = 0.01
sys_model.a = 1
sys_model.b = 1
sys_model.c = 0

print("Start Data Gen")
DataGen(args, sys_model, DatafolderName + dataFileName[0])
print("Data Load")
print(dataFileName[0])
[train_input,train_target, _, _, _, _,_,_,_] =  torch.load(DatafolderName + dataFileName[0], map_location=device)
print("trainset input size:",train_input.size())
print("trainset target size:",train_target.size())
#%% Check of dimensions of inputs and outputs - Data Preprocessing

train_input = torch.squeeze(train_input) 
train_target = torch.squeeze(train_target)
print("trainset input size:",train_input.size())
print("trainset target size:",train_target.size())

# %% test of REKF on a test sample

sys_model.m1x_0 = torch.zeros(m,1)
REKF = RobustKalman(sys_model, train_input, 1e-3, True, False)

[Xrekf,_,comp_time_rekf] = REKF.fnREKF()

mse_rekf = mse_loss(Xrekf[:,:Xrekf.size()[1]-1], train_target)
print("\n#####   Test REKF   #####",f"\nMSE: {mse_rekf.item():.4f}",f"\nComputational Time: {comp_time_rekf:.4f}")

#%% Test plot of data

plt.figure()

plt.plot(torch.squeeze(torch.transpose(train_target,0, 1)).numpy()[20:-1, :],color ='red' , label='Test Target')
plt.plot(torch.squeeze(torch.transpose(Xrekf, 0, 1)).numpy()[20:-1, :],color = 'blue' , label = 'Estimated Values')
plt.legend()
plt.title("Test Prediction vs Target State - REKF")

plt.show()

#%% ##################### FROM HERE WE TEST THE IMPLEMENTATION OF RT-KalmanNet #########################

sys_model.m1x_0 = torch.zeros(m,1)
RT_KalmanNet = RobustKalman(sys_model, train_input,1e-3,True,True, input_feat_mode=3)
model = RT_KalmanNet

# Hyper-parameters
epochs = 200    # defining the number of epochs
lr = 1e-3   # learning rate
wd = 1e-3   # weight decay

optimizer = optim.Adam(RT_KalmanNet.nn.parameters(), lr=lr, weight_decay=wd)
criterion = nn.MSELoss(reduction='mean')  # Minimizing the square error wrt the state estimate

opt_MSE = float('inf')
opt_model_folder = "RobustKalmanPY/"

print("\n\n#####   Training RT-KalmanNet   #####\n")
for epoch in range(epochs):

    # Generate data
    DataGen(args, sys_model, DatafolderName + dataFileName[0])
    [train_input, train_target, cv_input, cv_target, _, _, _, _, _] = torch.load(DatafolderName + dataFileName[0], map_location=device)

    # Normalize data
    train_input = torch.squeeze(train_input)
    train_target = torch.squeeze(train_target)

    # Zeroing gradients
    optimizer.zero_grad()

    # Set input
    RT_KalmanNet.y = train_input

    # Forward pass
    [Xrekf,_,_] = RT_KalmanNet.fnREKF(train=True)
    Xrekf = Xrekf[:, 1:]

    # Compute loss
    loss = criterion(Xrekf, train_target)

    # Backward pass
    loss.backward(retain_graph=True)

    # Optimization step
    optimizer.step()

    # Cross-Validation
    cv_input = torch.squeeze(cv_input)
    cv_target = torch.squeeze(cv_target)

    RT_KalmanNet.y = cv_input
    [Xrekf,_,_] = RT_KalmanNet.fnREKF()
    Xrekf = Xrekf[:, 1:]

    cv_loss = criterion(Xrekf, cv_target)

    if (cv_loss < opt_MSE):
        opt_MSE = cv_loss
        torch.save(RT_KalmanNet.nn, 'RobustKalmanPY/opt_RT_KNet.pt')

    if (epoch + 1) % 1 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, MSE Training: {loss.item():.4f}')

print("Training finished")
print(f"Cross-Validation MSE Optimal Model: {opt_MSE.item():.4f}\n")
#%% ##################### TEST OF RT-KalmanNet #########################

DataGen(args, sys_model, DatafolderName + dataFileName[0])
[test_input, test_target, _, _, _, _, _, _, _] = torch.load(DatafolderName + dataFileName[0], map_location=device)

# Preprocess data - removing unecessary dimensions
test_input = torch.squeeze(test_input)
test_target = torch.squeeze(test_target)

# Load the Optimal Model
RT_KalmanNet.nn = torch.load('RobustKalmanPY/opt_RT_KNet.pt')

# Compute the RT-KalmaNet prediction
RT_KalmanNet.y = test_input
[Xrekf,_,comp_time_RT_KNet] = RT_KalmanNet.fnREKF()
Xrekf = Xrekf[:, 1:].detach()

test_loss = criterion(Xrekf, test_target)
print("#####   Test RT-KalmanNet   #####",f"\nMSE: {test_loss.item():.4f}",f"\nComputational Time: {comp_time_RT_KNet:.4f}")


# Plot Prediction vs Target State
plt.figure()    # figsize = (50, 20)
plt.plot(torch.squeeze(torch.transpose(test_target,0, 1)).numpy()[20:-1, :],color ='red' , label='Test Target')
plt.plot(torch.squeeze(torch.transpose(Xrekf, 0, 1)).numpy()[20:-1, :],color = 'blue' , label = 'Estimated Values')
plt.legend()
#plt.xlabel('Sample', fontsize=16)
#plt.ylabel('State', fontsize=16)
plt.title("Test Prediction vs Target State - RT KalmanNet")

plt.show()

#%% ##################### KalmanNet #########################

args.N_E = 1000 #length of training dataset
args.N_CV = 100 #length of validation dataset
args.N_T = 200 #length of test dataset

DataGen(args, sys_model, DatafolderName + dataFileName[0])
print(dataFileName[0])
[train_input,train_target, cv_input, cv_target, test_input, test_target,_,_,_] =  torch.load(DatafolderName + dataFileName[0], map_location=device)

## Build Neural Network
KalmanNet_model = KalmanNetNN()
KalmanNet_model.NNBuild(sys_model, args)

## Train Neural Network
print("\n\n#####   Training KalmanNet   #####\n")
KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KalmanNet_Pipeline.setssModel(sys_model)
KalmanNet_Pipeline.setModel(KalmanNet_model)
print("Number of trainable parameters for KNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
KalmanNet_Pipeline.setTrainingParams(args) 
print("Composition Loss:",args.CompositionLoss)
[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)

## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,knet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)