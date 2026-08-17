# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:34:03 2025

@author: ricca
"""
import torch
import math

#%%

torch.pi = torch.acos(torch.zeros(1)).item() * 2

#########################
### Design Parameters ###
#########################
m = 2 #dimension of the state
n = 2 #dimension of the output
#variance = 0

#first order and second order moment of the initial condition
m1x_0 = torch.ones(m, 1)
#m1x_0 = torch.zeros(m, 1)
m2x_0 = 0 * 0 * torch.eye(m)

#defining parameters for the dynamics see paper pag.10
alpha_full = 0.9
beta_full = 1.1
phi_full = 0.1*math.pi
delta_full = 0.01
a = b = 1
c = 0

#################################################################
### State evolution function f for Synthetic Non Linear Model:###
#################################################################
def f(x):
    return alpha_full * torch.sin(beta_full * x + phi_full) + delta_full


#################################################################
### Observation function f for Synthetic Non Linear Model:    ###
#################################################################
def h(x):
    return a * (b * x + c) ** 2

###############################################
### process noise Q and observation noise R ###
###############################################
Q_non_diag = False
R_non_diag = False

Q_structure = torch.eye(m)
R_structure = torch.eye(n)

if(Q_non_diag):
    q_d = 1
    q_nd = 1/2
    Q = torch.tensor([[q_d, q_nd, q_nd],[q_nd, q_d, q_nd],[q_nd, q_nd, q_d]])

if(R_non_diag):
    r_d = 1
    r_nd = 1/2
    R = torch.tensor([[r_d, r_nd, r_nd],[r_nd, r_d, r_nd],[r_nd, r_nd, r_d]])
