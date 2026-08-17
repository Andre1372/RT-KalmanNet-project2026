#%% Basic libraries for matrix manipulation and math functions

import torch
import time
import torch.nn.functional as func
from KNet.RT_KalmanNet_nn_original import RT_KalmanNet_nn

#%%
# NOTE! There is a combination of numpy and torch thus if changing something use Tensors!
class RobustKalman():
    def __init__(self, SysModel, test_data, c : float = 1e-3, hard_coded: bool = False,use_nn: bool = False, input_feat_mode: int = 0, hidden_layers: list = [50], sl_model: int = 0, set_noise_matrices: bool = False, Q_mat = torch.eye(3), R_mat = torch.eye(3)):
        
        # Select whether to use the NN or regular REKF model
        self.use_nn = use_nn
        
        # Import the model from the SysModel class
        self.model = SysModel 
        self.x0 = torch.transpose(SysModel.m1x_0, 0, 1)
        # Setting the noise covariance matrices
        if set_noise_matrices:
            # In case the sys_model matrices are not used
            self.Q = Q_mat
            self.R = R_mat
        else:
            # In case we are using the same noise matrices which generated the data
            self.Q = SysModel.Q
            self.R = SysModel.R
        self.T = SysModel.T 
        self.y = test_data
        self.c_init = torch.tensor(c)
        self.c = self.c_init.clone()
        self.hard_coded = hard_coded
        self.sl_model = sl_model
        
        # Preallocation of memory for the computation
        self.n = torch.Tensor.numpy(self.Q).shape[0] #state dimension
        self.p = torch.Tensor.numpy(self.R).shape[0] #output dimension
        
        if self.use_nn:
            self.Xrekf = torch.zeros(self.n, self.T+1, requires_grad=True) 
        else:
            self.Xrekf = torch.zeros(self.n, self.T+1)

        """
        with torch.no_grad():
            self.Xrekf[:,0] = self.x0.squeeze(0)
        """
        self.Xrekf_prev = self.x0.squeeze(0)
        self.y_prev = torch.zeros(self.p)
        self.Xn_prev = torch.zeros(self.n)
        self.Xn = torch.zeros(self.n, self.T)
        self.V = torch.zeros(self.n, self.n, self.T+1) 
        
        if self.sl_model == 0:
            self.V_prev = 1e-3*torch.eye(2, 2)
        else:
            self.V_prev = 1e-3*torch.eye(3, 3)
            
        self.A = torch.zeros(self.n, self.n, self.T) 
        self.C = torch.zeros(self.p, self.n, self.T) 
        self.G = torch.zeros(self.n, self.p, self.T)
        self.th = torch.zeros(self.T)


        if self.use_nn:
            self.input_feat_mode = input_feat_mode
            # Select the input feature set
            if self.input_feat_mode == 0:   # Case {F2}
                input_size_fcl = self.p
            elif self.input_feat_mode == 1 or self.input_feat_mode == 2:    # Case {F1,F2,F4}, {F1,F3,F4}
                input_size_fcl = self.p + self.p + self.n
            elif self.input_feat_mode == 3:     # Case {F1,F2,F3,F4}
                input_size_fcl = self.p + self.p + self.n + self.n
            else:
                raise SystemExit("'input_feat_mode' must be an integer value between 0 and 3")

            # Initialize NN
            self.nn = RT_KalmanNet_nn(input_size_fcl,10, hidden_layers,1)

    def reset_state(self, test_data):
        # Re-run the REKF recursion state for a new trajectory (self.nn weights are kept as-is)
        self.y = test_data
        self.c = self.c_init.clone()

        if self.use_nn:
            self.Xrekf = torch.zeros(self.n, self.T+1, requires_grad=True)
            self.nn.previous_output = torch.zeros_like(self.nn.previous_output)
        else:
            self.Xrekf = torch.zeros(self.n, self.T+1)

        self.Xrekf_prev = self.x0.squeeze(0)
        self.y_prev = torch.zeros(self.p)
        self.Xn_prev = torch.zeros(self.n)
        self.Xn = torch.zeros(self.n, self.T)
        self.V = torch.zeros(self.n, self.n, self.T+1)

        if self.sl_model == 0:
            self.V_prev = 1e-3*torch.eye(2, 2)
        else:
            self.V_prev = 1e-3*torch.eye(3, 3)

        self.A = torch.zeros(self.n, self.n, self.T)
        self.C = torch.zeros(self.p, self.n, self.T)
        self.G = torch.zeros(self.n, self.p, self.T)
        self.th = torch.zeros(self.T)

    # Below one can choose to use either the closed form Jacobian or the numerical one from Pytorch
    def fnComputeJacobianF(self, x_n_temp):
        if self.hard_coded:
            f_jac = torch.tensor([[self.model.alpha*self.model.beta*(torch.cos(self.model.phi+ self.model.beta*x_n_temp[0])), 0],[0, self.model.alpha*self.model.beta*(torch.cos(self.model.phi+ self.model.beta*x_n_temp[1]))]])
        else:
            f_jac = torch.autograd.functional.jacobian(self.model.f, x_n_temp)
        return f_jac
    
    def fnComputeJacobianH(self, x_rekf_temp):
        if self.hard_coded:
            h_jac = torch.tensor([[2*self.model.a*self.model.b*(self.model.c + self.model.b*x_rekf_temp[0]), 0],[0, 2*self.model.a*self.model.b*(self.model.c + self.model.b*x_rekf_temp[1])]])
        else:
            h_jac = torch.autograd.functional.jacobian(self.model.h,x_rekf_temp)
            
        return h_jac
    
    def fnComputeTheta(self, P_pred):
        
        value = torch.tensor([1])
        t1 = torch.tensor([0])
        e = torch.linalg.eig(P_pred)[0]
        r = torch.max(torch.abs(e))
        t2 = (1-1e-5)*(torch.pow(r,-1))
        
        while torch.abs(value) >= 1e-5:
            theta = 0.5*(t1+t2)
            value = torch.trace(torch.linalg.solve(torch.eye(self.n) - theta * P_pred,torch.eye(self.n))-torch.eye(self.n)) + torch.log(torch.det(torch.eye(self.n) - theta * P_pred)) - self.c
            if value > 0:
                t2 = theta
            else:
                t1 = theta
        
        return theta

    # Computation of the REKF
    def fnREKF(self, train: bool = False):
        # Setting the NN to training or evaluation
        if self.use_nn:
            if train:
                self.nn.train()
            else:
                self.nn.eval()
                torch.no_grad()

            self.c_array = []

        start = time.time()

        # Forward Step
        for i in range(0, self.T):

            # C_t
            self.C[:, :, i] = self.fnComputeJacobianH(self.Xrekf_prev)

            # L_t

            L = self.V_prev @ torch.transpose(self.C[:, :, i], 0, 1) @ torch.linalg.solve(
                self.C[:, :, i] @ self.V_prev @ torch.transpose(self.C[:, :, i], 0, 1) + self.R, torch.eye(self.p))

            # h(\hat x_t)
            hn = self.model.h(self.Xrekf_prev)

            # \hat x_t|t
            self.Xn[:, i] = self.Xrekf_prev + (L @ (self.y[:, i] - hn))

            # A_t
            self.A[:, :, i] = self.fnComputeJacobianF(self.Xn[:, i])

            # G_t
            self.G[:, :, i] = self.A[:, :, i] @ L

            # \hat x_t+1
            self.Xrekf = self.Xrekf.clone()

            self.Xrekf[:, i + 1] = torch.squeeze(self.model.f(self.Xn[:, i]))

            # P_t+1
            P = self.A[:, :, i] @ self.V_prev @ torch.transpose(self.A[:, :, i], 0, 1) - self.A[:, :,i] @ self.V_prev @ torch.transpose(self.C[:, :, i], 0, 1) @ torch.linalg.solve(self.C[:, :, i] @ self.V_prev @ torch.transpose(self.C[:, :, i], 0, 1) + self.R,torch.eye(self.p)) @ self.C[:, :, i] @ self.V_prev @ torch.transpose(self.A[:, :, i], 0, 1) + self.Q

            if self.use_nn:
                # Compute input features F1,F2,F3,F4
                self.f1 = self.y[:,i] - self.y_prev
                self.f2 = self.y[:, i] - hn
                self.f3 = self.Xn[:,i] - self.Xn_prev
                self.f4 = self.Xn[:, i] - self.Xrekf_prev

                # Stacking input features [F1, F2, F4]
                if self.input_feat_mode == 0:
                    input_features = self.f2
                elif self.input_feat_mode == 1:
                    input_features = torch.cat([self.f1, self.f2, self.f4], dim=0)
                elif self.input_feat_mode == 2:
                    input_features = torch.cat([self.f1, self.f3, self.f4], dim=0)
                else:
                    input_features = torch.cat([self.f1, self.f2, self.f3, self.f4], dim=0)

                # Forward Step
                self.c = self.nn(input_features)

                self.c_array.append(self.c.item())

            # th_t
            self.th[i] = self.fnComputeTheta(P)

            # V_t+1
            self.V[:, :, i + 1] = torch.linalg.solve(
                torch.linalg.solve(P, torch.eye(self.n)) - self.th[i] * torch.eye(self.n), torch.eye(self.n))
            self.V_prev = self.V[:, :, i + 1]

            # Update Xrekf_prev
            self.Xrekf_prev = self.Xrekf[:, i + 1]

            # Update Xn_prev
            self.Xn_prev = self.Xn[:, i]

            # Update y_prev
            self.y_prev = self.y[:,i]

        end = time.time()
        t = end - start

        if self.use_nn:
            return [self.Xrekf, self.c_array, self.V, t]
        else:
            return [self.Xrekf, self.V, t]