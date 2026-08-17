"""
This file contains the class Pipeline_REKF, 
which is used to train and test RTKnet.
"""

import torch
import torch.nn as nn
import random
import time
from torch.utils.data import DataLoader, TensorDataset


class Pipeline_REKF:

    # Parameters whose name contains this substring are excluded from weight decay.
    NO_WEIGHT_DECAY_ON = "output_layer"

    def __init__(self, folderName, modelName, allow_tbptt=False):
        super().__init__()
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"
        # Only the Proposed RT-KalmanNet's fnREKF() accepts a bptt_truncation
        # argument; keep this False for the Original model, whose fnREKF() does not.
        self.allow_tbptt = allow_tbptt

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel): # this set the state space model
        self.ssModel = ssModel

    def setModel(self, model): # this set the model of the neural network
        self.model = model

    def setTrainingParams(self, args):
        self.args = args
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.N_steps = args.n_steps     # Number of Training Steps
        self.N_B = args.n_batch         # Number of Samples in Batch
        self.learningRate = args.lr     # Learning Rate
        self.weightDecay = args.wd      # L2 Weight Regularization - Weight Decay
        self.bptt_warmup_frac = args.bptt_warmup_frac   # Warmup for BPTT
        self.bptt_truncation = args.bptt_truncation     # Truncation for BPTT
        self.grad_clip = args.grad_clip                 # max grad norm (None disables)
        self.seed = args.seed                           # reproducibility of minibatch sampling

        # Loss Function
        self.loss_fn = nn.MSELoss(reduction='mean')
        # Optimizer
        decay = [p for nm, p in self.model.nn.named_parameters() if self.NO_WEIGHT_DECAY_ON not in nm]
        no_decay = [p for nm, p in self.model.nn.named_parameters() if self.NO_WEIGHT_DECAY_ON in nm]
        self.optimizer = torch.optim.Adam(
            [
                {'params': decay, 'weight_decay': self.weightDecay},
                {'params': no_decay, 'weight_decay': 0.0},
            ],
            lr=self.learningRate,
        )

    def _best_model_path(self, path_results):
        return path_results + 'best-model_' + self.modelName + '.pt'
    
    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results):
        """Trains the RT-KalmanNet and saves the best cross-validation checkpoint.
        SysModel: Maintained for compatibility, but not used in this function.
        """

        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.N_E = len(train_input)    # length of training dataset
        self.N_CV = len(cv_input)      # length of cross-validation dataset
        assert self.N_B <= self.N_E, "n_batch must be <= number of training sequences"

        # Target convention, identical for LDF.m (.mat) and Simulations/utils.DataGen
        # data: column i of the target is the state measured by column i of the input.
        # Both tensors must therefore be pre-aligned to T columns by the caller; the
        # .mat targets carry a trailing, unmeasured x_T column that task3.ipynb drops.
        T = self.model.T
        assert train_input[0].shape[-1] == T and train_target[0].shape[-1] == T, \
            f"train tensors must have T={T} columns, got input " \
            f"{train_input[0].shape[-1]}, target {train_target[0].shape[-1]}"
        assert cv_input[0].shape[-1] == T and cv_target[0].shape[-1] == T, \
            f"cv tensors must have T={T} columns, got input " \
            f"{cv_input[0].shape[-1]}, target {cv_target[0].shape[-1]}"

        best_model_path = self._best_model_path(path_results)

        # Memory to store the loss (linear + dB) at every training step.
        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])
        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        train_loader = DataLoader(TensorDataset(train_input, train_target), batch_size=self.N_B, shuffle=True)

        # Warm-up/fine-tune split: the first `warmup_steps` epochs train with
        # truncated BPTT (self.bptt_truncation), the remaining ones with full BPTT.
        if self.allow_tbptt:
            if self.bptt_warmup_frac is None:
                warmup_steps = self.N_steps
            else:
                assert 0.0 < self.bptt_warmup_frac <= 1.0, "bptt_warmup_frac must be in (0, 1]"
                warmup_steps = round(self.bptt_warmup_frac * self.N_steps)

        for ti in range(self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.model.nn.train()
            epoch_loss = 0.0

            step_bptt_truncation = (self.bptt_truncation if ti < warmup_steps else None) if self.allow_tbptt else None

            for train_input_batch, train_target_batch in train_loader:
                self.optimizer.zero_grad()

                # fnREKF() runs on a single trajectory [p, T] at a time, so the batch
                # dimension is looped over explicitly and the results stacked back together.
                Xn_batch = []
                for b in range(train_input_batch.size(0)):
                    self.model.reset_state(train_input_batch[b])
                    if self.allow_tbptt:
                        self.model.fnREKF(train=True, bptt_truncation=step_bptt_truncation)
                    else:
                        self.model.fnREKF(train=True)
                    Xn_batch.append(self.model.Xn)
                Xn = torch.stack(Xn_batch, dim=0)

                # Filtering loss: the posterior x_{t|t} (model.Xn, [m, T]) against the
                # target column-by-column, one per measurement y_t.
                loss = self.loss_fn(Xn, train_target_batch)

                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.nn.parameters(), self.grad_clip)
                self.optimizer.step()

                epoch_loss += loss.item()

            self.MSE_train_linear_epoch[ti] = epoch_loss / len(train_loader)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################
            self.model.nn.eval()
            with torch.no_grad():
                Xn_cv_batch = []
                for b in range(self.N_CV):
                    self.model.reset_state(cv_input[b])
                    self.model.fnREKF(train=False)
                    Xn_cv_batch.append(self.model.Xn)
                Xn_cv = torch.stack(Xn_cv_batch, dim=0)
                cv_loss = self.loss_fn(Xn_cv, cv_target)

            self.MSE_cv_linear_epoch[ti] = cv_loss.item()
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

            if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model.nn.state_dict(), best_model_path)

            ########################
            ### Training Summary ###
            ########################
            if self.allow_tbptt:
                phase = f"trunc={step_bptt_truncation}" if step_bptt_truncation is not None else "full-BPTT"
                print(f"Step {ti + 1}/{self.N_steps} [{phase}], "
                      f"MSE Training: {self.MSE_train_dB_epoch[ti].item():.4f} [dB], "
                      f"MSE Validation: {self.MSE_cv_dB_epoch[ti].item():.4f} [dB]")
            else:
                print(f"Step {ti + 1}/{self.N_steps}, "
                      f"MSE Training: {self.MSE_train_dB_epoch[ti].item():.4f} [dB], "
                      f"MSE Validation: {self.MSE_cv_dB_epoch[ti].item():.4f} [dB]")

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, SysModel, test_input, test_target, path_results, ckpt_path=None):
        """Evaluates the RT-KalmanNet on the test set.
        SysModel: Maintained for compatibility, but not used in this function.
        ckpt_path: Checkpoint to evaluate; if None, falls back to the best-CV
            checkpoint saved by NNTrain under path_results.
        """
        # Load the best-CV weights (or an explicitly provided checkpoint) into the net.
        if ckpt_path is None:
            ckpt_path = self._best_model_path(path_results)
        self.model.nn.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

        self.N_T = len(test_input)
        m = test_target.size(1)
        T_test = self.model.T

        # Same target convention as NNTrain: tensors pre-aligned to T columns.
        assert test_input[0].shape[-1] == T_test and test_target[0].shape[-1] == T_test, \
            f"test tensors must have T={T_test} columns, got input " \
            f"{test_input[0].shape[-1]}, target {test_target[0].shape[-1]}"

        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        x_out_test = torch.zeros([self.N_T, m, T_test])

        self.model.nn.eval()

        start = time.time()
        with torch.no_grad():
            for k in range(self.N_T):
                self.model.reset_state(test_input[k])
                self.model.fnREKF(train=False)
                Xhat = self.model.Xn                    # [m, T], posterior x_{t|t}
                x_out_test[k] = Xhat
                self.MSE_test_linear_arr[k] = self.loss_fn(Xhat, test_target[k]).item()
        end = time.time()
        comp_time = end - start

        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, comp_time]