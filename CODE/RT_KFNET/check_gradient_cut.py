# -*- coding: utf-8 -*-
"""Where the gradient stops in the Original architecture, one cut at a time.

The Original RT-KalmanNet has two gradient cuts. They sit on two different paths
and each one alone is enough to make the network untrainable:

    cut 1, INSIDE the neural module: c_{t-1} -> c_t is broken by the
        .clone().detach() on the output feedback (RT_KalmanNet_nn_original.py);
    cut 2, BETWEEN the module and the filter: c_t -> theta_t is broken by the
        bisection, where c enters only as an if-condition (robust_kalman_original.py,
        fnComputeTheta).

Part 1 measures cut 1 on the neural module alone: d c_T / d phi_t, where phi_t is
the module's INPUT at time t, i.e. the F1..F4 feature vector (not the state x_t --
the module never sees the state). That derivative never crosses the filter, so it
says nothing about cut 2, which is exactly why it isolates cut 1.

Part 2 measures cut 2, first locally (d theta / d c against its true value,
computed twice without autograd) and then end to end (d loss / d weights, the
derivative training actually needs).

Three conditions throughout, all fed the same single trajectory:

    A) Original as it is                        cut 1 closed, cut 2 closed
    B) Original with the detach removed         cut 1 OPEN,   cut 2 closed
    C) Proposed (GRU + implicit function thm)   cut 1 open,   cut 2 open

B is what separates the two cuts: it differs from A by one line, so what changes
between them is caused by that line and what does NOT change is caused by the
other cut.

Run with CWD = CODE/RT_KFNET, like the other scripts.
"""

import torch
import torch.nn as nn

from KNet.RT_KalmanNet_nn_original import RT_KalmanNet_nn as OriginalNN
from KNet.RT_KalmanNet_nn_proposed import RT_KalmanNet_nn as ProposedNN
from RobustKalmanPY.robust_kalman_original import RobustKalman as OriginalRK
from RobustKalmanPY.robust_kalman_proposed import RobustKalman as ProposedRK
from Simulations.Extended_sysmdl import SystemModel

T = 10                      # trajectory length
MODEL_SEED = 0
DATA_SEED = 1
FEAT_MODE = 3               # {F1,F2,F3,F4} -> module input of size 2*m + 2*p = 8
HIDDEN_LAYERS = [20] * 5    # Original DNN body, as in task3.py/task4.py
GRU_HIDDEN = 64
C_OP = 0.5                  # operating point for Part 2a: sigmoid(0), what the
                            # Proposed network outputs at initialization
CONDITIONS = {"A": "A) Original", "B": "B) no detach", "C": "C) Proposed"}

# 2-D linear Gauss-Markov model: the simplest system on which the REKF recursion
# is well posed. Nothing here depends on the model being linear.
A_SYS = torch.tensor([[1.0, 0.1], [0.0, 0.95]])
C_SYS = torch.eye(2)
Q_SYS = 1e-3 * torch.eye(2)
R_SYS = 1e-2 * torch.eye(2)
X0 = torch.tensor([[1.0], [0.5]])


class NoDetachOriginalNN(OriginalNN):
    """Condition B: identical to the Original, except the feedback stays attached."""

    def forward(self, x):
        c_t = super().forward(x)
        self.previous_output = c_t  # instead of c_t.clone().detach()
        return c_t


def make_trajectory():
    """Generates the single trajectory the whole script runs on.

    Returns:
        sys_model, y [p, T] measurements, x [n, T] true states.
    """
    sys_model = SystemModel(lambda x: A_SYS @ x, Q_SYS, lambda x: C_SYS @ x, R_SYS, T, T, 2, 2)
    sys_model.InitSequence(X0, 1e-3 * torch.eye(2))
    torch.manual_seed(DATA_SEED)
    sys_model.GenerateSequence(Q_SYS, R_SYS, T)
    return sys_model, sys_model.y, sys_model.x


def make_filter(condition, sys_model, y):
    """Builds condition A, B or C. A and B get bit-identical network weights."""
    torch.manual_seed(MODEL_SEED)
    if condition == "C":
        return ProposedRK(sys_model, y, use_nn=True, input_feat_mode=FEAT_MODE,
                          gru_hidden_size=GRU_HIDDEN)

    filt = OriginalRK(sys_model, y, use_nn=True, input_feat_mode=FEAT_MODE,
                      hidden_layers=HIDDEN_LAYERS)
    if condition == "B":
        torch.manual_seed(MODEL_SEED)
        filt.nn = NoDetachOriginalNN(filt.nn.fcl.in_features, 10, HIDDEN_LAYERS, 1)
    return filt


def record_operating_point(sys_model, y):
    """Runs the Original filter once and records what its two halves actually receive.

    Both parts below are then evaluated where the filter really operates, instead
    of at arbitrary random inputs.

    Returns:
        features: the T input vectors F1..F4 fed to the network;
        P_preds: the T prediction covariances fed to the theta solver.
    """
    filt = make_filter("A", sys_model, y)

    features, P_preds = [], []
    filt.nn.register_forward_pre_hook(lambda _module, args: features.append(args[0].detach().clone()))

    bisection = filt.fnComputeTheta

    def recording_bisection(P_pred):
        P_preds.append(P_pred.detach().clone())
        return bisection(P_pred)

    filt.fnComputeTheta = recording_bisection
    filt.fnREKF(train=False)
    return features, P_preds


# ----------------------------------------------------------------------------
# Part 1 -- cut 1: the recurrence inside the neural module
# ----------------------------------------------------------------------------

def module_input_grads(condition, features):
    """Feeds the T recorded feature vectors to one module and differentiates c_T.

    The T feature vectors phi_1..phi_T are leaves, so d c_T / d phi_t is the
    sensitivity of the last tolerance to the input t steps back: it is non-zero
    only if the module carries gradient through its own recurrent state.

    Returns:
        [d c_T / d phi_1, ..., d c_T / d phi_T], None where autograd has no path.
    """
    torch.manual_seed(MODEL_SEED)
    phis = [f.clone().requires_grad_(True) for f in features]

    if condition == "C":
        module = ProposedNN(phis[0].numel(), gru_hidden_size=GRU_HIDDEN)
        h_t = module.init_hidden()
        for phi in phis:
            c_t, h_t = module(phi, h_t)
    else:
        module_class = OriginalNN if condition == "A" else NoDetachOriginalNN
        module = module_class(phis[0].numel(), 10, HIDDEN_LAYERS, 1)
        for phi in phis:
            c_t = module(phi)

    c_t.backward()
    return [phi.grad for phi in phis]


def run_part1(features):
    """Prints ||d c_T / d phi_t|| for the three conditions."""
    print("PART 1 -- cut 1: the recurrence INSIDE the neural module")
    print(f"  measured: ||d c_{T} / d phi_t||, with phi_t = the F1..F4 feature vector")

    header = "  ".join(f"dc{T}/dphi{t + 1:<2d}" for t in range(T))
    print(f"{'':>14} | {header}")
    for condition in ("A", "B", "C"):
        grads = module_input_grads(condition, features)
        cells = "  ".join(f"{'None':>11}" if g is None else f"{g.norm():>11.2e}" for g in grads)
        print(f"{CONDITIONS[condition]:>14} | {cells}")


# ----------------------------------------------------------------------------
# Part 2 -- cut 2: the path from the tolerance into the filter
# ----------------------------------------------------------------------------

def gamma(P_pred, theta):
    """gamma(P, theta) = tr[(I - theta P)^-1 - I] + log det(I - theta P).

    Written here independently of both filters, so that the reference values of
    Part 2a do not inherit anything from the implementations under test.
    """
    I = torch.eye(P_pred.shape[0])
    M = I - theta * P_pred
    return torch.trace(torch.linalg.inv(M) - I) + torch.log(torch.det(M))


def dtheta_dc_finite_difference(filt, P_pred, rel_step=1e-2):
    """Ground truth: central difference on the bisection itself, no autograd involved.

    The bisection stops at |gamma - c| < 1e-5, so each theta carries an error of
    about 1e-5 / gamma'; with a step of rel_step * c the slope inherits a relative
    error of about 2e-3 / c, i.e. ~0.4% at c = 0.5. Agreement below 1% is therefore
    all this estimate can assert -- and all that is needed to show the derivative
    is not zero.
    """
    h = rel_step * C_OP
    thetas = []
    for c in (C_OP - h, C_OP + h):
        filt.c = torch.tensor(c)
        thetas.append(float(filt.fnComputeTheta(P_pred)))
    return (thetas[1] - thetas[0]) / (2 * h)


def dtheta_dc_implicit(filt, P_pred):
    """Second reference: 1 / gamma'(P, theta*), the implicit function theorem value.

    Differentiating gamma(P, theta(c)) = c at fixed P gives gamma' * dtheta/dc = 1.
    """
    filt.c = torch.tensor(C_OP)
    theta_star = filt.fnComputeTheta(P_pred).detach().clone().requires_grad_(True)
    gamma_grad = torch.autograd.grad(gamma(P_pred, theta_star), theta_star)[0]
    return float(1.0 / gamma_grad)


def dtheta_dc_autograd(filt, P_pred):
    """What autograd reports for d theta / d c, or None when it sees no path at all."""
    filt.c = torch.tensor(C_OP, requires_grad=True)
    theta = filt.fnComputeTheta(P_pred)
    if not theta.requires_grad:
        return None
    return float(torch.autograd.grad(theta.sum(), filt.c)[0])


def loss_grad(filt, x_true):
    """Runs the full REKF over the trajectory and backprops MSE(Xn, x_true).

    This is the derivative training needs, and the only one that has to cross
    cut 2: the loss sees the tolerance only through theta.

    Returns:
        loss, number of parameter tensors reached by a non-zero gradient,
        number of parameter tensors, sum of |grad| over all of them.
    """
    filt.reset_state(filt.y)
    filt.fnREKF(train=True)
    loss = nn.functional.mse_loss(filt.Xn, x_true)

    filt.nn.zero_grad(set_to_none=True)
    loss.backward()
    params = list(filt.nn.parameters())
    reached = sum(1 for p in params if p.grad is not None and p.grad.any())
    total = sum(p.grad.abs().sum().item() for p in params if p.grad is not None)
    return loss.item(), reached, len(params), total


def run_part2(sys_model, y, x_true, P_pred):
    """Prints the local derivative d theta / d c and then the end-to-end d loss / d w."""
    reference_filter = make_filter("A", sys_model, y)
    fd = dtheta_dc_finite_difference(reference_filter, P_pred)
    ift = dtheta_dc_implicit(reference_filter, P_pred)

    print("\n\nPART 2a -- cut 2: does d theta / d c even exist?")
    print(f"  operating point: c = {C_OP}, P_pred as visited by the filter at the last step.\n")
    print(f"  {'finite differences on the bisection':<38} {fd:>12.6f}   (ground truth, no autograd)")
    print(f"  {'implicit function theorem, 1/gamma_prime':<38} {ift:>12.6f}   "
          f"(reference, disagreement {abs(ift - fd) / abs(fd):.2%})")
    print("\n  So the true derivative exists and is far from zero. What autograd reports:\n")
    for condition in ("A", "B", "C"):
        value = dtheta_dc_autograd(make_filter(condition, sys_model, y), P_pred)
        shown = "None -- theta.requires_grad is False" if value is None else f"{value:.6f}"
        print(f"  {CONDITIONS[condition]:>14} | d theta / d c = {shown}")

    print("\n\nPART 2b -- cut 2, end to end: d loss / d weights over the trajectory")
    print("  loss = MSE(Xn, x_true) after a full fnREKF() run, as in the training pipeline.")
    print(f"{'':>14} | {'loss':>10} | {'params reached':>16} | {'sum |grad|':>12}")
    for condition in ("A", "B", "C"):
        loss, reached, n_params, total = loss_grad(make_filter(condition, sys_model, y), x_true)
        print(f"{CONDITIONS[condition]:>14} | {loss:>10.6f} | {f'{reached} of {n_params}':>16} | {total:>12.3e}")


def main():
    sys_model, y, x_true = make_trajectory()
    features, P_preds = record_operating_point(sys_model, y)

    print(f"One trajectory of a 2-D linear model, T = {T}, "
          f"module input = {features[0].numel()} features (F1..F4)\n")
    run_part1(features)
    run_part2(sys_model, y, x_true, P_preds[-1])


if __name__ == "__main__":
    main()
