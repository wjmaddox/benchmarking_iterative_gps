import torch
import time
import gpytorch

from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_torch, fit_gpytorch_scipy
from botorch import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import (
    deterministic_probes, 
    cg_tolerance,
    max_preconditioner_size,
    max_cholesky_size,
    cholesky_jitter,
)

import sys
sys.path.append("/home/wesley_m/PyTorch-LBFGS/functions")
from LBFGS import FullBatchLBFGS

def _wang_like_lbfgs(mll):
    full_X = mll.model.train_inputs[0]
    full_Y = mll.model.train_targets
    
    # shuffled so this should be okay
    mll.model.set_train_data(
        full_X[:10000],
        full_Y[:10000],
        strict=False
    )
    
    optimizer = FullBatchLBFGS(mll.model.parameters())


    # with deterministic_probes(True):
        # fit_gpytorch_scipy(mll, options={"maxiter": 10})
    def closure():
        optimizer.zero_grad()
        output = mll.model(*mll.model.train_inputs)
        with cholesky_jitter(1e-3):
            loss = -mll(output, mll.model.train_targets)
        return loss

    loss = closure()
    loss.backward()

    for i in range(10):
        # perform step and update curvature
        options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
    
    # now reset
    mll.model.set_train_data(full_X, full_Y, strict=False)
    
    with deterministic_probes(False):
        fit_gpytorch_torch(mll, options={"maxiter": 2000, "lr": 0.1})
    
class DefaultGPyTorchModel(object):
    def __init__(self, device, dtype, with_adam=True, seed=0, **kwargs):
        self.model = None
        self.with_adam = with_adam
        if with_adam:
            self.optimizer = lambda mll: fit_gpytorch_torch(
                mll, options={"maxiter": 2000}
            )
        else:
            self.optimizer = _wang_like_lbfgs
        
        # if with_adam else fit_gpytorch_model
        self.device = device
        self.dtype = dtype

        torch.random.manual_seed(seed)
        
        self.kernel_init = gpytorch.kernels.keops.MaternKernel

    def fit(self, X, Y):
        self.model = SingleTaskGP(
            torch.tensor(X, device=self.device, dtype=self.dtype),
            torch.tensor(Y, device=self.device, dtype=self.dtype).view(-1,1),
            covar_module = gpytorch.kernels.ScaleKernel(
                self.kernel_init(
                    nu=2.5,
                    ard_num_dims=X.shape[-1],
                    lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                ),
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
            )
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        start = time.time()
        self.optimizer(mll)
        end = time.time()
        print("Model fitting time: ", end - start)
        self.fitting_time = end - start

    def predict(self, Xs):
        # nuke caches so that we can use the same model for different root decomp sizes
        self.model.train()

        # now set in posterior mode
        self.model.eval()
        self.model.likelihood.eval()
        start = time.time()
        
        posterior = self.model.posterior(torch.tensor(Xs, device=self.device, dtype=self.dtype))
        m = posterior.mean.detach().cpu().numpy()
        v = posterior.variance.detach().cpu().numpy()
        end = time.time()
        print("Model prediction time: ", end - start)
        self.pred_time = end - start
        
        return m, v


class CholeskyGP(DefaultGPyTorchModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_init = gpytorch.kernels.MaternKernel
        
    def fit(self, X, Y):
        with max_cholesky_size(10000000):
            super().fit(X, Y)
    def predict(self, Xs):
        with max_cholesky_size(10000000):
            return super().predict(Xs)

class IterativeGP(DefaultGPyTorchModel):
    def __init__(self, device, dtype, with_adam=True, seed=0, cg_tolerance=1., precond_size=15):
        super().__init__(device=device, dtype=dtype, with_adam=with_adam, seed=seed)
        self.cg_tolerance = cg_tolerance
        self.precond_size = precond_size

    def fit(self, X, Y):
        with deterministic_probes(not self.with_adam), cg_tolerance(self.cg_tolerance), max_preconditioner_size(self.precond_size):
            super().fit(X, Y)

    def predict(self, Xs):
        with cg_tolerance(self.cg_tolerance), max_preconditioner_size(self.precond_size):
            return super().predict(Xs)

