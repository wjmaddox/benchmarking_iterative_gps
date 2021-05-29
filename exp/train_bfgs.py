import torch
import botorch  as bo
import gpytorch as gp

from gp_bfgs.train_utils import set_seeds, prepare_dataset


def main(seed=None, device=0, dataset=None, data_dir=None):
  assert dataset is not None

  set_seeds(seed)
  device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"

  data_iter = prepare_dataset(dataset, uci_data_dir=data_dir, device=device)
  _, train_x, train_y = next(data_iter)
  # _, val_x, val_y = next(data_iter)
  # _, test_x, test_y = next(data_iter)

  ## @NOTE: GPytorch #1543.
  train_x, train_y = train_x.contiguous(), train_y.contiguous()
  train_y = train_y.unsqueeze(-1)

  covar_module=gp.kernels.ScaleKernel(
    base_kernel=gp.kernels.MaternKernel(
      nu=1.5, 
      ard_num_dims=train_x.shape[-1],
      lengthscale_prior=gp.priors.torch_priors.GammaPrior(3.0, 6.0)
    ),
    outputscale_prior=gp.priors.torch_priors.GammaPrior(2.0, 0.15),
  )

  model = bo.models.SingleTaskGP(train_x, train_y,
                                 covar_module=covar_module).to(device)
  mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model).to(device)

  bo.fit.fit_gpytorch_model(mll)


if __name__ == '__main__':
  import fire
  fire.Fire(main)