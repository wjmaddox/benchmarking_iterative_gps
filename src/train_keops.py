import os
import torch
import gpytorch as gp
from tqdm.auto import tqdm
import wandb
from pathlib import Path
from timeit import default_timer as timer
from lbfgs.functions.LBFGS import FullBatchLBFGS

from utils import set_seeds, prepare_dataset, EarlyStopper


class KeOpsModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, nu=None, min_noise=1e-4):
        assert train_x.is_contiguous(), 'Need contiguous x for KeOps'

        likelihood = gp.likelihoods.GaussianLikelihood(
                      noise_constraint=gp.constraints.GreaterThan(min_noise))
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.base_covar_module = gp.kernels.keops.MaternKernel(ard_num_dims=train_x.size(-1), nu=nu) \
          if nu is not None else gp.kernels.keops.RBFKernel(ard_num_dims=train_x.size(-1))
        self.covar_module = gp.kernels.ScaleKernel(self.base_covar_module)

    def forward(self, x):
        assert x.is_contiguous(), 'Need contiguous x for KeOps'

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


def train(x, y, model, mll, optim):
  model.train()

  def closure():
    optim.zero_grad()
    return -mll(model(x), y)

  t_start = timer()
    
  loss = closure()

  loss_ts = timer() - t_start

  t_start = timer()

  loss.backward()

  if isinstance(optim, FullBatchLBFGS):
    optim.step({ 'closure': closure, 'current_loss': loss })
  else:
    optim.step()

  bw_ts = timer() - t_start

  return {
    'train/mll': -loss.detach().item(),
    'train/loss_ts': loss_ts,
    'train/bw_ts': bw_ts,
    'train/total_ts': loss_ts + bw_ts
  }


def test(x, y, model, label='test'):
  model.eval()

  t_start = timer()

  # pred_y = model.likelihood(model(x))
  pred_y = model(x)
  pred_ts = timer() - t_start

  rmse = (pred_y.mean - y).pow(2).mean(0).sqrt()
  mae = (pred_y.mean - y).abs().mean(0)
  nll = - torch.distributions.Normal(pred_y.mean,
    pred_y.variance.add(model.likelihood.noise).sqrt()).log_prob(y).mean()

  return {
    f'{label}/rmse': rmse.item(),
    f'{label}/mae': mae.item(),
    f'{label}/pred_ts': pred_ts,
    f'{label}/nll': nll
  }


def main(seed=None, device=0, dataset=None, data_dir=None, 
         epochs=20, p_epochs=20, log_int=1,
         nu=None, min_noise=1e-4, lbfgs=False, lr=1.,
         lanc_iter=100, pre_size=100, cg_tol=1e-2):

    set_seeds(seed)
    device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"

    data_iter = prepare_dataset(dataset, uci_data_dir=data_dir, device=device)
    _, train_x, train_y = next(data_iter)
    _, val_x, val_y = next(data_iter)
    _, test_x, test_y = next(data_iter)

    print(f'"{dataset}": D = {train_x.size(-1)}, Train N = {train_x.size(0)}, Val N = {val_x.size(0)} Test N = {test_x.size(0)}')

    wandb.init(config={
      'method': 'KeOps',
      'dataset': dataset,
      'lr': lr,
      'lanc_iter': lanc_iter,
      'pre_size': pre_size,
      'cg_tol': cg_tol,
      'nu': nu,
      'D': train_x.size(-1),
      'N_train': train_x.size(0),
      'N_test': test_x.size(0),
      'N_val': val_x.size(0)
    })
    
    model = KeOpsModel(train_x, train_y, nu=nu, min_noise=min_noise).to(device)
    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    if bool(lbfgs):
      optimizer = FullBatchLBFGS(model.parameters(), lr=lr)
    else:
      optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopper(patience=p_epochs)

    for i in tqdm(range(epochs)):
      with gp.settings.cg_tolerance(cg_tol), \
          gp.settings.max_preconditioner_size(pre_size), \
          gp.settings.max_root_decomposition_size(lanc_iter):
        train_dict = train(train_x, train_y, model, mll, optimizer)

      wandb.log(train_dict, step=i + 1)

      if (i % log_int) != 0:
        continue
      
      with gp.settings.eval_cg_tolerance(cg_tol), \
          gp.settings.max_preconditioner_size(pre_size), \
          gp.settings.max_root_decomposition_size(lanc_iter), \
          gp.settings.fast_pred_var(), torch.no_grad():
        val_dict = test(val_x, val_y, model, label='val')
        test_dict = test(test_x, test_y, model)

        stopper(-val_dict['val/rmse'], dict(
          state_dict=model.state_dict(),
          summary={
            'test/best_rmse': test_dict['test/rmse'],
            'test/best_nll': test_dict['test/nll'],
            'val/best_step': i + 1
          }
        ))
        wandb.log(val_dict, step=i + 1)
        wandb.log(test_dict, step=i + 1)
        wandb.log({
          'param/noise': model.likelihood.noise.item(),
          'param/outputscale': model.covar_module.outputscale.item(),
        }, step=i + 1)
        for d in range(train_x.size(-1)):
          wandb.log({
            f'param/lengthscale/{d}': model.covar_module.base_kernel.lengthscale[0][d].item()
          }, step=i + 1)
        for k, v in stopper.info().get('summary').items():
          wandb.run.summary[k] = v
        torch.save(stopper.info().get('state_dict'), Path(wandb.run.dir) / 'model.pt')
        wandb.save('*.pt')

        if stopper.is_done():
          break

    for k, v in stopper.info().get('summary').items():
      wandb.run.summary[k] = v
    torch.save(stopper.info().get('state_dict'), Path(wandb.run.dir) / 'model.pt')
    wandb.save('*.pt')


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  import pykeops
  import tempfile
  with tempfile.TemporaryDirectory() as dirname:
    pykeops.set_bin_folder(dirname)

    import fire
    fire.Fire(main)
