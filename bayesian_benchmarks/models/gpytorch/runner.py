import argparse
import logging

import numpy as np
from scipy.stats import norm
import torch

from bayesian_benchmarks.data import get_regression_data
from bayesian_benchmarks.database_utils import Database

from gpytorch import settings

import sys
sys.path.append("./")
from models import CholeskyGP, IterativeGP

def parse_args():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_cholesky", action="store_true")
    parser.add_argument("--cg_tolerance", default=1.0, type=float)
    parser.add_argument("--precond_size", default=15, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--dtype", default="float", type=str)
    parser.add_argument("--with_adam", action="store_true")
    parser.add_argument("--dataset", default='energy', nargs='?', type=str)
    parser.add_argument("--split", default=0, nargs='?', type=int)
    parser.add_argument("--seed", default=0, nargs='?', type=int)
    parser.add_argument("--database_path", default='', nargs='?', type=str)
    return parser.parse_args()

def prediction_loop(model, data):
    m, v = model.predict(data.X_test)

    res = {}

    l = norm.logpdf(data.Y_test, loc=m, scale=v**0.5)
    res['test_loglik'] = np.average(l)

    lu = norm.logpdf(data.Y_test * data.Y_std, loc=m * data.Y_std, scale=(v**0.5) * data.Y_std)
    res['test_loglik_unnormalized'] = np.average(lu)

    d = data.Y_test - m
    du = d * data.Y_std

    res['test_mae'] = np.average(np.abs(d))
    res['test_mae_unnormalized'] = np.average(np.abs(du))

    res['test_rmse'] = np.average(d**2)**0.5
    res['test_rmse_unnormalized'] = np.average(du**2)**0.5
    return res

def main(dataset, split, seed, device, dtype, with_adam, is_cholesky, cg_tolerance, precond_size, database_path=None):
    device = torch.device("cuda:"+str(device))
    dtype = torch.float if dtype == "float" else torch.double

    model_init=CholeskyGP if is_cholesky else IterativeGP
    model = model_init(
            device = device, dtype = dtype, with_adam = with_adam, cg_tolerance=cg_tolerance, precond_size=precond_size
    )

    data = get_regression_data(dataset, split=split)
    model.fit(data.X_train, data.Y_train)

    res_list = []
    if not is_cholesky:
        # loop through several root decomp sizes
        root_decomp_sizes = [10, 100, 250, 500, 1000, 2000, 5000, 10000]
        for rd in root_decomp_sizes:
            try:
                with settings.max_root_decomposition_size(rd):
                    res = prediction_loop(model, data)
                res.update({"root_decomp": rd, "model": "iterative_gp"})
                res_list.append(res)
            except:
                print("Failed at :", rd)
                continue 
    else:
        with settings.fast_pred_var(False):
            res = prediction_loop(model, data)
            res.update({"root_decomp": 1e10, "model": "cholesky_gp"})
            res_list.append(res)

    return res_list

if __name__ == "__main__":
    from random import choice
    from string import ascii_lowercase
    string = ''.join(choice(ascii_lowercase) for i in range(10))
    logging.basicConfig(filename="logs/"+string+".log", level=logging.DEBUG)

    args = parse_args()
    res_list = main(**vars(args))

    # add all extra arguments
    [res.update(args.__dict__) for res in res_list]

    print("now writing to :", args.database_path)
    
    torch.save(f=args.database_path+"/"+string+".pt", obj=res_list)

    # with Database(args.database_path) as db:
    #    for res in res_list:
    #        db.write("regression", res)
    print("finished writing")

