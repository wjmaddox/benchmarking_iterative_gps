# When are Iterative GPs Numerically Accurate?

This is a code repository for the paper "When are Iterative GPs Numerically Accurate?" by [Wesley Maddox](https://wjmaddox.github.io), [Sanyam Kapoor](https://sanyamkapoor.com), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/). 

### Citation

```
@article{maddox2021iterative,
      title={When are Iterative Gaussian Processes Reliably Accurate?}, 
      author={Wesley J. Maddox and Sanyam Kapoor and Andrew Gordon Wilson},
      year={2021},
      publication={ICML OPTML Workshop},
      url={https://arxiv.org/abs/2112.15246},
}
```

## Models

Our models, both iterative and cholesky-based, are in the models/gpytorch/models.py.

The scripts that can be used to reproduce our results are: 
- `models/gpytorch/runner.py`
- `notebooks/iterative_gps_reliability.ipynb` (explainer)
- `src/train_keops.py` (for optimization trajectories)

## Repository References

The benchmarking library that we cloned is Hugh Salimbeni's bayesian_benchmarks library, available at https://github.com/hughsalimbeni/bayesian_benchmarks.

For full comparison to other libraries that use the same library, we use standardization over both the train/test splits.

For LBFGS, we used (and link to) Michael Shi's LBFGS library, available at https://github.com/hjmshi/PyTorch-LBFGS/.

