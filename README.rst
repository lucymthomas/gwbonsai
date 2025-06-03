================
gwbonsai
================

Building and Optimising Neural network Surrogate models for Astrophysical Inference
-----------

Tools to build and optimise gravitational wave surrogate models. 
To help you prune and perfect your surrogates!

## Introduction

This package is designed to aid in the development of gravitational wave surrogate
models which make use of neural network parametric fits. It provides routines for:

1. Optimising the hyperparameters of the network, leveraging Optuna (or hyperopt -- still
under testing). It does this by splitting out the hyperparameters into functional and
size/shape subsets.

2. Optimising the size and distribution of the training dataset used in the network training.

For more details about these routines, please see the paper ([PRD](https://doi.org/10.1103/PhysRevD.111.104029), [arXiv](https://arxiv.org/abs/2501.16462)).

## Citing this package

If you find this package useful for your research, please cite the paper:

```
    @article{Thomas:2025rje,
    author = "Thomas, Lucy M. and Chatziioannou, Katerina and Varma, Vijay and Field, Scott E.",
    title = "{Optimizing neural network surrogate models: Application to black hole merger remnants}",
    eprint = "2501.16462",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "DCC:LIGO-P2400620",
    doi = "10.1103/PhysRevD.111.104029",
    journal = "Phys. Rev. D",
    volume = "111",
    number = "10",
    pages = "104029",
    year = "2025"
    }
```

## Feedback

This package is still under development, so if there are features you would like to see
in future versions, please let us know via the issues tab, or send an email to 
lmthomas@caltech.edu.
