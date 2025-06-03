import importlib

# Backend auto-detection
try:
    import torch
    backend = 'pytorch'
except ImportError:
    try:
        import tensorflow as tf
        backend = 'tensorflow'
    except ImportError:
        raise ImportError("Neither PyTorch nor TensorFlow is installed.")

# Dynamic import of optimisation module
if backend == 'pytorch':
    train = importlib.import_module('.train.train_torch', package=__name__)
    optimise = importlib.import_module('.optimise_hyper.optimise_torch', package=__name__)
elif backend == 'tensorflow':
    train = importlib.import_module('.train.train_tensorflow', package=__name__)
    #optimise_functional = importlib.import_module('.optimise_hyper.optimise_functional_tensorflow', package=__name__)
    #optimise_size_shape = importlib.import_module('.optimise_hyper.optimise_size_shape_tensorflow', package=__name__)

# Expose backend and optimise in gwbonsai namespace
__all__ = ['train','backend'] #optimise_functional', 'optimise_size_shape',