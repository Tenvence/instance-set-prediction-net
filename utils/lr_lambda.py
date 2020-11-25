import numpy as np


def get_warm_up_lr_lambda(warm_up_epochs):
    return lambda epoch: (epoch + 1) / warm_up_epochs


def get_step_lr_lambda(step_size, gamma=0.1):
    return lambda epoch: gamma ** (epoch // step_size + 1)


def get_cosine_lr_lambda(total_epochs):
    return lambda epoch: np.cos(np.pi * epoch / total_epochs) / 2 + .5


def get_warm_up_step_lr_lambda(warm_up_epochs, step_size, gamma=0.1):
    warm_up_lambda = get_warm_up_lr_lambda(warm_up_epochs)
    step_lambda = get_step_lr_lambda(step_size, gamma)
    return lambda epoch: warm_up_lambda(epoch) if epoch < warm_up_epochs else step_lambda(epoch)


def get_warm_up_cosine_lr_lambda(warm_up_epochs, cosine_epochs):
    warm_up_lambda = get_warm_up_lr_lambda(warm_up_epochs)
    cosine_lambda = get_cosine_lr_lambda(cosine_epochs)
    return lambda epoch: warm_up_lambda(epoch) if epoch < warm_up_epochs else cosine_lambda(epoch - warm_up_epochs)
