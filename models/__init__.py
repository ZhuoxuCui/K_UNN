import torch
import numpy as np
import torch.optim as optim


def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        #return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
        #                  betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
        #                  eps=config.optim.eps)
        return optim.Adam(parameters, lr=config.optim.lr)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))