import os
import scipy.io as sio
from os.path import join
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from utils.dataset import get_dataset
from models import get_optimizer
from models.net import CovDecoder
from utils.utils import *
import mat73
import scipy.io as scio
import itertools
from scipy.io import loadmat

def get_model(config):
        
    configZ = config.model.modelZ
    configC = config.model.modelC
    configCH = config.model.modelCH
    configP = config.model.modelP

    CNN_theta = CovDecoder(configZ).to(config.device)
    CNN_varphi = CovDecoder(configC).to(config.device)
    CNN_varphi_H = CovDecoder(configCH).to(config.device)
    CNN_psi = CovDecoder(configP).to(config.device)
    return CNN_theta, CNN_varphi, CNN_varphi_H, CNN_psi

criterion = nn.MSELoss()

class Runner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):

        # load dataset
        dataloader = get_dataset(self.config, 'test')
        CNN_theta, CNN_varphi, CNN_varphi_H, CNN_psi = get_model(self.config)
        init_weights(CNN_theta, init_type='normal', init_gain=1e-2)
        init_weights(CNN_varphi, init_type='normal', init_gain=1e-5)
        init_weights(CNN_varphi_H, init_type='normal', init_gain=1e-5)
        init_weights(CNN_psi, init_type='normal', init_gain=1e-5)
        paramst = CNN_theta.parameters()
        paramsv = CNN_varphi.parameters()
        paramsvh = CNN_varphi_H.parameters()
        paramsp = CNN_psi.parameters()
        params = itertools.chain(paramst,paramsv)
        params = itertools.chain(params,paramsvh)
        params = itertools.chain(params,paramsp)
        optimizer = get_optimizer(self.config, params)
        phi_size = self.config.testing.phi_size
        kernel_size = self.config.testing.kernel_size
        eta = torch.randn([1,2,3,3]).to(self.config.device)
        zeta = torch.randn([1,2,3,3]).to(self.config.device)
        
        for index, point in enumerate(dataloader):
            _ ,atb,mask,filt = point
            atb = atb.type(torch.FloatTensor).to(self.config.device)
            mask = mask.to(self.config.device)
            filt = filt.to(self.config.device)

            T = 0
            for epoch in range(self.config.testing.n_epochs):
                t_start = time.time()
                hat_z_H = CNN_theta(eta)
                hat_z_H = c2r(r2c(hat_z_H)/filt)
                z_h = c2r(ifft2c(r2c(hat_z_H)))
                tv = TV(Abs(z_h),'L1')
                hat_exp_phi = CNN_psi(zeta)
                hat_z = c2r(torch.conj(torch.flip(torch.flip(r2c(hat_z_H), [2]), [3])))
                lam = 0  ### lam = 0.2 for vd_regular sampling
                hat_z = lam*hat_z + (1-lam)*FCNN(hat_z_H, hat_exp_phi, phi_size)
                hat_csm = CNN_varphi(zeta)
                hat_csm_H = CNN_varphi_H(zeta)
                hat_x = FCNN(hat_z,hat_csm,kernel_size) 
                hat_x_H = FCNN(hat_z_H,hat_csm_H,kernel_size) 
                hat_x = c2r(torch.cat([r2c(hat_x),r2c(hat_x_H)],1)) 
                loss = criterion(c2r(r2c(hat_x)*mask*filt), c2r(r2c(atb)*filt)) + 0.00007*tv
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t_end = time.time()
                T = T + t_end - t_start
                print('EPOCH %d: LOSS: %.4f' %(epoch, loss.item()))


            result_path = "/data0/yuanyuan/zhuoxu/K_UNN/results"
            out = r2c(hat_x).cpu().data.numpy()
            out = np.transpose(out[0],[1,2,0])
            sio.savemat(join(result_path, 'K_UNN.mat'), {'recon': out})
        