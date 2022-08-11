import os, sys
import torch
import scipy.io as scio
import numpy as np
from torch.utils.data import DataLoader
from utils.data_proc import getTestingData_vcc,IFFT2c,FFT2c
from utils.utils import *
import h5py

org, atb, mask, filt, minv = getTestingData_vcc(nImg=10)

class FastmriDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset_name, mode):
        super(FastmriDataSet, self).__init__()
        self.dataset_name = dataset_name
        if self.dataset_name == 'fastMRI_knee':
            # store the raw tensors
            self.org = org
            self.atb = atb
            self.mask = mask
            self.filt = filt
            self.minv = minv
        elif self.dataset_name == 'UIH':
            sys.exit("CardiacDataSet: Need to implement UIH")
        else:
            sys.exit("CardiacDataSet: No dataset load")

    def __getitem__(self, index):
        if self.dataset_name == 'fastMRI_knee':
            orgk = self.org[index, :]
            atbk = self.atb[index, :]
            mask = self.mask[index, :]
            filtk = self.filt[index, :]
            #minvk = self.minv[index, :]
            return orgk, atbk, mask, filtk
        elif self.dataset_name == 'UIH':
            sys.exit("CardiacDataSet: Need to implement UIH")

    def __len__(self):
        return self.org.shape[0]


def get_dataset(config, mode):
    print("Dataset name:", config.data.dataset_name)
    if config.data.dataset_name == 'fastMRI_knee':
        dataset = FastmriDataSet(config.data.dataset_name, mode)
    else:
        dataset = CardiacDataSet(config.data.dataset_name, mode)
    
    #if mode == 'training':
    #    data = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, 
    #                        num_workers=config.training.num_workers, pin_memory=True)
    if mode == 'test':
        data = DataLoader(dataset, batch_size=config.testing.batch_size, shuffle=False, 
                            num_workers=config.testing.num_workers, pin_memory=True)
    else:
        sys.exit("===== No dataset loaded ======")

    print(mode, "data loaded")
    
    return data