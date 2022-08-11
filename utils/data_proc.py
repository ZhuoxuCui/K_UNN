import time
import numpy as np
from numpy.core.fromnumeric import transpose
from scipy.io import loadmat
#from skimage.measure import compare_psnr, compare_ssim
import h5py
import mat73
#from utils_np import IFFT2c

#import scipy.io as sio
#import os
#from os.path import join
#org=np.load('/data0/ziwen/zhuoxucui/data/brain_modl/trn_data_90im_4_subjects.npy')
#result_path = "/data0/ziwen/zhuoxucui/result"
#sio.savemat(join(result_path, 'trn_data_90im_4_subjects.mat'), {'org': org})
def FFT2c(x):
    nb, nc, nx, ny = np.shape(x)

    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x,[0,1,3,2])
    x = np.fft.fft(x, axis=-1)
    x = np.transpose(x,[0,1,3,2])
    x = np.fft.fftshift(x, axes=2)/np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.fft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3)/np.math.sqrt(ny)
    return x

def IFFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x,[0,1,3,2])
    x = np.fft.ifft(x, axis=-1)
    x = np.transpose(x,[0,1,3,2])
    x = np.fft.fftshift(x, axes=2)*np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.ifft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3)*np.math.sqrt(ny)
    return x

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def getTestingData_vcc(nImg=64):
    tic()
    A = h5py.File('/data0/yuanyuan/zhuoxu/K_UNN/data/knee_15ch_tst.h5')
    org_real = A['org_real'][:]
    org_img = A['org_img'][:]
    org = org_real + 1j * org_img
    del org_real, org_img

    org = org[nImg:nImg+1, :, :, :]
    A.close()
    org = np.transpose(org, [0, 3, 2, 1])
    org = np.concatenate([org, np.conjugate(np.flip(np.flip(org, 2), 3))], 1)

    B = h5py.File('/data0/yuanyuan/zhuoxu/K_UNN/mask/cartesian_384x384_5x_acs12.mat')
    mask = B['mask'][:]
    mask = np.transpose(mask, [1, 0])
    B.close()
    mask = mask.astype(np.complex64)
    mask = np.tile(mask, [1, 15, 1, 1])
    mask = np.concatenate([mask,(np.flip(np.flip(mask, 2), 3))], 1)

    C = h5py.File('/data0/yuanyuan/zhuoxu/K_UNN/mask/filter_384x384_03.mat')
    h = C['weight'][:]
    h = np.transpose(h, [1, 0])
    C.close()
    h = h.astype(np.complex64)
    h = np.tile(h, [nImg, 1, 1])
    h = np.expand_dims(h,1)
    
    toc()
    print('Undersampling')
    tic()
    orgk, atb, minv = generateUndersampled(org, mask)
    
    atb = c2r(atb)
    orgk = c2r(orgk)
    toc()
    print('Data prepared!')
    return orgk, atb, mask,h, minv

def c2r(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    nImg,nCh,nrow,ncol=inp.shape
    out=np.zeros((nImg,nCh*2,nrow,ncol),dtype=dtype)
    out[:,0:nCh,:,:]=np.real(inp)
    out[:,nCh:nCh*2,:,:]=np.imag(inp)
    return out

def usp(x,mask,nch,nrow,ncol):
    """ This is a the A operator as defined in the paper"""
    kspace=np.reshape(x,(nch,nrow,ncol))
    res=kspace[mask!=0]
    return kspace,res

def usph(kspaceUnder,mask,nch,nrow,ncol):
    """ This is a the A^T operator as defined in the paper"""
    temp=np.zeros((nch,nrow,ncol),dtype=np.complex64)
    temp[mask!=0]=kspaceUnder
    minv=np.std(temp)
    temp=temp/minv
    return temp,minv

def generateUndersampled(org,mask):
    nSlice,nch,nrow,ncol=org.shape
    orgk=np.empty(org.shape,dtype=np.complex64)
    atb=np.empty(org.shape,dtype=np.complex64)
    minv=np.zeros((nSlice,),dtype=np.complex64)
    for i in range(nSlice):
        A  = lambda z: usp(z,mask[i],nch,nrow,ncol)
        At = lambda z: usph(z,mask[i],nch,nrow,ncol)
        orgk[i],y=A(org[i])
        atb[i],minv[i]=At(y)
        orgk[i]=orgk[i]/minv[i]
    del org
    return orgk,atb,minv


