
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
import scipy.io

import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA
import traceback
from statistics import mean
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import pickle

def process_neural_data(file_path):
    """
    Load neural data in dictionary format with 'spks', 'istim', 'frame_start', 'xpos', and 'ypos'
    as keys. Spks are taken by frame_start and z-scored.
    """
    dat = np.load(file_path)
    spks = dat['spks'][:, dat['frame_start']]
    print('load data done!')
    print('spks shape: ', spks.shape)
    # zscore neural activity
    spks = spks - spks.mean(axis=1)[:, np.newaxis]
    spks = spks / (spks**2).mean(axis=1)[:, np.newaxis]**.5
    NN, NT = spks.shape
    nstim = NT
    istim = dat['istim'][:nstim]
    print('istim shape: ', istim.shape)
    xpos = dat['xpos']
    ypos = dat['ypos']
    return spks, istim, xpos, ypos

def load_neural_data(file_path):
    """
    Load neural data without preprocessing.
    """
    dat = np.load(file_path)
    return dat['spks'], dat['istim'], dat['xpos'], dat['ypos']

def load_stim_data(file_path):
    """
    Load stimuli data without preprocessing.
    
    Args:
        file_path: File path to .mat file containing dictionary with 
        'img' key of a preprocessed timepoints x pixels matrix.
    """
    dat = scipy.io.loadmat(file_path)
    return dat['img']

def process_stim_data(file_path, istim, Lyd=None, Lxd=None):
    """
    Load stimuli data with preprocessing (Z-score, istim, frame_start).
    
    Args:
        file_path (str): file path to unprocessed stimuli file, which is .mat
        dictionary containing ['img'] key that contains a Ly x Lx x timepoint matrix.
        istim (array): istim data from spk matrix.
        Lyd, Lxd (int): Y and X dimensions to which to downsample original image.
    """
    stims = scipy.io.loadmat(file_path, squeeze_me=True)
    # images that were shown
    img = stims['img']
    Ly, Lx, _  = img.shape
    wx = 0
    dx = 0
    # downsample images
    nstim = len(istim)
    if Lyd is None:
        Lyd = Ly
    if Lxd is None:
        Lxd = Lx
    print("Downsampling stimuli data from ({}, {}) to ({}, {})".format(Ly, Lx, Lyd, Lxd))
    Z = np.zeros((nstim, Lyd, Lxd))
    for j in range(nstim):
        I = img[:,:,istim[j]].copy()
        I = cv2.resize(I, (Lxd, Lyd))        
        I = np.float64(I)
        Z[j] = I - np.mean(I)    
    Z = np.reshape(Z, (nstim, -1))
    mu = Z.mean(axis=0)
    std = Z.std()
    Z -= mu
    Z /= std  # (NT, Lyd*Lxd)
    print('load stimuli done! shape: ', Z.shape)
    return Z

def reduced_rank_regression(X, Y, rank=None, lam=0):
    """ predict Y from X using regularized reduced rank regression 
        
        *** subtract mean from X and Y before predicting
        *** ADD TO FUNCTION.
        
        if rank is None, returns A and B of full-rank (minus one) prediction
        
        Prediction:
        >>> Y_pred = X @ B @ A.T
        
        Parameters
        ----------

        X : 2D array, input data (n_samples, n_features)
        
        Y : 2D array, data to predict (n_samples, n_predictors)
        
        Returns
        --------

        A : 2D array - prediction matrix 1 (n_predictors, rank)
        
        B : 2D array - prediction matrix 2 (n_features, rank)
        
    """
    min_dim = min(Y.shape[1], min(X.shape[0], X.shape[1])) - 1
    if rank is None:
        rank = min_dim
    else:
        rank = min(min_dim, rank)

    # make covariance matrices
    CXX = (X.T @ X + lam * np.eye(X.shape[1])) / X.shape[0]
    CYX = (Y.T @ X) / X.shape[0]

    # compute inverse square root of matrix
    s, u = eigh(CXX)
    #u = model.components_.T
    #s = model.singular_values_**2
    CXXMH = (u * (s + lam)**-0.5) @ u.T

    # project into prediction space
    M = CYX @ CXXMH
    
    # do svd of prediction projection
    model = PCA(n_components=rank).fit(M)
    c = model.components_.T
    s = model.singular_values_
    A = M @ c
    B = CXXMH @ c
    
    return A, B


def get_rfs(stim, spks, img_shape, lam = 0.01, last_n_stim=0, regression_type="linear", rank=None):
    """
    Calculate and plot receptive fields using ridge regression.
    
    Args:
        stim (matrix): Normalized timepoints by stimuli matrix.
        spks (matrix): Normalized n_neurons by time points matrix.
        img_shape (tuple): (y, x) image shape.
        lam (float, optional): Lambda for ridge regression.
        prev_stim (boolean, def=False): Calculate RF based off previous stimulus.
        rank (for reduced rank regression)
    
    Returns:
        B0 (matrix): (Lyd x Lxd x neurons) matrix of receptive fields.
        Spred (matrix): (neuron x timepoint) matrix of spikes based off the RF calculation.
    """

    
    if last_n_stim != 0:
        stim = stim[:-last_n_stim]
        spks = spks[:,last_n_stim:]

    NN, NT = spks.shape
    Lyd, Lxd = img_shape
    npix = stim.shape[-1]
    
    # Get receptive fields 
    if regression_type == "linear":
        B0 = np.linalg.solve((stim.T @ stim + lam * np.eye(npix)),  \
                         (stim.T @ spks.T))
    elif regression_type == "reduced_rank":
        A, B = reduced_rank_regression(stim, spks.T, rank=rank, lam=lam)
        B0 = B @ A.T
    else:
        print("Please specify valid regression type.") 
        
    B0 = np.reshape(B0, (Lyd, Lxd, -1))
    
    # Smooth each RF
    B0 = gaussian_filter(B0, [.5, .5, 0])
    
    # Get predicted spikes, normalize, sort by variance
    Spred = np.reshape(B0, (-1,NN)).T @ stim.T
    Spred -= spks
    varexp = 1.0 - (Spred**2).mean(axis=-1)
    asort = np.argsort(varexp)[::-1]
    
    # figure = graph_RFs(B0, asort)
    
    return B0, Spred

def graph_RFs(B0, asort):
    """
    Args:
        B0 (matrix of neurons x Lxd x Lyd)
    """
    figure = plt.figure(figsize=(18,10))
    for j in range(64):
        A = B0[:,:,asort[j]]
        vl = np.max(np.abs(A))
        plt.subplot(8,8,j+1)
        plt.imshow(A, cmap='bwr', vmin = -vl, vmax = vl)
    return figure


def save_stimuli(stim, out_file):
    """
    Save stimuli
    
    Args:
        stim (matrix): Matrix of stimuli
        out_file (str): Directory to save stimuli in 
    """
    # out_file = "/Users/soniajoseph/Janelia/data/processed_stim"
    dat = {}
    dat['img'] = stim
    scipy.io.savemat(out_file, stim)
    print("Processed stimuli saved.")

def save_rf(B0, Spred, out_file):
    dat = {}
    dat['B0'] = B0
    dat['Spred'] = Spred
    pickle.dump(dat, open(out_file, 'wb'), protocol=4)
