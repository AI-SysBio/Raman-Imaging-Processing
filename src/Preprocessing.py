# -*- coding: utf-8 -*-
"""
    Raw Raman hyperspectral image files contain:
        - origimagedata: Raw data ndarray of shape of (Width, Wavenumbers, Height)
        - wavenumber: 1-d ndarray corresponding to the 2nd axis of data
        
    process the hyperspectral Raman image with:
        - Cosmic ray detection
        - Denoising by singular value decomposition
        - Baseline correction
        - Normalization
        
        
    This function return a processed yperspectral Raman image with the dictionary:
        
     rawdata: Raw data ndarray of shape of (Width, Height, Wavenumbers)
        data: processed data ndarray of shape of (Width, Height, Wavenumbers)
    normdata: normalized processed data*1e3, ndarray of shape of (Width, Height, Wavenumbers)
  wavenumber: 1-d ndarray corresponding to the 3rd axis of data
      ignore: boolean ndarray of shape of (Width, Height)
              If the value is True, the spectrum at that position has to 
              be ignored since the intensity of that spectrum is extremely strong
              at some wavenumbers (due to cosmic ray).
        
"""


import scipy.io
import numpy as np
from libRaman import *
import matplotlib.pyplot as plt


def preprocess(fpath, fname):
               
    # ------------------------------------------------------------------------------- #
    # -------------------------- Preprocessing parameters --------------------------- #    
    # ------------------------------------------------------------------------------- #
    
    rpf_degree=7
    baseline_correction = "als"
    denoising = "svd"
    svd_components=10
    cut_side = False #side of Raman image sometimes has problem
    
    # ------------------------------------------------------------------------------- # 
    
    
    
    
    data = scipy.io.loadmat(fpath)    
    X_init = data["origimagedata"]
    if X_init.shape[2]:
        if X_init.shape[2] < max(X_init.shape[0],X_init.shape[1]):
            X_init = np.transpose(data["origimagedata"], (2, 0, 1))
    wn =  data["wavenumber"].reshape(-1)
    if cut_side:
        X_init = X_init[:,2:-2,:] #cute the sides
    X = np.copy(X_init)
    
    #plot the average spectra
    mean = np.mean(X, axis=2)
    plt.imshow(mean, origin='lower')
    plt.title("Average intensity over all wavenumbers")
    #plt.savefig("Processed_Measurements/mask/%s_1AverageI.png" % (fname[:-4]))
    plt.show()
    
    

    if denoising in {"svd"}:
        print("   Starting SVD...")
        u, s, v = np.linalg.svd(X.reshape(-1, X.shape[-1]), full_matrices=False)
        s[svd_components:] = 0
        X = np.matmul(np.matmul(u, np.diag(s)), v).reshape(X.shape)
         
    print("   Detecting Cosmic rays...")
    ignore = cosmicray_detect(X)       
        

    X2 = np.copy(X)
    if baseline_correction in {"of", "offset"}:
        print("   Starting Offset correction...")
        B = offset_correction(X2)
        X2 -= B
    elif baseline_correction in {"rb", "rolling_ball"}:
        print("   Starting Rolling ball...")
        B = baseline_rolling_ball(X2, 20)
        X2 -= B
    elif baseline_correction in {"rpf", "recursive_polynomial_fitting"}:
        print("   Starting Recursive Polynomial Fitting...")
        B = baseline_recursive_polynomial(X2, data["w"], rpf_degree)
        X2 -= B
    elif baseline_correction in {"rpf_cheb", "recursive_polynomial_fitting_chebyshev"}:
        print("   Starting Recursive Chebyshev Polynomial Fitting...")
        B = baseline_recursive_polynomial_chebyshev(X2, data["w"], rpf_degree)
        X2 -= B
    elif baseline_correction in {"als","asymetric_leat_square"}:
        print("   Starting Baseline Removal by Asymetric Least Square Smoothing...")
        B = baseline_asymetric_least_square(X2)
        X2 -= B
        
    #normalization:
    print("   Starting Normalization...")
    nX = normalize(X2)    
        
        

    return { "rawdata" : X_init, "data": X2, "normdata": nX, "wavenumber": wn, "ignore": ignore}
    

