# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:37:21 2018

@author: tk
"""


import numpy as np
import sys
import scipy as sp
from scipy import sparse, misc
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt





from libWavelet import wavedec, wrcoef

    



def baseline_rolling_ball(X, w):
    """Baseline of spectra by rolling ball [1]

    Parameters
    ----------
    X : array3d
        spectra matrix

    w : int
        parameter for smoothness

    Returns
    -------
    array3d
        baseline.

    [1] Kneen et. al., "Algorithm for fitting XRF, SEM and PIXE X-ray spectra backgrounds", Nuclear Instruments and Methods in Physics Research, 1996
    """
    x, y, z = X.shape
    def _f(op, a):
        L = [op(a[:, :, max(0, i-w):i+w], axis=2).reshape(x, y, 1) for i in range(z)]
        return np.concatenate(L, axis=2)
    return _f(np.mean, _f(np.max, _f(np.min, X)))






def baseline_recursive_polynomial_chebyshev(X, wavenumber, degree, max_iter=1000, tol=0.05, eps=1e-16):
    """Baseline of spectra by recursive polynomial fitting
        
    Nick implementation of chebichev recursive polynomial fitting
    
    """
    
    
    def cheb_polyfit(cheb,y,tol,itr):
        #solves for the polynomial coefficients and recurses until tolerance is reached, or until specified iterations are reached         
        yi = np.copy(y)
        for i in range(max_iter):
            
            b = np.linalg.lstsq(cheb,yi,rcond=None)[0]   #b = cheb\yi(:) in matlab
            f = np.matmul(cheb,b)
            
            yi = np.minimum(y, f)
            pc = np.count_nonzero(y < f)/len(y)

            if pc <= tol:
                break  
        
        """
        plt.plot(y[100:])
        plt.plot(yi[100:])
        plt.plot(y[100:]-yi[100:])
        plt.show() 
        """
        
        return yi
    
    
    
    def chebyshev_polygen(n,x):
        #additional function for calculating Chebyshev polys
        m = len(x)
        A = np.zeros((m,n+1))
        A[:,0] = np.ones((m,1)).reshape((-1))
        if n > 1:
            A[:,1] = x.reshape((-1))
            if n > 2:
                for k in range(2,n+1):
                    A[:,k] = 2*x*A[:,k-1] - A[:,k-2]
        return A
         
    
    nx = X.shape[0]
    ny = X.shape[1]
    nw = X.shape[2]
    
    x = np.linspace(-1,1,nw)
    cheb = chebyshev_polygen(degree,x)
    
    bsp = np.zeros((nx,ny,nw));
    for xi in range(nx):
        for yi in range(ny):
            bsp[xi,yi,:] = cheb_polyfit(cheb,X[xi,yi,:],tol,max_iter)
            
    return bsp    


    


def baseline_recursive_polynomial(X, wavenumber, degree, max_iter=1000, tol=0.05, eps=1e-16):
    """Baseline of spectra by recursive polynomial fitting [1]

    Parameters
    ----------
    X : array3d
        spectra matrix

    wavenumber : array
        wavenumbers

    degree : int
        degree of polynomial

    max_iter : int [optional]
        max iteration of recursive fitting

    torelance : float [optional]
        torelance

    Returns
    -------
    array3d
        baseline.
        
    [1] Lieber, Chad A., and Anita Mahadevan-Jansen. "Automated method for subtraction of fluorescence from biological Raman spectra." Applied spectroscopy 57.11 (2003): 1363-1367.
    """
    def _baseline(ys):
        p_prev = np.repeat(0.0, degree+1)
        ys0 = np.copy(ys)
        for i in range(max_iter):
            #print(wavenumber.shape, ys0.shape)
            p = np.polyfit(wavenumber, ys0, degree)
            ys1 = np.polyval(p, wavenumber)
            ys0 = np.minimum(ys, ys1)
            if np.count_nonzero(ys < ys1) < tol * len(ys):
                break
            if np.linalg.norm(p_prev - p) < eps:
                break
            p_prev = p
        return ys0
    res = np.apply_along_axis(_baseline, X.ndim-1, X)
    return res



def baseline_asymetric_least_square(X, lam=2e4, p=0.05, niter=10):
    """Baseline of spectra by Asymmetric Least Squares Smoothing [1]

    Parameters
    ----------
    X : array3d
        hyperspectral image matrix

    lam : float [optional]
        smoothness parameter

    p : float [optional]
        asymetry parameter

    niter : int [optional]
         number of iterations

    Returns
    -------
    array3d
        baseline.
        
    [1] Eilers, Paul HC, and Hans FM Boelens. "Baseline correction with asymmetric least squares smoothing." Leiden University Medical Centre Report 1.1 (2005): 5.
    
    There are two parameters: p for asymmetry and λ for smoothness. Both have to be tuned to the data at hand.
    We found that generally 0.001 ≤ p ≤ 0.1 is a good choice (for a signal with positive peaks) and 10^2 ≤ λ ≤ 10^9,
    but exceptions may occur. In any case one should vary λ on a grid that is approximately linear for log λ. 
    Often visual inspection is sufficient to get good parameter values. 
    """
    
    def baseline_als(y, lam, p, niter):
        L = len(y)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
            
        """    
        plt.plot(y[100:])
        plt.plot(z[100:])
        plt.plot(y[100:]-z[100:])
        plt.show()  
        """
        
        return z
    


    nx = X.shape[0]
    ny = X.shape[1]
    nw = X.shape[2]
       
    bsp = np.zeros((nx,ny,nw));
    for xi in range(nx):
        for yi in range(ny):
            bsp[xi,yi,:] = baseline_als(X[xi,yi,:],lam,p,niter)
            
    return bsp    




def offset_correction(X):
    
    nx = X.shape[0]
    ny = X.shape[1]
    nw = X.shape[2]
    
    bsp = np.zeros((nx,ny,nw));
    for xi in range(nx):
        for yi in range(ny):
            bsp[xi,yi,:] = np.min(X[xi,yi,:])
    
    return bsp



def cosmicray_detect(X, coef=8, verbose=0):
    w, h = (X.shape[0], X.shape[1])
    ray_pos = np.tile(False, (w, h))
    for wn in range(X.shape[-1]):
        std = np.std(X[:,:,wn])
        m = np.mean(X[:,:,wn])
        idx = X[:,:,wn] > m + coef * std
        ray_pos[idx] = True
    return ray_pos


def normalize(X):
    nX = np.copy(X)
    for xi in range(nX.shape[0]):
        for yi in range(nX.shape[1]):
            sumW = sum(np.abs(nX[xi,yi,:]))
            if sumW != 0:
                nX[xi,yi,:] /= sumW/nX.shape[2]
                
    return nX


