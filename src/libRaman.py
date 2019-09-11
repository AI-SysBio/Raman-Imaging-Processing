
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os



def Identify_Background(X, wn, ignore, fdisp, compute_new_background):
    
    if not os.path.exists("Background_masks"):
            os.makedirs("Background_masks")
    
     
    #keep only the high wavenumber region for the background estimation
    wb1 = np.argmin(abs(wn-2800))
    wb2 = np.argmin(abs(wn-3040))
    wavenumber_index_0 = np.arange(wb1,wb2)
    Xhigh = np.sum(X[:,:,wavenumber_index_0], axis=2)
        
    ny = X.shape[1]    
        
    #do kmean clustering until error is reached
    #(The distance within each cluster should be the distance produced by Poisson error)
    K=1
    Kmax = 30 
    ratio = 10
    while ratio>1:
            
        K += 1
        kmeans = KMeans(n_clusters=K).fit(Xhigh.reshape(-1,1))
            
        k = np.reshape(kmeans.labels_, (-1, ny))
            
        c = kmeans.cluster_centers_
            
        dc = np.zeros(K)
        di = np.zeros(K)
        p = np.zeros(K)
            
            
        for j in range(K):
            p[j] = np.size(np.where(k==j))/np.size(k)/2
            xk,yk = np.where(k==j)
            imk = Xhigh[xk,yk]
            dc[j] = np.mean(np.sqrt(np.square(imk - c[j])))  #compute poisson error
            di[j] = np.mean(np.sqrt(2*imk))  #compute actual error
    
        dc = np.sum(p*dc)
        di = np.sum(p*di)
        ratio = dc/di
            
        print(K, ratio)
            
        if K >= Kmax: #stop if the number of clusters reaches Kmax
            break
            
       
    #define the lowest cluster as background
    kmin = np.argmin(c)
    mask_back = (k==kmin)
        
        
    xb,yb = np.where(np.logical_and( mask_back==1 , ignore == False))
    Xback = np.mean(X[xb,yb,:], axis = 0)
    
    plt.imshow(Xhigh, origin='lower')
    plt.scatter(yb,xb, s=10, color = "black")
    plt.savefig("Background_masks/%s_img.png" %  (fdisp))
    plt.show()
    
    plt.plot(wn,Xback, color = "black")
    plt.plot(wn,np.mean(X,axis = (0,1)), color = "green")
    plt.xlim([660,3040])
    plt.ylim([35,90])
    plt.savefig("Background_masks/%s_spec.png" %  (fdisp))
    plt.show()


    return mask_back




def Irradiance_Profile_Correction(X, tol = 0.05):
    
    nw = X.shape[2]
    Xscan = np.mean(X, axis=0) #mean of each frame in scan direction
    Xscan_n = np.zeros(Xscan.shape)#rescale to set the maxium to one for each line
    for wi in range(nw):
        Xscan_n[:,wi] = Xscan[:,wi]/np.max(Xscan[:,wi])
    
    #estimate the irradiation profile with recursive gaussian fitting
    IrP = np.zeros(Xscan_n.shape);
    x = np.arange(Xscan_n.shape[0])
    
    for wi in range(nw):
        yi = Xscan_n[:,wi]
        pc=1
        y = np.copy(yi)

        while pc >= tol:
            yfit = gaussfit(x,y)
            if sum(yfit) > 0: #in case the fit failed
                y = np.minimum(yi, yfit)
                pc = np.count_nonzero(yi < yfit)/len(yi) 
            else:
                break
        
        if sum(yfit) > 0:
            IrP[:,wi] = gaussfit(x,y)
        
    nwhere = np.where(np.mean(IrP,axis = 0) > 0) #we dont consider the failed fits
    plt.plot(x,np.mean(Xscan_n, axis = 1))
    plt.plot(x,np.mean(IrP[:,nwhere[0]], axis = 1))
    plt.show()        
        
    for wi in range(nw):   #scale the max to 1 for all the gaussians
        if np.mean(IrP[:,wi],axis = 0) > 0:
            IrP[:,wi] = IrP[:,wi]/np.max(IrP[:,wi])

    Irp_mean = np.mean(IrP[:,nwhere[0]], axis = 1) #average the result over all wavenumbers
    
     
    X_ir = np.zeros(X.shape)
    for yi in range(X.shape[1]):
         X_ir[:,yi,:] = X[:,yi,:]/Irp_mean[yi]
         
    plt.imshow(np.mean(X_ir, axis=2), origin='lower')
    plt.show()
    
    return X_ir



def gaussfit(x,y):
    
    def gaus(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    
    #initial guess for mean and variance
    n = len(x)
    mean = sum(x*y)/n
    sigma = np.sqrt(sum(y*(x-mean)**2)/n)
    
    #fitting the data with a gaussian
    try:
        popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])
        yfit = gaus(x,*popt)

    except RuntimeError:
        print("      Warning: one Curve_fit failed")
        yfit = np.zeros(n)
    
    return yfit
    
    





def cosmicray_detect(X, coef=8, verbose=0):
    w, h = (X.shape[0], X.shape[1])
    ray_pos = np.tile(False, (w, h))
    for wn in range(X.shape[-1]):
        std = np.std(X[:,:,wn])
        m = np.mean(X[:,:,wn])
        idx = X[:,:,wn] > m + coef * std
        ray_pos[idx] = True
    return ray_pos



def baseline_recursive_polynomial_chebyshev(X, wavenumber, degree, max_iter=1000, tol=0.05, eps=1e-16):
    """Baseline of spectra by recursive polynomial fitting
        
    Implementation of chebichev recursive polynomial fitting
    (Faster than standard polyfit)
    
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
    nw = X.shape[1]
    x = np.linspace(-1,1,nw)
    cheb = chebyshev_polygen(degree,x)
    bsp = np.zeros((nx,nw));
    for xi in range(nx):
        bsp[xi,:] = cheb_polyfit(cheb,X[xi,:],tol,max_iter)
            
    return bsp




def offset_correction(X):
    
    nx = X.shape[0]
    nw = X.shape[1]
    
    bsp = np.zeros((nx,nw));
    for xi in range(nx):
        bsp[xi,:] = np.min(X[xi,:])
    
    return bsp 