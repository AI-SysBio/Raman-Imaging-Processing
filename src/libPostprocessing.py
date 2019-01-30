# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:37:21 2018

@author: tk
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from colorama import Fore
from colorama import Style

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from libWavelet import *
import pywt




def perform_PCA(X,yc,yi,plot):

    pca = PCA()
    X_reduced = pca.fit_transform(X) 
    
    plt.plot(np.arange(1,11),pca.explained_variance_ratio_[0:10]*100, '-o')
    plt.title("PCA ratios")
    plt.ylabel("PCA ratios [%]")
    plt.xlabel("PC")
    plt.show()    
        
    colors = ["blue", "red", "green", "gold", "darkviolet", "orange", "crimson", "cadetblue", "navy", "olive", "darkcyan", "darkorange", "brown", "coral", "chartreuse", "chocolate", "maroon","dodgerblue","goldenrod", "darkred","darkblue"]
    colors_c = []
    for i in range(len(yc)):
        colors_c.append(colors[int(yc[i])])
    
    colors_i = []
    for i in range(len(yi)):
        colors_i.append(colors[int(yi[i])+2])
            
        
           
    plt.scatter(X_reduced[:,0], X_reduced[:,1], color = colors_i, alpha=0.1)   
    plt.title("PCA of Raman wavenumbers by date")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    
    
        
    if plot:        
        plt.scatter(X_reduced[:,0], X_reduced[:,2], color = colors_i, alpha=0.1)    
        plt.title("PCA of Raman wavenumbers by date")
        plt.xlabel("PC1")
        plt.ylabel("PC3")
        plt.show()
        
        
        plt.scatter(X_reduced[:,1], X_reduced[:,2], color = colors_i, alpha=0.1)   
        plt.title("PCA of Raman wavenumbers by date")
        plt.xlabel("PC2")
        plt.ylabel("PC3")
        plt.show()
        
        
        
        
        
    plt.scatter(X_reduced[:,0], X_reduced[:,1], color = colors_c, alpha=0.1)   
    plt.title("PCA of Raman wavenumbers FTC/Nthy")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
        
    if plot:        
        plt.scatter(X_reduced[:,0], X_reduced[:,2], color = colors_c, alpha=0.1)    
        plt.title("PCA of Raman wavenumbers FTC/Nthy")
        plt.xlabel("PC1")
        plt.ylabel("PC3")
        plt.show()
        
        
        plt.scatter(X_reduced[:,1], X_reduced[:,2], color = colors_c, alpha=0.1)   
        plt.title("PCA of Raman wavenumbers FTC/Nthy")
        plt.xlabel("PC2")
        plt.ylabel("PC3")
        plt.show()
    
    return pca
    
   
    

    
    






    
def classify_knn_PCA(X,pca,nPC,n_test,n_train,yc,ycell,yim, weighted, plot):  
    
       
    X = X[:,nPC]
    
    
    #transform the matrix with weighted value taht depends on the PC importance
    X_weighted = np.copy(X)
    Mean_i = np.mean(X, axis=0)
     
    if weighted:
        for PC in range(np.size(nPC)):
            X_weighted[:,PC] = Mean_i[PC] + (X_weighted[:,PC] - Mean_i[PC])*pca.explained_variance_[PC]
        
    #X_weighted = np.copy(X)
        
        
    
    # count the number of FTC and Nthy spectra in the training set
    X_test = X_weighted[n_test,:].reshape(np.size(n_test),np.size(nPC))
    X_train = X_weighted[n_train,:].reshape(np.size(n_train),np.size(nPC))
    yc_train = yc[n_train]
    yc_test = yc[n_test]
    NTHY = np.size(np.where(yc_train == 0))
    NFTC = np.size(np.where(yc_train == 1))
    ratio = NTHY/NFTC
    
    if plot:
        print("  Training set:")
        print("   ", NTHY, "Nthy spectra")
        print("   ", NFTC, "FTC spectra")
        print("    Ratio FTC/Nthy = 1.00:%.2f" % (ratio))
        
        
    #train a kNN classifier for nearest neighboor classification:
    n_neighbors = 10
    kNN = KNeighborsClassifier(n_neighbors=n_neighbors)
    kNN.fit(X_train, yc_train)
    y_pred_train = kNN.predict(X_train)
    
    acc = accuracy_score(yc_train,y_pred_train)
    print("   kNN training accuracy is", acc)
    
    
    #test: classify each spectra of a test cell with kNN
    if plot:
        print("\n  Predictions on Test set:")
    acc = 0
    n_cell = 0
    
    pred_index_ = []
    true_index_ = []   
    
    yim_test = yim[n_test]
    nimg = np.max(yim_test)+1
    ycancer = ["Nthy", "FTC"]
    for img_index in range(0,nimg):
        n_img = np.where(yim_test == img_index)
        if np.size(n_img) > 0:
            ncells = int(np.max(ycell[n_test][n_img]))+1
            for cell_index in range(0,ncells):
                n_sample = np.where(np.logical_and(ycell[n_test] == cell_index, yim_test == img_index))
                if np.size(n_sample) > 0:
                    n_cell += 1
                    yc_cell = yc_test[n_sample]
                    X_cell = X_test[n_sample,:].reshape(np.size(n_sample),np.size(nPC))
                    
                    #change prediction to account for the P(Y)
                    y_nei_nc = kNN.predict_proba(X_cell)[:,0]*n_neighbors
                    y_nei_can = kNN.predict_proba(X_cell)[:,1]*n_neighbors
                    y_pred_proba = y_nei_can*ratio/(y_nei_can*ratio+y_nei_nc)
                    index_proba = np.sum(y_pred_proba)/np.size(y_pred_proba)
            
                    y_pred = y_pred_proba > 0.5
                    index = np.sum(y_pred)/np.size(y_pred)
                    
                    real_index = yc_cell[0]
                    
                    if plot:
                        colors = ["blue", "red"]
                        colors_c = []
                        for i in range(len(yc_train)):
                            colors_c.append(colors[int(yc_train[i])])
                        plt.scatter(X_train[:,0], X_train[:,1], color = colors_c, alpha=0.1)   
                        plt.scatter(X_cell[:,0], X_cell[:,1], color = "black", alpha=1)   
                        plt.title("PCA of Raman wavenumbers Jan/March/June/July")
                        plt.xlabel("PC1")
                        plt.ylabel("PC2")
                        plt.show()
                    
                    
                    
                    if index>0.5:    
                        prediction = 1
                    else:
                        prediction = 0
                        
                    if ycancer[prediction] == ycancer[real_index]:
                        acc += 1
                        
                    if plot:    
                        if ycancer[prediction] == ycancer[real_index]:
                            print(f"\n     {Fore.GREEN}Classifying %s cell [%s,%s]{Style.RESET_ALL}" % (ycancer[real_index],img_index,cell_index))
                            print(f"     {Fore.GREEN}Cancer index is %.3f{Style.RESET_ALL}" % index)
                            print(f"     {Fore.GREEN}Cancer index proba is %.3f{Style.RESET_ALL}" % index_proba)
                            
                        else:
                            print(f"\n     {Fore.RED}Classifying %s cell [%s,%s]{Style.RESET_ALL}" % (ycancer[real_index],img_index,cell_index))
                            print(f"     {Fore.RED}Cancer index is %.3f{Style.RESET_ALL}" % index)
                            print(f"     {Fore.RED}Cancer index proba is %.3f{Style.RESET_ALL}" % index_proba)
                            
                    true_index_.append(real_index)
                    pred_index_.append(index)
                        
    
    acc /= n_cell 
    if plot:     
        print("\n  Prediction accuracy with %s cells is %s/%s = %.3f" % (n_cell, int(round(acc*n_cell,0)), n_cell, acc))
    
    true_index = np.array(true_index_)
    pred_index = np.array(pred_index_)
    
    return acc,n_cell,pred_index,true_index


















def classify_svm_PCA(X,pca,nPC,n_test,n_train,yc,ycell,yim, plot):  
    
       
    X = X[:,nPC]
        
    # count the number of FTC and Nthy spectra in the training set
    X_test = X[n_test,:].reshape(np.size(n_test),np.size(nPC))
    X_train = X[n_train,:].reshape(np.size(n_train),np.size(nPC))
    yc_train = yc[n_train]
    yc_test = yc[n_test]
    NTHY = np.size(np.where(yc_train == 0))
    NFTC = np.size(np.where(yc_train == 1))
    ratio = NTHY/NFTC
    
    if plot:
        print("  Training set:")
        print("   ", NTHY, "Nthy spectra")
        print("   ", NFTC, "FTC spectra")
        print("    Ratio FTC/Nthy = 1.00:%.2f" % (ratio))
        
        
    #train a kNN classifier for nearest neighboor classification:
    class_weight = "balanced"
    
    SVM = svm.SVC(class_weight = class_weight)
    SVM.fit(X_train, yc_train)
    y_pred_train = SVM.predict(X_train)
    
    acc = accuracy_score(yc_train,y_pred_train)
    print("   SVM training accuracy is", acc)
    
    
    #test: classify each spectra of a test cell with kNN
    if plot:
        print("\n  Predictions on Test set:")
    acc = 0
    n_cell = 0
    
    pred_index_ = []
    true_index_ = []   
    
    yim_test = yim[n_test]
    nimg = np.max(yim_test)+1
    ycancer = ["Nthy", "FTC"]
    for img_index in range(0,nimg):
        n_img = np.where(yim_test == img_index)
        if np.size(n_img) > 0:
            ncells = int(np.max(ycell[n_test][n_img]))+1
            for cell_index in range(0,ncells):
                n_sample = np.where(np.logical_and(ycell[n_test] == cell_index, yim_test == img_index))
                if np.size(n_sample) > 0:
                    n_cell += 1
                    yc_cell = yc_test[n_sample]
                    X_cell = X_test[n_sample,:].reshape(np.size(n_sample),np.size(nPC))
                    
                    #change prediction to account for the P(Y)
                    y_pred_proba = SVM.predict(X_cell)
            
                    y_pred = y_pred_proba > 0.5
                    index = np.sum(y_pred)/np.size(y_pred)
                    
                    real_index = yc_cell[0]
                    
                    if plot:
                        colors = ["blue", "red"]
                        colors_c = []
                        for i in range(len(yc_train)):
                            colors_c.append(colors[int(yc_train[i])])
                        plt.scatter(X_train[:,0], X_train[:,1], color = colors_c, alpha=0.1)   
                        plt.scatter(X_cell[:,0], X_cell[:,1], color = "black", alpha=1)   
                        plt.title("PCA of Raman wavenumbers")
                        plt.xlabel("PC1")
                        plt.ylabel("PC2")
                        plt.show()
                    
                    
                    
                    if index>0.5:    
                        prediction = 1
                    else:
                        prediction = 0
                        
                    if ycancer[prediction] == ycancer[real_index]:
                        acc += 1
                        
                    if plot:    
                        if ycancer[prediction] == ycancer[real_index]:
                            print(f"\n     {Fore.GREEN}Classifying %s cell [%s,%s]{Style.RESET_ALL}" % (ycancer[real_index],img_index,cell_index))
                            print(f"     {Fore.GREEN}Cancer index is %.3f{Style.RESET_ALL}" % index)
                            
                        else:
                            print(f"\n     {Fore.RED}Classifying %s cell [%s,%s]{Style.RESET_ALL}" % (ycancer[real_index],img_index,cell_index))
                            print(f"     {Fore.RED}Cancer index is %.3f{Style.RESET_ALL}" % index)
                            
                    true_index_.append(real_index)
                    pred_index_.append(index)
                        
    
    acc /= n_cell 
    if plot:     
        print("\n  Prediction accuracy with %s cells is %s/%s = %.3f" % (n_cell, int(round(acc*n_cell,0)), n_cell, acc))
    
    true_index = np.array(true_index_)
    pred_index = np.array(pred_index_)
    
    return acc,n_cell,pred_index,true_index































def classify_LDA_PCA(X,pca,nPC,n_test,n_train,yc,ycell,yim, plot):  
    
       
    X = X[:,nPC]
    X_weighted = np.copy(X)
        
        
    
    # count the number of FTC and Nthy spectra in the training set
    X_test = X_weighted[n_test,:].reshape(np.size(n_test),np.size(nPC))
    X_train = X_weighted[n_train,:].reshape(np.size(n_train),np.size(nPC))
    yc_train = yc[n_train]
    yc_test = yc[n_test]
    NTHY = np.size(np.where(yc_train == 0))
    NFTC = np.size(np.where(yc_train == 1))
    ratio = NTHY/NFTC
    
    if plot:
        print("  Training set:")
        print("   ", NTHY, "Nthy spectra")
        print("   ", NFTC, "FTC spectra")
        print("    Ratio FTC/Nthy = 1.00:%.2f" % (ratio))
        
        
    #train a kNN classifier for nearest neighboor classification:
    mda = LinearDiscriminantAnalysis()
    X_reduced_train = mda.fit_transform(X_train, yc_train)
    X_reduced_test = mda.transform(X_test)
    
    y_pred_train = X_reduced_train[:,0]>0
    
    acc = accuracy_score(yc_train,y_pred_train)
    print("   LDA training accuracy is", acc)
    
    colors = ["blue", "red"]
    colors_c = []
    for i in range(len(y_pred_train)):
        colors_c.append(colors[int(y_pred_train[i])])
            
        
    
    
    #test: classify each spectra of a test cell with kNN
    if plot:
        print("\n  Predictions on Test set:")
    acc = 0
    n_cell = 0
    
    pred_index_ = []
    true_index_ = []   
    
    yim_test = yim[n_test]
    nimg = np.max(yim_test)+1
    ycancer = ["Nthy", "FTC"]
    for img_index in range(0,nimg):
        n_img = np.where(yim_test == img_index)
        if np.size(n_img) > 0:
            ncells = int(np.max(ycell[n_test][n_img]))+1
            for cell_index in range(0,ncells):
                n_sample = np.where(np.logical_and(ycell[n_test] == cell_index, yim_test == img_index))
                if np.size(n_sample) > 0:
                    n_cell += 1
                    yc_cell = yc_test[n_sample]
                    X_cell = X_reduced_test[n_sample]

                    y_pred = X_cell[:,0]>0
                    index = np.sum(y_pred)/np.size(y_pred)
                    
                    real_index = yc_cell[0]
                    
                    if plot:    
                        plt.figure(figsize=(7,2))
                        plt.scatter(X_reduced_train[:,0],y_pred_train, color = colors_c, alpha=0.1,s=25)
                        plt.scatter(X_cell[:,0],np.full((np.size(y_pred)),0.5), color = "black", alpha=0.1,s=25)
                        plt.ylim([-0.4,1.4])
                        plt.xlim([np.min(X_reduced_train[:,0]),np.max(X_reduced_train[:,0])])
                        plt.title("LDA of Raman spectra")
                        plt.xlabel("PC1")
                        plt.show()
                    
                    
                    
                    if index>0.5:    
                        prediction = 1
                    else:
                        prediction = 0
                        
                    if ycancer[prediction] == ycancer[real_index]:
                        acc += 1
                        
                    if plot:    
                        if ycancer[prediction] == ycancer[real_index]:
                            print(f"\n     {Fore.GREEN}Classifying %s cell [%s,%s]{Style.RESET_ALL}" % (ycancer[real_index],img_index,cell_index))
                            print(f"     {Fore.GREEN}Cancer index is %.3f{Style.RESET_ALL}" % index)
                            
                        else:
                            print(f"\n     {Fore.RED}Classifying %s cell [%s,%s]{Style.RESET_ALL}" % (ycancer[real_index],img_index,cell_index))
                            print(f"     {Fore.RED}Cancer index is %.3f{Style.RESET_ALL}" % index)
                            
                    true_index_.append(real_index)
                    pred_index_.append(index)
                        
    
    acc /= n_cell 
    if plot:     
        print("\n  Prediction accuracy with %s cells is %s/%s = %.3f" % (n_cell, int(round(acc*n_cell,0)), n_cell, acc))
    
    true_index = np.array(true_index_)
    pred_index = np.array(pred_index_)
    
    return acc,n_cell,pred_index,true_index







def classify_wn(X,wFUSE,wn,n_test,n_train,yc,ycell,yim, weighted, plot):  

       
    wkeep = []
    for wi in range(len(wFUSE)):
        wkeep.append(np.argmin(abs(wn-wFUSE[wi])))
        
    X = X[:,np.array(wkeep)]
    
    
    
    # count the number of FTC and Nthy spectra in the training set
    X_test = X[n_test]
    X_train = X[n_train]
    yc_train = yc[n_train]
    yc_test = yc[n_test]
    NTHY = np.size(np.where(yc_train == 0))
    NFTC = np.size(np.where(yc_train == 1))
    ratio = NTHY/NFTC
    
    if plot:
        print("  Training set:")
        print("   ", NTHY, "Nthy spectra")
        print("   ", NFTC, "FTC spectra")
        print("    Ratio FTC/Nthy = 1.00:%.2f" % (ratio))
        
        
    #train a kNN classifier for nearest neighboor classification:
    n_neighbors = 10
    kNN = KNeighborsClassifier(n_neighbors=n_neighbors)
    kNN.fit(X_train, yc_train)
    y_pred_train = kNN.predict(X_train)
    
    acc = accuracy_score(yc_train,y_pred_train)
    print("   kNN training accuracy is", acc)
    
    
    #test: classify each spectra of a test cell with kNN
    if plot:
        print("\n  Predictions on Test set:")
    acc = 0
    n_cell = 0
    
    pred_index_ = []
    true_index_ = []   
    
    yim_test = yim[n_test]
    nimg = np.max(yim_test)+1
    ycancer = ["Nthy", "FTC"]
    for img_index in range(0,nimg):
        n_img = np.where(yim_test == img_index)
        if np.size(n_img) > 0:
            ncells = int(np.max(ycell[n_test][n_img]))+1
            for cell_index in range(0,ncells):
                n_sample = np.where(np.logical_and(ycell[n_test] == cell_index, yim_test == img_index))
                if np.size(n_sample) > 0:
                    n_cell += 1
                    yc_cell = yc_test[n_sample]
                    X_cell = X_test[n_sample]
                    
                    if weighted:
                        #change prediction to account for the P(Y)
                        y_nei_nc = kNN.predict_proba(X_cell)[:,0]*n_neighbors
                        y_nei_can = kNN.predict_proba(X_cell)[:,1]*n_neighbors
                        y_pred_proba = y_nei_can*ratio/(y_nei_can*ratio+y_nei_nc)
                        index_proba = np.sum(y_pred_proba)/np.size(y_pred_proba)
                
                        y_pred = y_pred_proba > 0.5
                        index = np.sum(y_pred)/np.size(y_pred)
                        
                    else:
                        y_pred = kNN.predict(X_cell)[:,1]
                        index = np.sum(y_pred)/np.size(y_pred)
                    
                    real_index = yc_cell[0]
                    
                    if plot:
                        colors = ["blue", "red"]
                        colors_c = []
                        for i in range(len(yc_train)):
                            colors_c.append(colors[int(yc_train[i])])
                        plt.scatter(X_train[:,0], X_train[:,1], color = colors_c, alpha=0.1)   
                        plt.scatter(X_cell[:,0], X_cell[:,1], color = "black", alpha=1)   
                        plt.title("PCA of Raman wavenumbers Jan/March/June/July")
                        plt.xlabel("PC1")
                        plt.ylabel("PC2")
                        plt.show()
                    
                    
                    
                    if index>0.5:    
                        prediction = 1
                    else:
                        prediction = 0
                        
                    if ycancer[prediction] == ycancer[real_index]:
                        acc += 1
                        
                    if plot:    
                        if ycancer[prediction] == ycancer[real_index]:
                            print(f"\n     {Fore.GREEN}Classifying %s cell [%s,%s]{Style.RESET_ALL}" % (ycancer[real_index],img_index,cell_index))
                            print(f"     {Fore.GREEN}Cancer index is %.3f{Style.RESET_ALL}" % index)
                            print(f"     {Fore.GREEN}Cancer index proba is %.3f{Style.RESET_ALL}" % index_proba)
                            
                        else:
                            print(f"\n     {Fore.RED}Classifying %s cell [%s,%s]{Style.RESET_ALL}" % (ycancer[real_index],img_index,cell_index))
                            print(f"     {Fore.RED}Cancer index is %.3f{Style.RESET_ALL}" % index)
                            print(f"     {Fore.RED}Cancer index proba is %.3f{Style.RESET_ALL}" % index_proba)
                            
                    true_index_.append(real_index)
                    pred_index_.append(index)
                        
    
    acc /= n_cell 
    if plot:     
        print("\n  Prediction accuracy with %s cells is %s/%s = %.3f" % (n_cell, int(round(acc*n_cell,0)), n_cell, acc))
    
    true_index = np.array(true_index_)
    pred_index = np.array(pred_index_)
    
    return acc,n_cell,pred_index,true_index







    


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
    nw = X.shape[1]
    
    x = np.linspace(-1,1,nw)
    cheb = chebyshev_polygen(degree,x)
    
    bsp = np.zeros((nx,nw));
    for xi in range(nx):
        bsp[xi,:] = cheb_polyfit(cheb,X[xi,:],tol,max_iter)
            
    return bsp  





def baseline_asymetric_least_square(X, lam=2e4, p=0.05, niter=10):
    
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
    nw = X.shape[1]
       
    bsp = np.zeros((nx,nw));
    for xi in range(nx):
        bsp[xi,:] = baseline_als(X[xi,:],lam,p,niter)
            
    return bsp   


def offset_correction(X):
    
    nx = X.shape[0]
    nw = X.shape[1]
    
    bsp = np.zeros((nx,nw));
    for xi in range(nx):
        bsp[xi,:] = np.min(X[xi,:])
    
    return bsp         
    
      
    
def wavelet_transform(X,wav_levels,max_level,wav_type):
    
    
    def wavelet_trans(y,wav_levels,max_level,wav_type):
        w = pywt.Wavelet(wav_type)
        C, L = wavedec(y, wavelet=w, level=max_level)
        D = np.zeros(len(y))
        for l in wav_levels:
            D += wrcoef(C, L, wavelet=w, level=l)
    
        """
        plt.plot(y)
        plt.plot(D)
        plt.plot(y-D)
        plt.show()  
        """
        
        return y-D,D
    
    
    nx = X.shape[0]
    nw = X.shape[1]
       
    low_freq = np.zeros((nx,nw))
    high_freq = np.zeros((nx,nw))
    for xi in range(nx):
        low_freq[xi,:], high_freq[xi,:] = wavelet_trans(X[xi,:],wav_levels,max_level,wav_type)
            
    return low_freq, high_freq  



def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction'''
    """
    Input:
        input_data = 2 dimensional array
        reference = 1 dimensional reference spectrum
    Output:
        normalized 2d array
    """
    #https://www.idtools.com.au/two-scatter-correction-techniques-nir-spectroscopy-python/

    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()

    # Get the reference spectrum. If not given, estimate it from the mean    
    if reference is None:    
        # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference

    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 

    return (data_msc, ref)