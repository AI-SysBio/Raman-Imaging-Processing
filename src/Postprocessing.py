
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes
from libPostprocessing import *



"""
Perform PCA and spectras postprocessing, 

Each file contains 
               data: n examples each with f wavenumbers (n*f 2d array), not normalized
         wavenumber: 1d array f, corresponding to the wavenumbers values
         cell_label: 1d array n, cell label within image (different for each cell in one image)
        image_label: 1d array n, file label (different for each image)
       cancer_label: 1d array n, 0 for nthy and 1 for cancer
         date_label: 1d array n, 0 for Jan, 1 for March, 2 for June, 3 for July
      nucleus_label: 1d array n, 1 if it's from nucleus, 1 if on periphery
           filename: 1d array n, original filename of each image
"""

def process_spectras(filename, base_correc):

    
    # ------------------------------------------------------------------------------- #
    # -------------------------- Postprocessing parameters -------------------------- #    
    # ------------------------------------------------------------------------------- #
    
    SVD = True; svd_components = 7  #singular value decomposition
    
    wavelet = False
    wav_levels = [3,7]  #which levels to keep after wavelet decomposition
    wav_basis='db8'; max_level = 8
    
    remove_silent_region = True  #set 1800cm-1 - 2800cm-1 to intensity 0
    
    remove_background = True  #subtract the mean spectra from noncell regions
    
    baseline_correction = "polyfit" #"polyfit","als","offset"
    
    normalize = True     
    normalization = "area" #"area","snv","msc"    
    without_chang_wv = False #True if you want to normalize only on the low wavenumber region (that is not important for FTC diagnosis)

    
    translate_Nthy = False  #if you want to perform covariate shift with nthy image as a reference
                             #(if there is no Nthy for a given date, shifting is not performed for that date)
                             
                             
    if base_correc == "offset":
        remove_background = True
        baseline_correction = "offset"
        
    elif base_correc == "poly":
        remove_background = True
        baseline_correction = "polyfit"
        
    elif base_correc == "als":
        remove_background = True
        baseline_correction = "als"
        
    elif base_correc == "poly_only":
        remove_background = False
        baseline_correction = "polyfit"
    
   # --------------------------------------------------------------------------------------------------------- #   
    
    
    
    
    
    data = np.load(filename)
    X = data["rawdata"]
    X_back = data["rawbackground"]
    wn = data["wavenumber"]


    yi = data["date_label"]
    yc = data["cancer_label"]
    yl = data["line_label"]
    ycell = data["cell_label"]  
    yim = data["image_label"] 
    yfile = data["filename"] 
    ypos = data["spectra_position"]
    
    Remove_Date = False; Date = 4
    if Remove_Date:
        nkeep = np.where(yi != Date)
        X = X[nkeep]
        yc = yc[nkeep]
        yi = yi[nkeep]
        yim = yim[nkeep]
        ycell = ycell[nkeep]   
        yfile = yfile[nkeep]
        ypos = ypos[nkeep]
        
        
    N = X.shape[0]
    f = X.shape[1]            
    print("   ", filename)
    print("    n = ", N, " spectras" )
    print("    f  = ", f, " features" ) 
      

    #Remove the background
    X2 = np.copy(X)
    if remove_background:
        print("\n   Removing the background...")
        for ni in range(N):
            X2[ni,:] -= X_back[yim[ni]-1,:]
            
        
    if SVD:
        print("\n   Starting SVD...")
        u, s, v = np.linalg.svd(X2.reshape(-1, X2.shape[-1]), full_matrices=False)
        s[svd_components:] = 0
        X2 = np.matmul(np.matmul(u, np.diag(s)), v).reshape(X2.shape)
        

        
    if baseline_correction in {"offset"}:
        print("\n   Substracting the offset...")
        B = offset_correction(X2)
        X2 -= B
    elif baseline_correction in {"polyfit"}:
        print("\n   Applying baseline correction polyfit..." )
        B = baseline_recursive_polynomial_chebyshev(X2, wn, 7)
        X2 -= B
    elif baseline_correction in {"als"}:
        print("\n   Applying baseline correction ALS..." )
        B = baseline_asymetric_least_square(X2)
        X2 -= B
        
        
    if remove_silent_region:
        print("\n   Removing silent region..." )
        w1 = 1800
        w2 = 2800
        i1 = np.argmin(abs(wn-w1))
        i2 = np.argmin(abs(wn-w2))
        X2[:,i1:i2] = 0      
        
        
    X_notnorm = np.copy(X2)
        
        
    if normalize:
        print("\n   Normalization..." )
        X3 = np.copy(X2)
        
        if normalization in {"area"}:
            if without_chang_wv:
                wh = np.argmin(abs(wn-2800))
                X3[:,wh:] = 0
            for i in range(X2.shape[0]): 
                sumW = np.sum(np.abs(X3[i,:]))
                if sumW != 0:
                    X2[i,:] /= sumW/f   
                    
        elif normalization in {"snv"}:
            #https://www.labcognition.com/onlinehelp/en/standard_normal_variate_correction.htm
            for i in range(X2.shape[0]): 
                ai = np.mean(X3[i,:])
                si = np.std(X3[i,:])
                X2[i,:] = (X3[i,:] - ai)/si     
                
        elif normalization in {"msc"}:
            #http://wiki.eigenvector.com/index.php?title=Advanced_Preprocessing:_Sample_Normalization
            X2 = msc(X3)
                
        
    if wavelet:
        print("\n   Starting Wavelet transform...")
        B,_ = wavelet_transform(X2,wav_levels,max_level,wav_basis)
        X2 -= B 
        

    
    X_trans = np.copy(X2)
    ndates = np.max(yi) +1
    if translate_Nthy:
        print("\n  Shifting center of gravity to 0 for each wavenumber...")
        X_trans = np.zeros((N,f))
        for j in range(ndates):
            nimg = np.where(yi == j)
            if np.size(nimg) != 0:
                for wi in range(f):
                    wimg = np.mean(X2[nimg,wi])
                    X_trans[nimg,wi] = X2[nimg,wi] - wimg 

    X = np.copy(X_trans)    
    
    nw = X.shape[1]
    nimg = np.max(yim)+1
    ndate = np.max(yi)+1
    X_av = np.empty(shape=[0, nw])
    y_c = np.array([])
    y_i = np.array([])
    y_im = np.array([])
    y_l = np.array([]) 
    y_file = []
    
    
    for date_index in range(0,ndate):
        n_date = np.where(yi == date_index)
        if np.size(n_date) > 0:
            for img_index in range(0,nimg):
                n_img = np.where(yim[n_date] == img_index)
                if np.size(n_img) > 0:
                    ncells = int(np.max(ycell[n_date][n_img]))+1
                    for cell_index in range(0,ncells):
                        n_sample = np.where(np.logical_and(np.logical_and(ycell == cell_index, yim == img_index),yi == date_index))
                        if np.size(n_sample) > 0:
                            
                            X_cell = X[n_sample[0],:]
                            X_av = np.append(X_av, np.mean(X_cell,axis=0).reshape(1,-1), axis=0)
                            y_c = np.append(y_c,yc[n_sample][0])
                            y_i = np.append(y_i,yi[n_sample][0])
                            y_im = np.append(y_im,yim[n_sample][0]) 
                            y_l = np.append(y_l,yl[n_sample][0])  
                            y_file.append(yfile[n_sample][0])
                    
                    
    data = {"wavenumber":wn, "data": X_trans, "notnormdata": X_notnorm, "cancer_label": yc, "date_label": yi, "line_label": yl, "image_label": yim, "cell_label":ycell, "spectra_position":ypos, "filename":yfile }  
    data_av = {"wavenumber":wn, "data": X_av, "cancer_label": y_c, "date_label": y_i, "line_label": y_l, "image_label": y_im, "filename":y_file }
    
    return data, data_av

                   


 
                    
def plot_spectra(data,norm = True):
    
    if norm:
        X = data["data"]
    else:
        X = data["notnormdata"]
    yc = data["cancer_label"]
    wn = data["wavenumber"]
    
    #broken plot
    nnc = np.where(yc == 0)
    ncan = np.where(yc == 1) 
    Mean1 = np.mean(X[ncan],axis=0)
    Mean2 = np.mean(X[nnc],axis=0) 

    w1 = 1800
    w2 = 2800
    i1 = np.argmin(abs(wn-w1))
    i2 = np.argmin(abs(wn-w2))     
    wn1 = wn[:i1]
    wn2 = wn[i2:]
    x = wn
    lower_CI = np.mean(X,axis=0) - np.std(X,axis=0)
    upper_CI = np.mean(X,axis=0) + np.std(X,axis=0)
    
    sps1, sps2 = GridSpec(2,1)
    fig = plt.figure(figsize=(10,6))
    fig.subplots_adjust(hspace=0.5)
    
    bax = brokenaxes(xlims=((min(wn1),max(wn1)), (min(wn2)-20,max(wn2))), hspace=0.15, subplot_spec=sps1)
    bax.plot(x, Mean2, color = "blue", linewidth = 2, label = "NthyOri 3-1")
    bax.plot(x, Mean1, color = "red", linewidth = 2, label = "FTC133")
    bax.fill_between(wn, lower_CI, upper_CI, color = '#539caf', alpha = 0.4)
    bax.legend(loc="upper left")
    bax.set_ylabel('Intensity [au]')  
    bax.set_xlabel('Wavenumber [cm-1]')
    bax.tick_params(axis='both', labelleft = False, labelbottom = False, bottom = False)
    
    bax = brokenaxes(xlims=((min(wn1),max(wn1)), (min(wn2)-20,max(wn2))), hspace=0.15, subplot_spec=sps2)
    bax.axhline(y=0, color='black', linestyle='--',alpha=0.3)
    bax.plot(x, Mean1-Mean2, color = "black", linewidth = 2)
    bax.fill_between(wn, lower_CI-np.mean(X,axis=0), upper_CI-np.mean(X,axis=0), color = '#539caf', alpha = 0.4)
    bax.set_ylabel('$\\Delta$ Intensity [au]') 
    bax.set_ylim(-1.5,1.5)
    bax.tick_params(axis='y', labelleft = False)
    
    plt.show()
    
    
    
    
    
    
def plot_pca(data, alpha = 0.1):
    
    plt.rcParams.update({'font.size': 14})
    
    print("\n  plotting PCA ...")
    
    X = data["data"]
    yc = data["cancer_label"]
    yi = data["date_label"]
    yl = data["line_label"]    
       

    
    Nkeep = 10000
    if X.shape[0] > Nkeep:
        nkeep = np.random.randint(X.shape[0],size=Nkeep)
        X = X[nkeep,:]
        yc = yc[nkeep]
        yi = yi[nkeep]
        yl = yl[nkeep]
    
    pca = PCA(n_components = 5)
    X_reduced = pca.fit_transform(X)
    
    
    colors = ["blue", "red", "green", "gold", "darkviolet", "cadetblue", "navy", "olive", "darkcyan", "darkorange", "brown", "coral", "chartreuse", "orange", "crimson", "chocolate", "maroon","dodgerblue","goldenrod", "darkred","darkblue","blue", "red", "gold", "darkviolet", "cadetblue", "navy", "olive", "darkcyan", "darkorange", "brown", "coral", "chartreuse", "orange", "crimson", "chocolate", "maroon","dodgerblue","goldenrod", "darkred","darkblue","blue", "red", "gold", "darkviolet", "cadetblue", "navy", "olive", "darkcyan", "darkorange", "brown", "coral", "chartreuse", "orange", "crimson", "chocolate", "maroon","dodgerblue","goldenrod", "darkred","darkblue"]
    
    colors_c = []
    for i in range(len(yc)):
        colors_c.append(colors[int(yc[i])])
    
    colors_i = []
    for i in range(len(yi)):
        colors_i.append(colors[int(yi[i])+3])
        
    colors_l = []
    for i in range(len(yl)):
        colors_l.append(colors[int(yl[i])])
        
                       
    
    plt.scatter(X_reduced[:,0], X_reduced[:,1], color = colors_i, alpha=alpha)   
    plt.title("PCA of Raman wavenumbers by date")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("off")
    plt.show()
       
    plt.scatter(X_reduced[:,0], X_reduced[:,1], color = colors_c, alpha=alpha)   
    plt.title("PCA of Raman wavenumbers by malignancy")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("off")
    plt.show()

    """    
    plt.scatter(X_reduced[:,0], X_reduced[:,1], color = colors_l, alpha=0.1)   
    plt.title("PCA of Raman wavenumbers by cell line")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("off")
    plt.show()
    """
    
    print("   first 2 PCA explained variance is %.2f" % (pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]))
        