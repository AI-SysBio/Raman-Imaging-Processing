
import os
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage
from scipy.interpolate import interp1d
from skimage.segmentation import slic

from Cell_segment import process_cells
from libRaman import Identify_Background, Irradiance_Profile_Correction, cosmicray_detect, baseline_recursive_polynomial_chebyshev


"""
    Raw Raman hyperspectral image files contain:
        - origimagedata: Raw data ndarray of shape of (Width, Wavenumbers, Height)
        - wavenumber: 1-d ndarray corresponding to the 2nd axis of data
        
    Process the hyperspectral Raman image with:  
          - Cosmic ray detection
          - Bias correction (proper to the device)
          - Correction for irradiance profile
          - Subtraction of BG spectrum
          - SVD denoising
          - Polyfit fluorescence correction
          - Spectra area normalization
        
    [ref] A. Pelissier, K. Hashimoto, K. Mochizuki, Y. Kumamoto, J. N. Taylor, K. Tabata, 
          JE. Clement, A. Nakamura,  Y. Harada, K. Fujita and T. Komatsuzaki. 
          Intelligent Measurement Analysis on Single Cell Raman Images for the Diagnosis 
          of Follicular Thyroid Carcinoma. arXiv preprint. (2019).
"""


def process_all(Folder, superpixel_size, cell_cluster_factor):
    
    if not os.path.exists("Superpixel_Measurements"):
        os.makedirs("Superpixel_Measurements")
        
    if not os.path.exists("Spectra_analysis"):
        os.makedirs("Spectra_analysis")
    
    erase_processed = False
    erase_cell_seg = False
    


    
    
    
    # ---------------------------------------------------------------------------
    # 1 - process all files -----------------------------------------------------
    # ---------------------------------------------------------------------------

    for fname in os.listdir(Folder):
        if ".mat" in fname:
            
            fpath = "%s/%s" % (Folder, fname)
                
            if "ftc" in fpath or "nthy" in fpath or "Nthy" in fpath or "FTC" in fpath or "HOTHC" in fpath or "RO82" in fpath or "8305" in fpath or "8505" in fpath:
                    
                base, ext = os.path.splitext(fname)
                preprofile = "Superpixel_Measurements\pp_%s.npz" % (base)
                    
                    

                if not os.path.isfile(preprofile) or erase_processed:  
                    print("\n   Starting Raman Data Processing of %s..." % base)
                    data = process_file(fpath, fname, superpixel_size)
                    np.savez(preprofile, **data)
                else:
                    print("   Detecting %s..." % base)
                    
                    
    # ---------------------------------------------------------------------------        
    # 2 - put it all together and keep only cell regions ------------------------
    # ---------------------------------------------------------------------------          
    
    cell_file = "Spectra_Analysis/cells_spectras_%s_%s.npz" % (superpixel_size, cell_cluster_factor)
    cell_file_av = "Spectra_Analysis/cells_spectras_%s_%s_av.npz" % (superpixel_size, cell_cluster_factor)
    if not os.path.isfile(cell_file) or erase_cell_seg or erase_processed:
        spectra_data, spectra_data_av = process_cells(superpixel_size, cell_cluster_factor)
        np.savez(cell_file, **spectra_data)
        np.savez(cell_file_av, **spectra_data_av)
    else:
        print("\n   Using %s ..." % cell_file[17:])
        spectra_data = np.load(cell_file)
        spectra_data_av = np.load(cell_file_av)
        
    return spectra_data, spectra_data_av
        
 



def process_file(fpath, fname, superpixel_size):
    
    fdisp = fname[:-4]
    
    data = hdf5storage.loadmat(fpath)    
    X_init = data["origimagedata"]
    if X_init.shape[2]:
        if X_init.shape[2] < max(X_init.shape[0],X_init.shape[1]):
            X_init = np.transpose(data["origimagedata"], (2, 0, 1))
    wn =  data["wavenumber"].reshape(-1)
    

   
    # preliminiray processing ---------------------------------------------------------------------
    
    cut_side = False
    if X_init.shape[1] == 400:
        cut_side = True
    
    if cut_side:
        X_init = X_init[:,2:-2,:] #cute the sides
    X = np.copy(X_init)
    
    
    #0 - Cosmic Ray detection -------------------
    print("\n   Detecting Cosmic rays...")
    ignore = cosmicray_detect(X)  
    
    
    Shape = np.shape(X)
    nx = Shape[0]
    ny = Shape[1]
    nw = Shape[2] 
    
    
    #0 - Bias correction ------------------------
    if 'no' in fname: #bias correction should be applied only to Osaka
        print("\n   Applying bias correction...")
        X -= 520
    
    #0 - Correct for Irradiance profile ------------------------
    print("\n   Starting Irradiance profile correction...")
    X_ir = Irradiance_Profile_Correction(X)
        
    #0 - Compute background mask
    print("\n   Starting Background region identification...")
    compute_new_background = True
    mask_back2D = Identify_Background(X_ir,wn, ignore, fdisp, compute_new_background)
        
   
    
    
    #1 - Superpixels -------------------------------------------------------
    print("\n   Superpixel segmentation...")
    wb1 = np.argmin(abs(wn-2800))
    wb2 = np.argmin(abs(wn-3000))
    wavenumber_index_0 = np.arange(wb1,wb2)
            
    Xmean = np.mean(X_ir[:,:,wavenumber_index_0], axis=2)
    #Xmean = np.mean(X_ir, axis=2)
    vmax = max(Xmean[ignore==False])*0.9
    image = np.zeros((nx,ny,3))
    for xi in range(nx):
        for yi in range(ny):
            image[xi,yi,0] = 0
            image[xi,yi,1] = Xmean[xi,yi]/vmax
            image[xi,yi,2] = Xmean[xi,yi]/vmax
            
    numSegments = int(nx*ny/superpixel_size)
    segments = slic(image, n_segments = numSegments, sigma = 5)
            
                
    num_pix = np.max(segments)+1
    X_superpixel = np.zeros((num_pix,nw))
    X_pos = np.zeros((num_pix,2))
            
    for i in range(num_pix):
        xs, ys = np.where(np.logical_and(segments==i, ignore==False))
        
        if np.size(xs) == 0:  
            X_superpixel[i,:] = np.zeros(nw)
        else:
            X_superpixel[i,:] = np.mean(X[xs,ys,:],axis=(0))
        
        if len(xs) >0:
            xav = int(np.mean(xs))
            yav = int(np.mean(ys))
            X_pos[i,0] = xav
            X_pos[i,1] = yav
            
        else:
            X_pos[i,0] = -1
            X_pos[i,1] = -1
            
            
            
            
            
            
    #2 - Interpolate ---------------------------------------------------------------------
    f = 600 #number of wavenumbers in interpolated plot
    wmin = 700
    wmax = 3000
    
    X_out = np.zeros((num_pix,f))
    w_new = np.linspace(wmin, wmax, num=f)
    for i in range(num_pix):
        fun = interp1d(wn, X_superpixel[i,:], kind='cubic')
        X_out[i,:] = fun(w_new)
        
        
        
        
        
    # 3 - Remove the background ---------------------------------------------------------------------
    print("\n   Starting Non-cell region subtraction...")

    mask_back = np.zeros(X_out.shape[0])
    for i in range(X_out.shape[0]):
        mask_back[i] = mask_back2D[int(X_pos[i,0]),int(X_pos[i,1])]
        if ignore[int(X_pos[i,0]),int(X_pos[i,1])] == 1:
            mask_back[i] = 0
        
    nb = np.where(mask_back==1)
    Xback = np.mean(X_out[nb], axis = 0)
    X_bk = np.zeros(X_out.shape)
    for xi in range(X_out.shape[0]):
            X_bk[xi,:] = X_out[xi,:] - Xback
            
            
    for xi in range(X_out.shape[0]):
        plt.plot(X_bk[xi,:])
    plt.show()
            
            
            
            
            
            
    
    # 4 - SVD ---------------------------------------------------------------------
    svd_components = 7
    print("\n   Starting SVD...")
    u, s, v = np.linalg.svd( X_bk.reshape(-1,  X_bk.shape[-1]), full_matrices=False)
    s[svd_components:] = 0
    X_bk = np.matmul(np.matmul(u, np.diag(s)), v).reshape( X_bk.shape)       







    #5 - polyfit --------------------------------------------------------------------- 
    print("\n   Starting Recursive Chebyshev Polynomial Fitting...")   
    rpf_degree = 7
    B = baseline_recursive_polynomial_chebyshev(X_bk, w_new, rpf_degree)
    X_bk -= B


    """
    # Offset correction
    B = offset_correction(X_bk)
    X_bk -= B
    """

    
     
    
    
    
        
    #6 - remove silent region ---------------------------------------------------------------------
    print("\n   Removing Silent Region...")
    w1 = 1800
    w2 = 2800
    i1 = np.argmin(abs(w_new-w1))
    i2 = np.argmin(abs(w_new-w2))
    Xcuts = np.copy(X_bk)
    Xcuts[:,i1:i2] = 0
            
        
    
            
            
            
    #7 - area normalize ---------------------------------------------------------------------
    print("\n   Starting Normalization...")
    
    
    X_final = np.zeros(Xcuts.shape)
    for i in range(num_pix): 
        sumW = np.sum(np.abs(Xcuts[i,:]))
        if sumW != 0:
            X_final[i,:] = Xcuts[i,:]/sumW*f
            
    
    plt.plot(np.mean(X_final,axis = 0))
    plt.show()
            



    return { "data" : X_final, "rawdata" : X_out, "background" : Xback, "position": X_pos, "wavenumber": w_new}

    
    

