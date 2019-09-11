
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from libRaman import baseline_recursive_polynomial_chebyshev

"""
    Compute the cell and nucleus mask of each image:  
        
    [ref] A. Pelissier, K. Hashimoto, K. Mochizuki, Y. Kumamoto, J. N. Taylor, K. Tabata, 
          JE. Clement, A. Nakamura,  Y. Harada, K. Fujita and T. Komatsuzaki. 
          Intelligent Measurement Analysis on Single Cell Raman Images for the Diagnosis 
          of Follicular Thyroid Carcinoma. arXiv preprint. (2019).
          
    -> All images are processed and concatenated together into a npz file containing:
               data: processed data, n examples each with f wavenumbers (n*f 2d array)
            rawdata: unprocessed data, n examples each with f wavenumbers (n*f 2d array)
         wavenumber: 1d array f, corresponding to the wavenumbers values
         cell_label: 1d array n, cell label within image (different for each cell in one image)
        image_label: 1d array n, file label (different for each image)
       cancer_label: 1d array n, 0 for nthy and 1 for cancer
         date_label: 1d array n, 0 for Jan, 1 for March, 2 for June, 3 for July
      nucleus_label: 1d array n, 1 if it's from nucleus, 1 if on periphery
           filename: 1d array n, original filename of each image
           position: X and Y position of each superpixel in each img
"""



def process_cells(superpixel_size, cell_cluster_factor):

    
    if not os.path.exists("Superpixel_Measurements/Nucleus"):
        os.makedirs("Superpixel_Measurements/Nucleus") 
    
    dirname = "./Superpixel_Measurements/"
    compute_nucleus_center = True  #say False if you want to use nucelus centers computed before
    
    f = 600
    nfile = 0
    cell_index_list = []
    cancer_index_list = []
    line_index_list = []
    date_index_list = []
    nucleus_index_list = []
    img_index_list = []
    original_filename_list = []
    ypos_list = []
    X_specs = np.empty(shape=[0, f])
    X_raw_specs = np.empty(shape=[0, f])
    
    
    for filename in os.listdir(dirname):
        
        if "ftc" in filename or "nthy" in filename or "Nthy" in filename or "FTC" in filename or "RO82" in filename:

            nfile += 1
            
            fpath = "%s/%s" % (dirname, filename) 
            fdisp = filename[3:-4]
            data = np.load(fpath)
           
            X = data["data"]
            Xraw = data["rawdata"]
            X_pos = data["position"]
            wn = data["wavenumber"]
            
            
            #1 - Compute the mask for cell segmentation
            mask_nucleus, mask_kcells = compute_masks(X,Xraw, X_pos, wn,fdisp, superpixel_size, cell_cluster_factor, compute_nucleus_center)

            
            #2 - process and merge all the images to one dataset
            X_out_list = []
            X_outr_list = []
            for i in range(X.shape[0]):
                xav = int(X_pos[i,0])
                yav = int(X_pos[i,1])
                if mask_kcells[i]:
                       
                        X_out_list.append(X[i,:])
                        X_outr_list.append(Xraw[i,:])
                           
                        nucleus_index_list.append(mask_nucleus[i])
                           
                        cell_index_list.append(mask_kcells[i])
                            
                        img_index_list.append(nfile)
                            
                        ypos_list.append([xav,yav])
                            
                        original_filename_list.append(fdisp)
                            
                                
                        if "Nthy" in filename or "nthy" in filename:
                             cancer_index_list.append(0)     
                             line_index_list.append(0)
                        elif "ftc" in filename or "FTC" in filename:
                            cancer_index_list.append(1)
                            line_index_list.append(1)
                        elif "RO82" in filename:
                            cancer_index_list.append(1)
                            line_index_list.append(2)
                        elif "HOTHC" in filename:
                            cancer_index_list.append(2)
                            line_index_list.append(3)
                        elif "8305C" in filename:
                            cancer_index_list.append(2)
                            line_index_list.append(4)
                        elif "8505C" in filename:
                            cancer_index_list.append(2)
                            line_index_list.append(5)
                        else:
                            cancer_index_list.append(-1)
                            line_index_list.append(-1)
                                
                        date_index_list.append(0)

                                  

            """
            print(fdisp)
            wn0 = np.argmin(abs(wn-2800))
            wn4 = np.argmin(abs(wn-3000))
            wavenumber_index_0 = np.arange(wn0,wn4)
            score = np.sum(np.mean(np.array(X_out_list)[:,wavenumber_index_0], axis = 0))
            print(score)
            plt.plot(np.mean(np.array(X_out_list), axis = 0))
            plt.title(score)
            plt.savefig("Test/%s.png" % fdisp)
            plt.show()
            """
            
            X_specs = np.append(X_specs, np.array(X_out_list), axis=0)
            X_raw_specs = np.append(Xraw, np.array(X_outr_list), axis=0)
            
            
    
    ycell = np.array(cell_index_list)  
    yc = np.array(cancer_index_list)
    yl = np.array(line_index_list)
    yd = np.array(date_index_list)
    yn = np.array(nucleus_index_list)
    yim = np.array(img_index_list)
    ypos = np.array(ypos_list)
    
    
    
    #save the processed spectra, and the corresponding average spectra of each cells

    nw = X_specs.shape[1]
    nimg = np.max(yim)+1
    ndate = np.max(yd)+1
    X_av = np.empty(shape=[0, nw])
    y_c = np.array([])
    y_i = np.array([])
    y_im = np.array([])
    y_l = np.array([]) 
        
    for date_index in range(0,ndate):
        n_date = np.where(yd == date_index)
        if np.size(n_date) > 0:
            for img_index in range(0,nimg):
                n_img = np.where(yim[n_date] == img_index)
                if np.size(n_img) > 0:
                    ncells = int(np.max(ycell[n_date][n_img]))+1
                    for cell_index in range(0,ncells):
                        n_sample = np.where(np.logical_and(np.logical_and(ycell == cell_index, yim == img_index),yd == date_index))
                        if np.size(n_sample) > 0:
                            
                            X_cell = X_specs[n_sample[0],:]
                            X_av = np.append(X_av, np.mean(X_cell,axis=0).reshape(1,-1), axis=0)
                            y_c = np.append(y_c,yc[n_sample][0])
                            y_i = np.append(y_i,yd[n_sample][0])
                            y_im = np.append(y_im,yim[n_sample][0]) 
                            y_l = np.append(y_l,yl[n_sample][0]) 
                    
    data = {"rawdata": X_raw_specs, "data": X_specs, "wavenumber": wn, "cell_label": ycell, "cancer_label": yc, "line_label": yl, "image_label": yim, "date_label": yd, "nucleus_label": yn, "filename": original_filename_list, "spectra_position": ypos }
    data_av = {"wavenumber":wn, "data": X_av, "cancer_label": y_c, "date_label": y_i, "line_label": y_l, "image_label": y_im}        
       
    return data, data_av
     





def compute_masks(X, X_raw, X_pos, wn, fdisp, superpixel_size, cell_cluster_factor, compute_nucleus_center):
    
    print("\n   Computing cell mask for %s" % fdisp)
    
    nucleus_file = "Superpixel_Measurements/Nucleus/nucleus_centers_%s.npy" % fdisp
    if compute_nucleus_center or not os.path.isfile(nucleus_file):
        compute_nucleus_center = True
        
    #1  Identify cell regions
    mask_cell = compute_mask_cell(X_raw, wn, cell_cluster_factor)
    
    #2 Identify Nucleus regions
    mask_nucleus = compute_mask_nucleus(X, X_pos, wn, superpixel_size, fdisp, compute_nucleus_center, mask_cell)
    
    #3  Segment_cell cell regions
    mask_kcells = segment_cells(mask_nucleus, X, X_pos, wn, superpixel_size, fdisp, compute_nucleus_center, mask_cell)
    
    return mask_nucleus, mask_kcells





def compute_mask_cell(X, wn, cell_cluster_factor):
    
    #1 - SVD
    svd_components = 7
    print("\n      Starting SVD...")
    u, s, v = np.linalg.svd( X.reshape(-1,  X.shape[-1]), full_matrices=False)
    s[svd_components:] = 0
    X = np.matmul(np.matmul(u, np.diag(s)), v).reshape( X.shape)       
    

    #2 - Polyfit
    print("      Starting Recursive Chebyshev Polynomial Fitting...")   
    rpf_degree = 7
    B = baseline_recursive_polynomial_chebyshev(X, wn, rpf_degree)
    X -= B
    
    #3 - normalization
    print("      Starting Normalization...")
    w1 = 1800
    w2 = 2800
    i1 = np.argmin(abs(wn-w1))
    i2 = np.argmin(abs(wn-w2))
    X[:,i1:i2] = 0
            
    for i in range(X.shape[0]): 
        sumW = np.sum(np.abs(X[i,:]))
        if sumW != 0:
            X[i,:] = X[i,:]/sumW*X.shape[1]

            
    
    #4 - kMean clusturing
    print("      Starting kmean clusturing...") 
    mask_cell = np.zeros(X.shape[0])
    n_cluster = 10
    wn0 = np.argmin(abs(wn-2800))
    wn4 = np.argmin(abs(wn-3000))
    wavenumber_index_0 = np.arange(wn0,wn4)
    Xcell = np.mean(X[:,wavenumber_index_0], axis=1)
    kmeans_cell = KMeans(n_clusters=n_cluster).fit(Xcell.reshape(-1, 1))
    threshold_cell = sorted(kmeans_cell.cluster_centers_)[int(n_cluster*cell_cluster_factor)]
    for i in range(X.shape[0]):
        if kmeans_cell.cluster_centers_[kmeans_cell.labels_[i]] >= threshold_cell:
            mask_cell[i] = 1
            
    return mask_cell
            
            


            
def compute_mask_nucleus(X, X_pos, wn, superpixel_size, fdisp, compute_nucleus_center, mask_cell):
    
    #kMean clustering  on the ratio ch2stretch/ch3 stretch, just to identify the nucleus center
    
    mask_nucleus_file = "Superpixel_Measurements/Nucleus/nucleus_mask_%s.npy" % fdisp
    
    if compute_nucleus_center:
    
        mask_nucleus = np.zeros(X.shape[0])
        
        print("\n   Clustering Nucleus/Cytoplasm...")
    
        wn1 = np.argmin(abs(wn-2842))
        wn2 = np.argmin(abs(wn-2910))
        wn3 = np.argmin(abs(wn-2930))
        wn4 = np.argmin(abs(wn-3000))
        wavenumber_index_nuc1 = np.arange(wn1,wn2)
        wavenumber_index_nuc2 = np.arange(wn3,wn4)
        ncell = np.where(mask_cell == 1)
        n_cluster = 10
        
        Xnuc = (np.mean(X[:,wavenumber_index_nuc2], axis=1) - np.mean(X[:,wavenumber_index_nuc1], axis=1))# * np.mean(X[:,wavenumber_index_0], axis=1)      
        
        kmeans_nuc = KMeans(n_clusters=n_cluster).fit(Xnuc[ncell].reshape(-1, 1))
        threshold_nuc = sorted(kmeans_nuc.cluster_centers_)[int(n_cluster*0.7)]
    
        
        for i in range(Xnuc[ncell].shape[0]):
            if kmeans_nuc.cluster_centers_[kmeans_nuc.labels_[i]] >= threshold_nuc:
                mask_nucleus[ncell[0][i]] = 1
                
        
        plot_check = True        
        if plot_check:
                
            back = np.zeros((400,450)) 
            n_nuc = np.where(mask_nucleus == 1)
            plt.imshow(back, origin='lower')
            xs, ys = np.where(back==0)
            plt.scatter(ys, xs, label="background", s=0.3, c = "black")
            plt.scatter(X_pos[:,1]+15,X_pos[:,0]+15, c = np.mean(X,axis = 1), s = 3.5)
            plt.scatter(X_pos[ncell][:,1]+15,X_pos[ncell][:,0]+15, color = "red", cmap='RdYlGn', s = 3.5)
            plt.scatter(X_pos[n_nuc][:,1]+15,X_pos[n_nuc][:,0]+15, color = "purple", cmap='RdYlGn', s = 3.5)
            plt.axis("off")
            plt.savefig("Superpixel_Measurements/Nucleus/nucleus_%s.png" % fdisp)
            plt.show() 
            
        np.save(mask_nucleus_file, mask_nucleus)
            
    else:
        mask_nucleus = np.load(mask_nucleus_file)

    return mask_nucleus




def segment_cells(mask_nucleus, X, X_pos, wn, superpixel_size, fdisp, compute_nucleus_center, mask_cell):
    
    #take the first N clusters as pixel spectra, then for each pixel, identify it to the closest nucleus
    
    #1 - compute the Nucleus centers
    nucleus_file = "Superpixel_Measurements/Nucleus/nucleus_centers_%s.npy" % fdisp
        
    if compute_nucleus_center:
        radius_min = 300
        
        Xpos_nuc = X_pos[np.where(mask_nucleus == 1)]
        
        max_d_min = int(np.sqrt(superpixel_size))+2
        X_list = []
        for i in range(Xpos_nuc.shape[0]):
            X_list.append([Xpos_nuc[i,0],Xpos_nuc[i,1]])
        Xh = np.array(X_list)     
        Z = linkage(Xh, 'single')
        clusters = fcluster(Z, max_d_min, criterion='distance')
        n_clust = np.max(clusters)
        
        nucleus_centers = []
        n_spec_min = int((radius_min/superpixel_size)**2)
        for i in range(1,n_clust+1):
            ni = np.where(clusters==i)
            if np.size(ni) < n_spec_min:  #if the cluster is too small to be a nucleus
                clusters[ni]=0
            else:
                nucleus_centers.append([np.mean(Xpos_nuc[ni,0]),np.mean(Xpos_nuc[ni,1])])
                
        nucleus_centers = np.array(nucleus_centers)
        print(nucleus_centers)
        
        """
        plt.figure(figsize=(10, 7))  
        fancy_dendrogram(Z, p=840, truncate_mode='lastp', annotate_above=1000000, no_labels=True, max_d=max_d_min)
        plt.show()
        """
        
        np.save(nucleus_file,nucleus_centers)
    
    
    else:
        nucleus_centers = np.load(nucleus_file)
        print("     loading %s" % nucleus_file)
    
    #2 Associate each spectra to the closest nucleus
    mask_kcells = np.zeros(X.shape[0])
    
    if np.size(nucleus_centers) > 0:
        for i in range(X.shape[0]):
            mask_kcells[i] = np.argmin((X_pos[i,0] - nucleus_centers[:,0])**2 + (X_pos[i,1] - nucleus_centers[:,1])**2)+1
            
    else:
        for i in range(X.shape[0]):
            mask_kcells[i] = 1
        
        
        
    #3 Keep only the cells regions
    for i in range(X.shape[0]):
        if mask_cell[i] == 0:
            mask_kcells[i] = 0

    return mask_kcells






def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram')
        #plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
        