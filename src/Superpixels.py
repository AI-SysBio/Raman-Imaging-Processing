
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
import random
import scipy.io

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io



"""
The files named "pp_....npz" are preprocessed data.
FTC-133 is the raman data of follicular thyroid carcinoma.
Nthy-ori 3-1 is the raman data of normal human primary thyroid follicular epithelial cells. 

Each file contains 
     rawdata: Raw data ndarray of shape of (Width, Height, Wavenumbers)
        data: processed data ndarray of shape of (Width, Height, Wavenumbers)
    normdata: normalized processed data*1e3, ndarray of shape of (Width, Height, Wavenumbers)
  wavenumber: 1-d ndarray corresponding to the 3rd axis of data
      ignore: boolean ndarray of shape of (Width, Height)
              If the value is True, the spectrum at that position has to 
              be ignored since the intensity of that spectrum is extremely strong
              at some wavenumbers. 
              
              
This code take the preprocessed hyperspectral images and do as follow:
    1) Segment the image in superpixel spectra
    2) Cluster background,cell,nucleus
    3) Segment the cell spectra in different cells (with hierarchical clustering or manually segmented cells)
    4) Save the superpixel spectra (average spectrum over each superpixel)

"""

def save_image(data, fn):
   
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(data, origin='lower')
    plt.savefig(fn, dpi = height) 
    plt.close()
    
    

def superpixel(dirname,use_manual_mask,superpixel_size,dates):

    
    f = 600 #number of wavenumbers in interpolated plot
    wmin = 700
    wmax = 3000
    w_new = np.linspace(wmin, wmax, num=f)
    dirname = "./Processed_Measurements/"
    
    
    nfile = 0
    cell_index_list = []
    cancer_index_list = []
    date_index_list = []
    nucleus_index_list = []
    img_index_list = []
    original_filename_list = []
    ypos_list = []
    X_specs = np.empty(shape=[0, len(w_new)])
    X_raw = np.empty(shape=[0, len(w_new)])
    X_back = np.empty(shape=[0, len(w_new)])
    X_backr = np.empty(shape=[0, len(w_new)])
    
    for filename in os.listdir(dirname):
        
        if "ftc" in filename or "nthy" in filename or "Nthy" in filename or "FTC" in filename:
            
            nfile += 1
            
            fpath = "%s/%s" % (dirname, filename) 
            data = np.load(fpath)
            fdisp = filename[3:-4]
            
            Measurements = data["data"]
            normMeas = data["normdata"]
            rawMeas = data["rawdata"]
            wn = data["wavenumber"]
            
            Shape = np.shape(Measurements)
            nx = Shape[0]
            ny = Shape[1]
            nw = Shape[2]        
            
            print("\n")        
            print(" Measurements :")
            print("    nx = ", nx, " points" )
            print("    ny = ", ny, " points" )
            print("    f  = ", nw, " features" )
            print("   ", fdisp)
            
            if not os.path.exists("Processed_Measurements/segmentation_superpixel"):
                os.makedirs("Processed_Measurements/segmentation_superpixel")

            if not os.path.exists("Spectra_Analysis"):
                os.makedirs("Spectra_Analysis")  
                
            if not os.path.exists("Spectra_Analysis/Classified_cells"):
                os.makedirs("Spectra_Analysis/Classified_cells")  
                
            Xmeanfull = np.mean(Measurements, axis=2)
            vmax = max(Xmeanfull[data["ignore"]==False])*0.9
            vmin = min(Xmeanfull[data["ignore"]==False]) 
            plt.imshow(Xmeanfull, vmax=vmax, vmin=vmin, origin='lower')
            plt.savefig("Processed_Measurements/segmentation_superpixel/%s_0av_or.png" %  (fdisp))
            plt.show()
            
            
            wb1 = np.argmin(abs(wn-2950))
            wb2 = np.argmin(abs(wn-2990))
            wavenumber_index_0 = np.arange(wb1,wb2)
            
            Xmean = np.mean(normMeas[:,:,wavenumber_index_0], axis=2)
            
            vmax = max(Xmean[data["ignore"]==False])*0.9
            vmin = min(Xmean[data["ignore"]==False])            
            plt.imshow(Xmean, vmax=vmax, vmin=vmin, origin='lower')
            ys, xs = np.where(data["ignore"])
            plt.scatter(xs, ys, s=0.7, c="r", label="cosmicray")
            plt.savefig("Processed_Measurements/segmentation_superpixel/%s_av_hv.png" %  (fdisp))
            plt.savefig("Spectra_Analysis/Classified_cells/%s_av_hv.png" %  (fdisp))
            plt.show()
            
            
            
      
            
            #load_mask
            if use_manual_mask:
                maskfile = ("Processed_Measurements/Maks_cells_manual/%s_mask.npy" % fdisp)
                if os.path.isfile(maskfile):
                    mask_kcells = np.load(maskfile)
                    #mask_kcells = np.flip(mask_kcells, axis=0)
                else:
                    print("Manually segmented mask is not found for image %s" % fdisp)
                    sys.exit(0)
            
            else:
                maskfile = ("Processed_Measurements/Mask_cells/%s_mask.npy" % fdisp)
                mask_kcells = segment_cells(dirname, filename)
                np.save(maskfile,mask_kcells)
           
            
            n_clust = int(np.max(mask_kcells))
            cm = plt.get_cmap('tab10')
            colors10 = cm.colors
            
            plt.imshow(np.zeros((nx,ny)), origin='lower')
            xs, ys = np.where(mask_kcells==0)
            plt.scatter(ys, xs, label="background", s=0.3, c = "black")
            for i in range(1,n_clust+1):
                xs, ys = np.where(mask_kcells==i)
                plt.scatter(ys, xs, label="cell %s" % i, s=0.3, c = colors10[(i-1)%10])
            #ys, xs = np.where(data["ignore"])
            #plt.scatter(xs, ys, s=0.7, c="r", label="cosmicray")
            #plt.legend(loc="lower left",markerscale=5)
            plt.title("%s: Cell clustering" % (fdisp))
            plt.axis('scaled')
            plt.savefig("Processed_Measurements/segmentation_superpixel/%s_seg2.png" %  (fdisp))
            plt.savefig("Spectra_Analysis/Classified_cells/%s_bseg.png" %  (fdisp))
            plt.show()   
            
            
            
        
        
            
            #Superpixel Segmentation
            print("\n  Superpixel segmentation...")
            image = np.zeros((nx,ny,3))
            for xi in range(nx):
                for yi in range(ny):
                    image[xi,yi,0] = 0
                    image[xi,yi,1] = Xmean[xi,yi]/vmax
                    image[xi,yi,2] = Xmean[xi,yi]/vmax
             
            numSegments = int(nx*ny/superpixel_size)
            segments = slic(image, n_segments = numSegments, sigma = 5)
             
            
            num_pix = np.max(segments)+1
            MEAN = np.zeros(num_pix)
            for i in range(num_pix):
                xs, ys = np.where(np.logical_and(segments==i, data["ignore"]==False))
                if np.size(xs) == 0:  
                    MEAN[i] = 0
                else:
                    MEAN[i] = np.mean(Xmean[xs,ys])
                
                      
            X_out = np.zeros((nx,ny))
            for xi in range(nx):
                for yi in range(ny):
                    X_out[xi,yi] = MEAN[segments[xi,yi]]
                    
            
            
            X_superpixel = np.zeros((num_pix,4))
            for i in range(num_pix):
                xs, ys = np.where(segments==i)
                xav = int(np.mean(xs))
                yav = int(np.mean(ys))
                X_superpixel[i,0] = xav
                X_superpixel[i,1] = yav
                X_superpixel[i,2] = mask_kcells[xav,yav]
                X_superpixel[i,3] = i
                
                
            plt.imshow(np.zeros((nx,ny)), origin='lower')
            xs, ys = np.where(mask_kcells==0)
            plt.scatter(ys, xs, label="background", s=0.3, c = "black")
            for i in range(num_pix):
                if X_superpixel[i,2]:
                    plt.scatter(X_superpixel[i,1], X_superpixel[i,0], s=15, c = colors10[(int(X_superpixel[i,2])-1)%10])   
            plt.savefig("Processed_Measurements/segmentation_superpixel/%s_zfinal.png" %  (fdisp))
            plt.show()
                
            ncells = n_clust
            print("    ", ncells,"cells in this image")
            
            
                
                
                
                
            #save in files
            ref_back = (mask_kcells == 0)
            ref_cell = (mask_kcells > 0)
            
            wthresh = 2932
            itresh = np.argmin(abs(wn-wthresh))
            xb, yb = np.where(np.logical_and(np.logical_and(ref_back, data["ignore"]==False), Measurements[:,:,itresh] <= 5, Measurements[:,:,itresh] >= 1))
            xc, yc = np.where(np.logical_and(ref_cell, data["ignore"]==False))
            print("    ", np.size(xb),"background spectra used")
            
            
            if np.size(xb) == 0:
                print("Image skipped because no background spectra")
                nfile -= 1
        
                
                
            else:
                plt.rcParams.update({'font.size': 14})
                mean_back = np.median(Measurements[xb,yb,:],axis=(0))
                mean_backr = np.median(rawMeas[xb,yb,:],axis=(0))
                mean_cell = np.median(rawMeas[xc,yc,:],axis=(0))
                funb = interp1d(wn, mean_back, kind='cubic')
                funbr = interp1d(wn, mean_backr, kind='cubic')
                funcr = interp1d(wn, mean_cell, kind='cubic')
                X_back = np.append(X_back,funb(w_new).reshape(1,-1), axis=0)
                X_backr = np.append(X_backr,funbr(w_new).reshape(1,-1), axis=0)
                plt.plot(w_new,offset_correction(funbr(w_new)), label = "background", color = "black")
                plt.plot(w_new,offset_correction(funcr(w_new)), label = "cell", color = "green")
                plt.legend()
                plt.xlabel("Wavenumber [cm-1]")
                plt.ylabel("Intensity [au]")
                plt.title("Mean spectra")
                plt.show()
                
                plt.imshow(X_out, origin='lower')
                plt.scatter(yb, xb, color = "black", s = 5)
                plt.axis("off")
                plt.savefig("Processed_Measurements/segmentation_superpixel/%s_seg3.png" %  (fdisp))
                plt.show()



                #Write the remaining spectra in files and interpolate
                mask_cells = np.zeros((nx,ny))
                X_out_list = []
                X_outr_list = []
                for i in range(num_pix):
                    xav = int(X_superpixel[i,0])
                    yav = int(X_superpixel[i,1])
                    if mask_kcells[xav,yav]:
                        
                        xs, ys = np.where(np.logical_and(segments==i, data["ignore"]==False))
                        if np.size(xs) != 0: #do not include errors
                            mean_spec = np.mean(Measurements[xs,ys,:],axis=(0)) 
                            mean_rspec = np.mean(rawMeas[xs,ys,:],axis=(0)) 
                            fun = interp1d(wn, mean_spec, kind='cubic')
                            funr = interp1d(wn, mean_rspec, kind='cubic')
                            X_out_list.append(fun(w_new))
                            X_outr_list.append(funr(w_new))
                            
                            nucleus_index_list.append(mask_cells[xav,yav]-1)
                            
                            cell_index_list.append(mask_kcells[xav,yav])
                            
                            img_index_list.append(nfile)
                            
                            ypos_list.append([xav,yav])
                            
                            original_filename_list.append(fdisp)
                            
                            if "ftc" in filename or "FTC" in filename:
                                cancer_index_list.append(1)
                            elif "Nthy" in filename or "nthy" in filename:
                                cancer_index_list.append(0)
                            else:
                                cancer_index_list.append(-1)
                                
                            files_found = False
                            for iname in range(len(dates)):
                                if fdisp in dates[iname]:
                                    files_found = True
                                    date_index_list.append(iname)
                            if files_found == False:
                                date_index_list.append(-1)
                                
                                
                X_specs = np.append(X_specs, np.array(X_out_list), axis=0)
                X_raw = np.append(X_raw, np.array(X_outr_list), axis=0)
                
            
            
            
    cell_index = np.array(cell_index_list)  
    cancer_index = np.array(cancer_index_list)
    date_index = np.array(date_index_list)
    nucleus_index = np.array(nucleus_index_list)
    img_index = np.array(img_index_list)
    y_pos = np.array(ypos_list)
    
    
    print("    ", np.size(cell_index), "spectra saved")
        
    data = {"rawdata": X_raw, "data": X_specs, "background": X_back, "rawbackground": X_backr,"wavenumber": w_new, "cell_label": cell_index, "cancer_label": cancer_index, "image_label": img_index, "date_label": date_index, "nucleus_label": nucleus_index, "filename": original_filename_list, "spectra_position": y_pos }
    return data

        

def offset_correction(X):
    k = min(X)
    X2 = X-k
    return X2




def segment_cells(dirname, filename, superpixel_size = 100):
    
            fpath = "%s/%s" % (dirname, filename) 
            fdisp = filename[3:-4]
            data = np.load(fpath)
            
            Measurements = data["data"]
            normMeas = data["normdata"]
            rawMeas = data["rawdata"]
            wn = data["wavenumber"]
           
            Shape = np.shape(Measurements)
            nx = Shape[0]
            ny = Shape[1]       
            

            wb1 = np.argmin(abs(wn-2950))
            wb2 = np.argmin(abs(wn-2990))
            wavenumber_index_0 = np.arange(wb1,wb2)
            Xmean = np.mean(normMeas[:,:,wavenumber_index_0], axis=2)
            vmax = max(Xmean[data["ignore"]==False])           

            #Superpixel Segmentation
            print("\n  Superpixel segmentation...")
            image = np.zeros((nx,ny,3))
            for xi in range(nx):
                for yi in range(ny):
                    image[xi,yi,0] = 0
                    image[xi,yi,1] = Xmean[xi,yi]/vmax
                    image[xi,yi,2] = Xmean[xi,yi]/vmax
             
            numSegments = int(nx*ny/superpixel_size)
            segments = slic(image, n_segments = numSegments, sigma = 5)
             
            """
            # show the output of SLIC
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(image, segments), origin='lower')
            plt.axis("off")
            plt.show()
            """
            
            num_pix = np.max(segments)+1
            MEAN = np.zeros(num_pix)
            for i in range(num_pix):
                xs, ys = np.where(np.logical_and(segments==i, data["ignore"]==False))
                if np.size(xs) == 0:  
                    MEAN[i] = 0
                else:
                    MEAN[i] = np.mean(Xmean[xs,ys])
                
                      
            X_out = np.zeros((nx,ny))
            for xi in range(nx):
                for yi in range(ny):
                    X_out[xi,yi] = MEAN[segments[xi,yi]]
                    
            
            
            """
            plt.imshow(X_out, origin='lower')
            plt.show()
            """
            
            
            
            
            
        
            
            
            
            #Clustering superpixel images
            print("\n  Clustering Nucleus/Cell/background")
            n_cluster = 8
            kmeans = KMeans(n_clusters=n_cluster).fit(X_out.reshape(-1,1))
            kmeans_labels = np.reshape(kmeans.labels_, (-1, ny))
            
            max_cluster = np.max(kmeans.cluster_centers_)
            threshold_cell = sorted(kmeans.cluster_centers_)[int(n_cluster*0.6)]
            threshold_background = sorted(kmeans.cluster_centers_)[int(n_cluster*0.6)-2]
            
            mask_cell = np.zeros((nx,ny))
            mask_background = np.zeros((nx,ny))
            clusters_sp = np.zeros((nx,ny))
                
        
            for xi in range(nx):
                for yi in range(ny):
                    clusters_sp[xi,yi] = kmeans.cluster_centers_[kmeans_labels[xi,yi]]
                    if kmeans.cluster_centers_[kmeans_labels[xi,yi]] == max_cluster:
                        mask_cell[xi,yi] = 2 
                    elif kmeans.cluster_centers_[kmeans_labels[xi,yi]] >= threshold_cell:
                        mask_cell[xi,yi] = 1 
                    elif kmeans.cluster_centers_[kmeans_labels[xi,yi]] < threshold_background:    
                        mask_background[xi,yi] = True  
                        
            plt.imshow(clusters_sp, origin='lower')
            plt.savefig("Processed_Measurements/segmentation_superpixel/%s_clust.png" %  (fdisp))
            plt.show()
            save_image(clusters_sp, "Processed_Measurements/segmentation_superpixel/%s_seg3.png" %  (fdisp))
            
            plt.imshow(mask_cell, origin='lower')
            plt.savefig("Processed_Measurements/segmentation_superpixel/%s_nuc.png" %  (fdisp))
            plt.show()
            


    
            
            #Segment the cells with hierachical clustering algorithm
            print("  Segmenting cells")
            
            #Create a smaller dataset with the superpixels
            X_superpixel = np.zeros((num_pix,4))
            for i in range(num_pix):
                xs, ys = np.where(segments==i)
                xav = int(np.mean(xs))
                yav = int(np.mean(ys))
                X_superpixel[i,0] = xav
                X_superpixel[i,1] = yav
                X_superpixel[i,2] = mask_cell[xav,yav]
                X_superpixel[i,3] = i
                    
                    
            max_d_min = int(np.sqrt((nx*ny)/numSegments)) + 3
            X_list = []
            pix_index = []
            for i in range(num_pix):
                if X_superpixel[i,2]:
                    X_list.append([X_superpixel[i,0],X_superpixel[i,1]])
                    pix_index.append(X_superpixel[i,3])
            X = np.array(X_list)     
            Z = linkage(X, 'single')
            clusters = fcluster(Z, max_d_min, criterion='distance')
            n_clust = int(10*np.max(clusters))
            cm = plt.get_cmap('tab10')
            colors10 = cm.colors
            #colors10 = np.concatenate((cm.colors,cm.colors),axis = 0) #in case more than 10 clusters
            set_link_color_palette([matplotlib.colors.rgb2hex(rgb[:3]) for rgb in colors10])

            
            
            mask_kcells = np.zeros((nx,ny))
            for i in range(len(X_list)):
                xs, ys = np.where(segments==pix_index[i])
                mask_kcells[xs,ys] = clusters[i]
            
            plt.imshow(np.zeros((nx,ny)), origin='lower')
            xs, ys = np.where(mask_kcells==0)
            plt.scatter(ys, xs, label="background", s=0.3, c = "black")
            for i in range(1,n_clust+1):
                xs, ys = np.where(mask_kcells==i)
                plt.scatter(ys, xs, label="cell %s" % i, s=0.3, c = colors10[(i-1)%10])
            #ys, xs = np.where(data["ignore"])
            #plt.scatter(xs, ys, s=0.7, c="r", label="cosmicray")
            #plt.legend(loc="lower left",markerscale=5)
            plt.title("%s: Cell clustering" % (fdisp))
            plt.axis('scaled')
            plt.savefig("Processed_Measurements/segmentation_superpixel/%s_seg1.png" %  (fdisp))
            plt.show()                      
            
                     
            
            n_spec_min = int(numSegments*2/100)
                        
            #split each cell individually with hierachical clustering ward (small clusters should ne be affected)
            if superpixel_size == 10:
                max_d_ward = 1700
            elif superpixel_size == 100:
                max_d_ward = 630
            else:
                print("  Please specify a ward distance for HAC")
                sys.exit(0)
                
                
            for ci in range(1,n_clust+1):
                X_list_i = []
                pix_index_i = []
                XX_link = []
                for j in range(len(X_list)):
                    if clusters[j]==ci:
                        X_list_i.append([X_list[j][0],X_list[j][1]])
                        pix_index_i.append(pix_index[j])
                        XX_link.append(j)
                if len(X_list_i) >= n_spec_min:
                    X_i = np.array(X_list_i)
                    Z = linkage(X_i, 'ward')
                    clusters_i = fcluster(Z, max_d_ward, criterion='distance')
                    n_clust_i = int(np.max(clusters_i))
                    
                    if n_clust_i > 1:
                        
                        plt.figure(figsize=(10, 7))  
                        fancy_dendrogram(Z, p=840, truncate_mode='lastp', annotate_above=1000000, no_labels=True, max_d=max_d_ward)
                        plt.show()
                        
                        
                        for j in range(len(X_list_i)):
                            clusters[XX_link[j]] = ci + 10*clusters_i[j]

                                    
                        
                    
                    
            #remove the small cells  
            for i in range(1,n_clust+1):
                ni = np.where(clusters==i)
                if np.size(ni) < n_spec_min:  #if the cluster is too small to be a cell
                    clusters[ni]=0
                             
        
                    
            
            
            #change the cluster labeling to successive numbers
            le = preprocessing.LabelEncoder()
            le.fit(clusters)  
            if 0 in clusters:
                clusters = le.transform(clusters)
            else:
                clusters = le.transform(clusters)+1 
            
            mask_kcells = np.zeros((nx,ny))
            for i in range(len(X_list)):
                xs, ys = np.where(segments==pix_index[i])
                mask_kcells[xs,ys] = clusters[i]
            
            if not os.path.exists("Processed_Measurements/Mask_cells"):
                os.makedirs("Processed_Measurements/Mask_cells")
            scipy.io.savemat('Processed_Measurements/Mask_cells/%s.mat' % fdisp ,{'mask_cells':mask_kcells}) 
            
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
        




    
if __name__ == "__main__":
    main()
