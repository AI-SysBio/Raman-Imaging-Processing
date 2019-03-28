# -*- coding: utf-8 -*-
"""
       
    1) Preprocessing of the Raw hyperspectral Raman image
        Typically takes 15min per image, mainly due to the baseline correction
        (see Preprocessing.py for details)
    
    2) Superpixel spectra extraction with cell segmentation
        Typically takes 1min per image
           manually segmented files can be used if use_manual_mask is set to True
           -> mask should be put in the folder Processed_Measurements/Mask_cells_manual
        (see Superpixels.py for details)
        -> The dates corresponding to each file names has to be written in Superpixels.py
    
    3) Background subtraction and further postprocessing
        process all the superpixel spectra with background subtraction and other methods
        (see Postprocessing.py for details)
    
    4) & 5) Classification with subcellular voting and spectral clustering        
            (each date is taken as a test set and is predicted with a classifier trained on other dates)
            (=> requires at least 2 different dates to work)
            (=> requires at least 1FTC image and 1NT image to work)
    
        (see Classification_voting.py and Classification_clusters,py for details)
    
    The files are being saved through the process so that they are loaded on the next launch
    if you change some processing parameters, you can put erase_ to True so the files are recomputed on the next run
        
"""

from Preprocessing import preprocess
from Superpixels import superpixel
from Postprocessing import process_spectras, plot_spectra, plot_pca
from Classification_cell_level import classify_cell_level
from Classification_clusters import RDT_subclust, classify_subclust, plot_clusters

import os
import os.path
import numpy as np
import matplotlib.pyplot as plt




def set_dates():
    
    labels = ["Date1", "Date2"] 
    
    dates = []
    #Date1
    dates.append(["ftc01","nthy01"])
    #Date2
    dates.append(["ftc02","nthy02"])

    return dates,labels



def main():
    
    erase_preprocessed = False
    erase_superpixel = False
    erase_postprocessed = False
    erase_cell_population = False
    use_manual_mask = False
    
    dates,labels = set_dates()
    
    
    

    
    
    
    #1 - Raman data preprocessing ---------------------------------------------------------------------------------------
    Folders = ["./Raw_Measurements"]
    print("\n") 
    print("\n") 
    print("=============================================================================================================") 
    print("========================================= Raman data Preprocessing ==========================================") 
    print("=============================================================================================================\n") 
      
   

    if not os.path.exists("Processed_Measurements"):
        os.makedirs("Processed_Measurements")
        
    for dirname in Folders:
        for fname in os.listdir(dirname):
            
            fpath = "%s/%s" % (dirname, fname)
            
            if "ftc" in fpath or "nthy" in fpath or "Nthy" in fpath or "FTC" in fpath:
                    
                base, ext = os.path.splitext(fname)
                preprofile = "Processed_Measurements\pp_%s.npz" % (base)
                if not os.path.isfile(preprofile) or erase_preprocessed:  
                    print("\n   Starting Raman Data Processing of %s..." % base)
                    data = preprocess(fpath, fname)
                    np.savez(preprofile, **data)
                else:
                    print("   Detecting %s..." % base)
                    
              
                
                 
                    
                    
    #2 - Raman data superpixel preprocessing  ---------------------------------------------------------------------------     
    print("\n\n") 
    print("=============================================================================================================") 
    print("======================================= Superpixel Cell segmentation ========================================") 
    print("=============================================================================================================")                 

    if not os.path.exists("Spectra_Analysis"):
        os.makedirs("Spectra_Analysis")    

    seg = ["auto","manual"]
    dirname = "./Processed_Measurements/"
    pixel_size = 10
    superpixel_file = "Spectra_Analysis/Superpixel_spectras_%s_%s.npz" % (seg[use_manual_mask],pixel_size)
    if not os.path.isfile(superpixel_file) or erase_superpixel:
        spectra_data = superpixel(dirname,use_manual_mask,pixel_size,dates)
        np.savez(superpixel_file, **spectra_data)
    else:
        print("\n   Using %s ..." % superpixel_file[17:])
    
    if use_manual_mask:
        print("   Manual segmentation used")
    else:
        print("     (Automated segmentation used)")
        
    
    
    
    
    
    
    #3 - Raman data analysis --------------------------------------------------------------------------------------------   
    print("\n") 
    print("\n") 
    print("=============================================================================================================") 
    print("============================================== Postprocessing ===============================================") 
    print("=============================================================================================================") 
    

    #base_correc = "offset"
    base_correc = "poly"
    #base_correc = "als"
    #base_correc = "poly_only"
    postpro_file = "Spectra_Analysis/postprocessed_spectras_%s_%s.npz" % (pixel_size,base_correc)
    postpro_file_av = "Spectra_Analysis/postprocessed_spectras_av_%s_%s.npz" % (pixel_size,base_correc)
    if not os.path.isfile(postpro_file) or erase_postprocessed or erase_superpixel:
        print("\n")
        data, data_av = process_spectras(superpixel_file, base_correc)
        np.savez(postpro_file, **data)
        np.savez(postpro_file_av, **data_av)
        
    else:
        print("\n   Using %s..." % postpro_file[17:])
        data = np.load(postpro_file)
        data_av = np.load(postpro_file_av)
        
    ndates = 0
    for i in range(np.max(data["date_label"] + 1)):
        if np.size(np.where(data["date_label"] == i)) > 0:
            ndates += 1
        
    print("     %s images" % np.max(data["image_label"]))
    print("     %s dates" % ndates)
        
    plt.rcParams.update({'font.size': 18})
    plot_spectra(data, norm = True)
    plot_pca(data, alpha = 0.05)
        
      


    #4 - classification at the cell level -------------------------------------------------------------------------------------
    print("\n") 
    print("\n") 
    print("=============================================================================================================") 
    print("=============================== Classification at the cell level on PC space ================================") 
    print("=============================================================================================================")        
        
    classify_cell_level(data_av,labels, plot = True)
        
    
    
    
    
    
    
    #5 - classification with subcellular class population ---------------------------------------------------------------
    print("\n") 
    print("\n") 
    print("=============================================================================================================") 
    print("================================ Classification with spectral subclusters ===================================") 
    print("=============================================================================================================") 
    
    if not os.path.exists("Sub_cellular_study"):
        os.makedirs("Sub_cellular_study") 
    
    n_clusters = 5
    subcell_file = "Sub_cellular_study/Cell_polulation_%s.npz" % n_clusters
    
    if not os.path.isfile(subcell_file) or erase_cell_population or erase_postprocessed or erase_superpixel:
        print("\n")
        data = RDT_subclust(postpro_file,n_clusters)
        np.savez(subcell_file, **data)
    
    else:
        print("\n   Using %s..." % subcell_file[19:])
        data = np.load(subcell_file)
    
    classify_subclust(data,labels, plot = False)
    plot_clusters(data)
    
    
    
    


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 14})
    main()
    