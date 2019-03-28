# Measurements-Informatics-on-Hyperspectral-Raman-Images

This repository contains the preprocessing and postprocessing tools to analyse hyperspectral Raman images:

       
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
    
    4) & 5) Classification with average cell spectra and subcellular spectral clustering        
            (each date is taken as a test set and is predicted with a classifier trained on other dates)
            (=> requires at least 2 different dates to work)
            (=> requires at least 1FTC image and 1NT image to work)
    
        (see Classification_cell_level.py and Classification_clusters,py for details)
