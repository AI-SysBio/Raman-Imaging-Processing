""" 
    Raman image preprocessing with a background subtraction method from:
        
    [ref] A. Pelissier, K. Hashimoto, K. Mochizuki, Y. Kumamoto, J. N. Taylor, K. Tabata, 
          JE. Clement, A. Nakamura,  Y. Harada, K. Fujita and T. Komatsuzaki. 
          Intelligent Measurement Analysis on Single Cell Raman Images for the Diagnosis 
          of Follicular Thyroid Carcinoma. arXiv preprint. (2019).
          

    -> the processing of the Raw hyperspectral Raman image typically takes 2min per image
       The files are being saved through the process so that they are loaded on the next launch
    
    
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
        


from Process_all import process_all
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes



def main():
    
    
    #1 - Raman data preprocessing ---------------------------------------------------------------------------------------
    Folder = "./Raw_Measurements"

    print("\n") 
    print("\n") 
    print("======================================================================================") 
    print("================================ Raman data Processing ===============================") 
    print("======================================================================================\n") 
      
   
    superpixel_size = 100
    cell_cluster_factor = 0.5
    data, data_av = process_all(Folder, superpixel_size, cell_cluster_factor)

        
    ndates = 0
    for i in range(np.max(data["date_label"] + 1)):
        if np.size(np.where(data["date_label"] == i)) > 0:
            ndates += 1
        
    print("     %s images" % np.max(data["image_label"]))
    print("     %s dates" % ndates)
    plot_spectra(data, norm = True)
    
   
    

def plot_spectra(data):
    
    X = data["data"]
    yc = data["cancer_label"]
    wn = data["wavenumber"]
    
    #plot with broken axis
    nnc = np.where(yc == 0)
    ncan = np.where(yc == 1) 
    Mean1 = np.mean(X[nnc],axis=0) 
    Mean2 = np.mean(X[ncan],axis=0)


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
    bax.plot(x, Mean1, color = "blue", linewidth = 2, label = "NTHY")
    bax.plot(x, Mean2, color = "red", linewidth = 2, label = "FTC")
    bax.fill_between(wn, lower_CI, upper_CI, color = '#539caf', alpha = 0.4)
    bax.legend(loc="upper left")
    bax.set_ylabel('Intensity [au]')  
    bax.set_xlabel('Wavenumber [cm-1]')
    bax.tick_params(axis='both', labelleft = False, labelbottom = False, bottom = False)
    
    bax = brokenaxes(xlims=((min(wn1),max(wn1)), (min(wn2)-20,max(wn2))), hspace=0.15, subplot_spec=sps2)
    bax.axhline(y=0, color='black', linestyle='--',alpha=0.3)
    bax.plot(x, Mean1-Mean2, color = "red", linewidth = 2)
    bax.fill_between(wn, lower_CI-np.mean(X,axis=0), upper_CI-np.mean(X,axis=0), color = '#539caf', alpha = 0.4)
    bax.set_ylabel('$\\Delta$ Intensity [au]') 
    bax.set_ylim(-1.5,1.5)
    bax.tick_params(axis='y', labelleft = False)
    
    plt.show()



if __name__ == "__main__":
    plt.rcParams.update({'font.size': 14})
    main()
    