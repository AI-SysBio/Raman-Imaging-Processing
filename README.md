# Hyperspectral Raman Image Processing

<img align="right" src="https://raw.githubusercontent.com/Aurelien-Pelissier/Raman-Imaging-Processing/master/img/Hyperspectral-Image.png" width=400>
Raman microscopy is a very promising technique to increase reliability in clinical diagnosis as it provides information on molecular vibrations and crystal structures, while being non-destructive and requiring minimal sample preparation. It allows high spatial resolution and can provide detailed images of individual cells, for which the extraction of chemical and spatial information of their sub-cellular components have the potential to give a more complete understanding of the underlying biological processes as well as better accuracies in clinical diagnosis. A Raman hyperspectral image consists of a three dimensional matrix, the first two axis corresponding to the pixel position and the third axis being the Raman intensity spectrum at that pixel. One image contains roughly 80000 spectra with 1000 wavenumbers each, in the range of 700 - 3000 cm-1. This repository contains the tool to process Hyperspectra Raman images:
       
    Process the hyperspectral Raman image with:  
          - Cosmic ray detection
          - Correction for Irradiance profile
          - Identification and subtraction of Background spectrum
          - Singular Value Decomposition Denoising
          - Polyfit fluorescence correction
          - Spectra area normalization
        
        
## References

A. Pelissier, K. Hashimoto, K. Mochizuki, Y. Kumamoto, J. N. Taylor, A. Nakamura, Y. Harada, K. Fujita, T. Komatsuzaki. (2019). Measuremental Informaticson Single cell Raman images to diagnose follicular thyroid carcinoma [https://arxiv.org/abs/1904.05675]
