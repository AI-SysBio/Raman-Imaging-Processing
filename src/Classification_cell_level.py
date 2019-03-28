
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

from scipy.spatial.distance import pdist,squareform
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from colorama import Fore
from colorama import Style
from pandas import DataFrame

from libPostprocessing import *


"""

   This code takes the postprocessed average cell spectra and perform classification with various classifier 
   by excluding the measurements taken at the same date to check for the consistancy of the measurements:

               data: n cell spectra each with f wavenumbers (n*f 2d array), after area normalized
         wavenumber: 1d array f, corresponding to the wavenumbers values
        image_label: 1d array n, file label (different for each image)
         date_label: 1d array n, integer from 0 to ndates
       cancer_label: 1d array n, 0 if from NT images and 1 if from FTC images
         line_label: 1d array n, integer corresponding from the cell line
           filename: 1d array n, original filename of each image
           
"""





def classify_cell_level(data_av,labels, plot = False):
     
    
    X_av = data_av["data"]
    wn = data_av["wavenumber"]
    yi = data_av["date_label"]
    yc = data_av["cancer_label"] 
    yl = data_av["line_label"]
    
    print("    ", np.size(yc), "cells detected")

    plot_class = False

    
    #Perform PCA
    print("\n  Computing PCA...")
    pca = PCA(n_components = 5)
    X = pca.fit_transform(X_av) 
    #plt.plot(pca.components_ [0])
    #plt.show() 

    c1,acc1 = test_classifier(X,yi,yc,yl,plot_class, "svm",labels)
    c2,acc2 = test_classifier(X,yi,yc,yl,plot_class, "knn",labels)
    c3,acc3 = test_classifier(X,yi,yc,yl,plot_class, "lda",labels)
    
    
    color = ["red","green"]
    n_cell = np.size(c1)
    colors_p1 = []; colors_p2 = []; colors_p3 = []
    for i in range(n_cell):
        colors_p1.append(color[c1[i]])
        colors_p2.append(color[c2[i]])
        colors_p3.append(color[c3[i]])
        
    color2 = ["blue", "red", "gold", "darkviolet", "cadetblue", "navy", "olive", "darkcyan", "darkorange", "brown", "coral", "chartreuse", "orange", "crimson", "chocolate", "maroon","dodgerblue","goldenrod", "darkred","darkblue"] 
    colors_c = []; colors_d = []
    for i in range(n_cell):
        colors_c.append(color2[int(yc[i])])
        colors_d.append(color2[int(yi[i]+2)])
        
        
    print("\n\n  Classification results:")
    print("    line 1 = date ")
    print("    line 2 = maligancy")
    print("    line 3 = cell line")
    print("    line 4 (SVM acc) = %.2f" %acc1 )
    print("    line 5 (LDA acc) = %.2f" %acc2 )
    print("    line 6 (KNN acc) = %.2f" %acc3 ) 
        
    plt.figure(figsize=(8.5,1))
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,0.5),c=colors_p1,s = 7)
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,0),c=colors_p2,s = 7)
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,-0.5),c=colors_p3,s = 7)
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,1.5),c=colors_c,s = 7)
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,2),c=colors_d,s = 7)
    plt.axis("off")
    plt.ylim([-0.7,2.2])
    plt.show()
    
    
    
    #distance matrix by FTC   
    plt.rcParams.update({'font.size': 12})
    nnthy = np.where(yc == 0)
    nftc = np.where(yc == 1)
    
    X_mal = np.concatenate((X[nnthy],X[nftc]), axis = 0)
    yi_mal = np.concatenate((yi[nnthy],yi[nftc]), axis = 0)
    yc_mal = np.concatenate((yl[nnthy],yl[nftc]), axis = 0)

    plt.figure(figsize=(10,10))
    distance = "cityblock"
    D = squareform(pdist(X_mal, distance))
    plt.imshow(D)
    plt.title("Distance matrix of average cell spectra (NT -> FTC, L1 norm)")
    plt.colorbar()
    plt.axis("off")
    plt.show() 
    
    colors_d = []; colors_p = []; colors_l = [] 
    
    for i in range(n_cell):
        colors_d.append(color2[int(yi_mal[i]+2)])
        colors_l.append(color2[int(yc_mal[i])])
        
    plt.figure(figsize=(8.5,0.5))
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,3),c=colors_l,s = 5)
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,2.5),c=colors_p,s = 5)
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,2),c=colors_d,s = 5)
    plt.axis("off")
    plt.ylim([1.3,3.2])
    plt.show()
    
    
    
    
def plot_scatter(X,w,wn,yc,yi,variable)    :
    ncan = np.where(yc == 1)
    nnc = np.where(yc == 0)
    
    njan = np.where(yi == 0)
    nmarch = np.where(yi == 1)
    njune = np.where(yi == 2)
    njuly = np.where(yi == 3)
    
    plt.scatter(X[ncan[0],w[0]],X[ncan[0],w[1]], color = "red", label = "FTC")
    plt.scatter(X[nnc[0],w[0]],X[nnc[0],w[1]], color = "blue", label = "Nthy")
    plt.xlabel("%s (w = %.0f cm-1)" % (variable[0],wn[w[0]]))
    plt.ylabel("%s (w = %.0f cm-1)" % (variable[1],wn[w[1]]))
    plt.title("Best two wavenumber for classification with %s: %.0f and %.0f" % (variable[2],wn[w[0]],wn[w[1]]))
    plt.legend()
    plt.show()
    
    plt.scatter(X[njan[0],w[0]],X[njan[0],w[1]], color = "blue", label = "Jan")
    plt.scatter(X[nmarch[0],w[0]],X[nmarch[0],w[1]], color = "red", label = "March")
    plt.scatter(X[njune[0],w[0]],X[njune[0],w[1]], color = "green", label = "June")
    plt.scatter(X[njuly[0],w[0]],X[njuly[0],w[1]], color = "gold", label = "July")
    plt.xlabel("%s (w = %.0f cm-1)" % (variable[0],wn[w[0]]))
    plt.ylabel("%s (w = %.0f cm-1)" % (variable[1],wn[w[1]]))
    plt.title("Best two wavenumber for classification with %s: %.0f and %.0f" % (variable[2],wn[w[0]],wn[w[1]]))
    plt.legend()
    plt.show()


    
  
def test_classifier(X,yi,yc,yl,plot,classifier,labels):
    
    print("\n\n  Predictions on testset with %s: " % classifier)
    acc_ = []
    n_cell_ = []
    y_score = np.array([])
    y_true = np.array([]).astype(int)
    y_date  = np.array([])
    ndates = int(np.max(yi)+1)
    
    
    for j in range(ndates):
        acc0,n_cell0,pred0,true0 = test_date(j,X,yi,yc, plot, classifier)
        acc_.append(acc0)
        n_cell_.append(n_cell0)
        y_score = np.concatenate((y_score,pred0)) 
        y_true = np.concatenate((y_true,true0)) 
        y_date = np.concatenate((y_date,np.full(n_cell0,j)))
    
    y_pred = y_score
    n_cell = sum(np.array(n_cell_))
    acc = np.dot(np.array(n_cell_),np.array(acc_))
    acc /= n_cell
    
    if np.max(y_true) > 1:
        AUC = 0
        Fscore = 0
    else:
        AUC = roc_auc_score(y_true, y_score)
        Fscore = f1_score(y_true, y_pred)
        
    C = confusion_matrix(y_true, y_pred)
     
    for j in range(ndates):    
        print("    %s:  %s/%s = %.3f" % (labels[j],int(round(acc_[j]*n_cell_[j],0)), n_cell_[j], acc_[j]))
    print(f"\n  {Style.BRIGHT}Total:{Style.RESET_ALL}")
    print(f"    {Style.BRIGHT}Acc =  %s/%s = %.3f{Style.RESET_ALL}" % (int(round(acc*n_cell,0)), n_cell, acc))
    print(f"    {Style.BRIGHT}F1score = %s{Style.RESET_ALL}" % round(Fscore,2))
    print(f"    {Style.BRIGHT}AUC = %s{Style.RESET_ALL}" % round(AUC,2))
        
    nthy = np.size(np.where(yc == 0))
    nftc = np.size(np.where(yc == 1))
    nutc = np.size(np.where(yc == 2))
    print("\n  Confusion Matrix: ")
    print("  (%s) (%s) (%s)\n" % (nthy,nftc,nutc))
    print(DataFrame(C)) 
    
    
    
    label_line = ["Nthy","FTC133","RO82W"]
    ncells = np.zeros(len(label_line))
    acc_line = np.zeros(len(label_line))
    for i in range(len(label_line)):
        nwhere_ = np.where(yl == i)
        if np.size(nwhere_) != 0:
            ncells[i] = np.size(nwhere_)
            acc_line[i] = np.count_nonzero((y_true[nwhere_]==y_pred[nwhere_]))/ncells[i]
    
    
    
    print("\n  Classification per cell lines: ")
    for i in range(len(label_line)):
        print("    %s:  %s/%s = %.3f" % (label_line[i],int(round(acc_line[i]*ncells[i],0)), int(ncells[i]), acc_line[i])) 
    
    
    print(" ----------------------------------------------------------------------------------------------------------- ")
    
    correct = (y_true==y_pred)
    
    return correct, acc

    
    
def test_date(date_index,X,yi,yc,plot, classifier):
    
    test_condition =   (yi == date_index)
    n_train = np.where(np.logical_not(test_condition))
    n_test = np.where(test_condition)
    
    if np.size(n_test) == 0:
        return 0.5,0,np.array([]),np.array([])
    else:
        if classifier in {"knn","KNN","kNN"}:
            acc,n_cell,pred_index,true_index = classify_knn(X,n_test,n_train,yc, plot)
        elif classifier in {"lda","LDA"}:
            acc,n_cell,pred_index,true_index = classify_LDA(X,n_test,n_train,yc, plot)
        elif classifier in {"svm","SVM"}:
            acc,n_cell,pred_index,true_index = classify_svm(X,n_test,n_train,yc, plot)
        return acc,n_cell,pred_index,true_index
    
    
    
def classify_knn(X,n_test,n_train,yc, plot):  
      
    
    # count the number of FTC and Nthy spectra in the training set
    n_cell = np.size(n_test)
    X_test = X[n_test,:].reshape(np.size(n_test),X.shape[1])
    X_train = X[n_train,:].reshape(np.size(n_train),X.shape[1])
    yc_train = yc[n_train]
    yc_test = yc[n_test]
    NTHY = np.size(np.where(yc_train == 0))
    NFTC = np.size(np.where(yc_train == 1))
    ratio = NTHY/NFTC

        
    #train a kNN classifier for nearest neighboor classification:
    n_neighbors = 5
    kNN = KNeighborsClassifier(n_neighbors=n_neighbors)
    kNN.fit(X_train, yc_train)
    y_pred_train = kNN.predict(X_train)
    acc = accuracy_score(yc_train,y_pred_train)
    print("   kNN training accuracy is", acc)
    
    
    #test: classify each spectra of a test cell with kNN, account for the ratio correction
    y_nc = kNN.predict_proba(X_test)[:,0]*n_neighbors
    y_can = kNN.predict_proba(X_test)[:,1]*n_neighbors
    y_pred_proba = y_can*ratio/(y_can*ratio+y_nc)
    y_pred = y_pred_proba > 0.5
    

    if plot:     
        print("  Prediction accuracy with %s cells is %s/%s = %.3f \n" % (n_cell, int(round(acc*n_cell,0)), n_cell, acc))
    
    return acc,n_cell,y_pred,yc_test






def classify_svm(X,n_test,n_train,yc, plot):  
      
    
    # count the number of FTC and Nthy spectra in the training set
    n_cell = np.size(n_test)
    X_test = X[n_test,:].reshape(np.size(n_test),X.shape[1])
    X_train = X[n_train,:].reshape(np.size(n_train),X.shape[1])
    yc_train = yc[n_train]
    yc_test = yc[n_test]
    NTHY = np.size(np.where(yc_train == 0))
    NFTC = np.size(np.where(yc_train == 1))
    ratio = NTHY/NFTC
    
    if plot:
        print("\n  Training set:")
        print("   ", NTHY, "Nthy spectra")
        print("   ", NFTC, "FTC spectra")
        print("    Ratio FTC/Nthy = 1.00:%.2f" % (ratio))
        
        
    class_weight = "balanced"

    SVM = svm.SVC(class_weight = class_weight, kernel = 'linear')
    SVM.fit(X_train, yc_train)
    y_pred_train = SVM.predict(X_train)
    acc = accuracy_score(yc_train,y_pred_train)
    print("   SVM training accuracy is", acc)
    
    y_pred = SVM.predict(X_test)
    
    acc = accuracy_score(yc_test, y_pred)

    if plot:     
        print("  Prediction accuracy with %s cells is %s/%s = %.3f \n" % (n_cell, int(round(acc*n_cell,0)), n_cell, acc))
    
    return acc,n_cell,y_pred,yc_test











def classify_LDA(X,n_test,n_train,yc, plot): 
    
    n_cell = np.size(n_test)
    X_test = X[n_test,:].reshape(np.size(n_test),X.shape[1])
    X_train = X[n_train,:].reshape(np.size(n_train),X.shape[1])
    yc_train = yc[n_train]
    yc_test = yc[n_test]
    NTHY = np.size(np.where(yc_train == 0))
    NFTC = np.size(np.where(yc_train == 1))
    ratio = NTHY/NFTC
    
    if plot:
        print("\n  Training set:")
        print("   ", NTHY, "Nthy spectra")
        print("   ", NFTC, "FTC spectra")
        print("    Ratio FTC/Nthy = 1.00:%.2f" % (ratio))
      


    mda = LinearDiscriminantAnalysis()     
    mda.fit(X_train, yc_train)
    X_reduced_train = mda.transform(X_train)
    y_pred_train = X_reduced_train[:,0]>0
    acc = accuracy_score(yc_train,y_pred_train)
    print("   LDA training accuracy is", acc)
    
    X_reduced_test = mda.transform(X_test)
    y_pred_test = X_reduced_test[:,0]>0
    acc = accuracy_score(yc_test,y_pred_test)    
    
    if plot:     
        print("  Prediction accuracy with %s cells is %s/%s = %.3f \n" % (n_cell, int(round(acc*n_cell,0)), n_cell, acc)) 
        
    return acc,n_cell,y_pred_test,yc_test    






    


    
