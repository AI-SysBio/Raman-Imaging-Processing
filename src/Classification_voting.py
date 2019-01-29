
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix

from colorama import Fore
from colorama import Style
from pandas import DataFrame

from libPostprocessing import *



    




def classify_voting(data,labels, plot = False):
     
    
    X = data["data"]
    wn = data["wavenumber"]
    yi = data["date_label"]
    yc = data["cancer_label"]
    ycell = data["cell_label"]    
    yim = data["image_label"] 
    yfile = data["filename"] 
    ypos = data["spectra_position"]
    
    plot_PCA = False
    plot_class = False
    wFUSE = [2827,2852,1584]
    weighted = True #weighted covariate correction
    
    #Perform PCA
    print("\n  Computing PCA...")
    pca = perform_PCA(X,yc,yi,plot_PCA)
    #plt.plot(pca.components_ [0])
    #plt.show()
    
    
    
    #test the performances 
    c1,_,_,acc1 = test_classifier(X,pca,yi,yc,ycell,yim,yfile,ypos,wFUSE,wn, weighted, "knn", plot_class, labels, plot_final = plot)    
    #c2,_,_,acc2 = test_classifier(X,pca,yi,yc,ycell,yim,yfile,ypos,wFUSE,wn, weighted, "svm", plot_class, labels)  #removed because training too long
    c3,_,_,acc3 = test_classifier(X,pca,yi,yc,ycell,yim,yfile,ypos,wFUSE,wn, weighted, "lda", plot_class, labels, plot_final = plot)
    c4,c,d,acc4 = test_classifier(X,pca,yi,yc,ycell,yim,yfile,ypos,wFUSE,wn, weighted, "wn", plot_class, labels, plot_final = plot)
    
    color2 = ["blue", "red", "gold", "darkviolet", "cadetblue", "navy", "olive", "darkcyan", "darkorange", "brown", "coral", "chartreuse", "orange", "crimson", "chocolate", "maroon","dodgerblue","goldenrod", "darkred","darkblue"] 
    color = ["red","green",""]
    n_cell = np.size(c1)
    colors_c = []; colors_d = []
    colors_p1 = []; colors_p3 = []; colors_p4 = []
    date_c = []
    for i in range(n_cell):
        colors_p1.append(color[c1[i]])
        colors_p3.append(color[c3[i]])
        colors_p4.append(color[c4[i]])
        colors_c.append(color2[c[i]])
        colors_d.append(color2[d[i]+2])
        if labels[c[i]] not in date_c:
            date_c.append(labels[c[i]])
        
    print("\n\n  Classification results:")
    print("    line 1 = date [", end='')
    for i in range(len(date_c)-1):
        print("%s, " % date_c[i] , end='')
    print("%s]"% date_c[-1])
    print("    line 2 = FTC/NT (red/blue)")
    print("    line 3 (kNN acc) = %.2f" %acc1 )
    print("    line 4 (LDA acc) = %.2f" %acc3 )
    print("    line 5 (wn  acc) = %.2f" %acc4)
    
    plt.figure(figsize=(13,1.5))
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,2),c=colors_d,s = 10)
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,1.5),c=colors_c,s = 10)    
    
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,0.5),c=colors_p1,s = 10)
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,0),c=colors_p3,s = 10)
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,-0.5),c=colors_p4,s = 10)
    plt.axis("off")
    #plt.tick_params(axis ="both", bottom = False, labelbottom=False, left = False, labelleft=False)
    plt.ylim([-0.7,2.2])
    plt.show()
    
    
    
    
def test_classifier(X,pca,yi,yc,ycell,yim,yfile,ypos,wFUSE,wn, weighted, classification, plot_class, labels, plot_final = False, imshow_results = False):  
    
    print("\n\n  Predictions on testset with %s: " % classification)
    if classification == "wn":
        print("    ",wFUSE, "cm-1")
    nPC = np.arange(10)  
    acc_ = []
    n_cell_ = []
    y_score = np.array([])
    y_true = np.array([])
    y_date  = np.array([])
    ndates = int(np.max(yi)+1)
    
    
    for j in range(ndates):
        acc0,n_cell0,pred0,true0 = test_date(j,X,pca,nPC,yi,yc,ycell,yim,wFUSE,wn, weighted, classification, plot_class)
        acc_.append(acc0)
        n_cell_.append(n_cell0)
        y_score = np.concatenate((y_score,pred0)) 
        y_true = np.concatenate((y_true,true0)) 
        y_date = np.concatenate((y_date,np.full(n_cell0,j)))
    
    y_pred = (y_score >= 0.5) 
    n_cell = sum(np.array(n_cell_))
    acc = np.dot(np.array(n_cell_),np.array(acc_))
    acc /= n_cell
    
    AUC = roc_auc_score(y_true, y_score)
    Fscore = f1_score(y_true, y_pred)
    C = confusion_matrix(y_true, y_pred)
     
    if plot_final:
        for j in range(ndates):    
            print("    %s:  %s/%s = %.3f" % (labels[j],int(round(acc_[j]*n_cell_[j],0)), n_cell_[j], acc_[j]))
        print(f"\n  {Style.BRIGHT}Total:{Style.RESET_ALL}")
        print(f"    {Style.BRIGHT}Acc =  %s/%s = %.3f{Style.RESET_ALL}" % (int(round(acc*n_cell,0)), n_cell, acc))
        print(f"    {Style.BRIGHT}F1score = %s{Style.RESET_ALL}" % round(Fscore,2))
        print(f"    {Style.BRIGHT}AUC = %s{Style.RESET_ALL}" % round(AUC,2))
        
        print("\n  Confusion Matrix: ")
        print(DataFrame(C)) 
    
    correct = (y_true==y_pred)
    
    
    
    
    #plot classification result
    imshow_results = False
    if imshow_results:    
        if not os.path.exists("Spectra_Analysis/Classified_cells"):
            os.makedirs("Spectra_Analysis/Classified_cells")  
        i=0
        nimg = np.max(yim)+1
        color = ["red","green"]
        back = np.zeros((250,400))
        for img_index in range(0,nimg):
            n_img = np.where(yim == img_index)
            if np.size(n_img) > 0:
                filename = yfile[n_img][0]
                plt.imshow(back, origin='lower')
                xs, ys = np.where(back==0)
                plt.scatter(ys, xs, label="background", s=0.3, c = "black")
                        
                ncells = int(np.max(ycell[n_img]))+1
                for cell_index in range(0,ncells):
                    n_sample = np.where(np.logical_and(ycell == cell_index, yim == img_index))
                    if np.size(n_sample) > 0:
                        pos_cell = ypos[n_sample[0]]
                        plt.scatter(pos_cell[:,1],pos_cell[:,0], c = color[correct[i]], cmap='RdYlGn', s = 15)
                        i+=1
                                        
                plt.title("Classification results for image %s" % filename)
                plt.savefig("Classified_cells/%s_class.png" %  (filename))
                plt.show()
    
    
    return correct,y_true.astype(int),y_date.astype(int),acc
    
    
    

def test_date(date_index,X,pca,nPC,yi,yc,ycell,yim,wFUSE,wn, weighted,classification, plot):
    
    test_condition =   (yi == date_index)
    n_train = np.where(np.logical_not(test_condition))
    n_test = np.where(test_condition)
    
    Xpc = pca.transform(X)
    
    if np.size(n_test) == 0:
        return 0.5,0,np.array([]),np.array([])
    else:
        if classification in {"knn","KNN","kNN"}:
            acc,n_cell,pred_index,true_index = classify_knn_PCA(Xpc,pca,nPC,n_test,n_train,yc,ycell,yim, weighted, plot)
        elif classification in {"lda","LDA"}:
            acc,n_cell,pred_index,true_index = classify_LDA_PCA(Xpc,pca,nPC,n_test,n_train,yc,ycell,yim, plot)
        elif classification in {"svm","SVM"}:
            acc,n_cell,pred_index,true_index = classify_svm_PCA(Xpc,pca,nPC,n_test,n_train,yc,ycell,yim, plot)
        elif classification in {"wn","wavenumber","WN","Wn"}:
            acc,n_cell,pred_index,true_index = classify_wn(X,wFUSE,wn,n_test,n_train,yc,ycell,yim, weighted, plot)
        return acc,n_cell,pred_index,true_index






    


    
