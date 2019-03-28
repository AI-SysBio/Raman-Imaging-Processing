
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes
from pandas import DataFrame

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from colorama import Fore
from colorama import Style

from libRDT import RDT_Clustering




    
    
def RDT_subclust(filename,n_clusters):
        
    distance = "cityblock"
    msize = 2 #markersize for scatterplot
    
    data = np.load(filename)
    
    
    X = data["data"]
    wn = data["wavenumber"]
    yi = data["date_label"]
    yc = data["cancer_label"]
    yl = data["line_label"]
    ycell = data["cell_label"]
    yim = data["image_label"]
    yfile = data["filename"]
    ypos = data["spectra_position"]
        
    keep_Date = False; Date = 1
    if keep_Date:
        nkeep = np.where(yi == Date)
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
    print("    n = ", N, " spectra" )
    print("    f  = ", f, " wavenumbers" ) 
        
    if not os.path.exists("Sub_cellular_study/Clusters"):
        os.makedirs("Sub_cellular_study/Clusters")
    
    
    wcheck = 2801
    icheck = np.argmin(np.abs(wn-wcheck))
    nkeep_forclustering = np.where(X[:,icheck] < 0.8)
    
    y_clust = RDT_Clustering(X,n_clusters,distance,nkeep_forclustering,plot=True)
        
        
    #reanrange the clusters by descending lipid intensity
    lip_val = np.zeros(n_clusters)
    for i in range(n_clusters):
        nclust = np.where(y_clust == i)
        lip_val[i] = np.max(np.mean(X[nclust],axis = 0))  
    ind_order = np.argsort(np.argsort(lip_val)) #need to call it twice for float somehow, see forum
    for i in range(np.size(y_clust)):
        y_clust[i]=ind_order[y_clust[i]]
        
                
    if n_clusters >= 6:
        color = ["blue", "darkviolet", "green", "gold","darkorange","red", "cadetblue", "orange", "crimson", "navy", "olive", "darkcyan", "darkorange","blue", "darkviolet", "green", "gold","darkorange","red", "cadetblue", "orange", "crimson", "navy", "olive", "darkcyan", "darkorange"]
    if n_clusters == 5:    
        color = ["blue", "darkviolet", "green", "gold","red"]
    if n_clusters == 4:    
        color = ["blue", "green", "gold","red"]
    if n_clusters == 3:    
        color = ["blue", "green","red"]
    if n_clusters == 2:
        color = ["blue", "red"]           
    
    
    nimg = np.max(yim)+1    
    n_cells=0
    for img_index in range(0,nimg):
        n_img = np.where(yim == img_index)
        if np.size(n_img) > 0:
            ncells = int(np.max(ycell[n_img]))+1
            filename = yfile[n_img][0]
            back = np.zeros((400,450))
            plt.imshow(back, origin='lower')
            xs, ys = np.where(back==0)
            plt.scatter(ys, xs, label="background", s=0.3, c = "black")
            for cell_index in range(0,ncells):
                n_sample = np.where(np.logical_and(ycell == cell_index, yim == img_index))
                pos_cell = ypos[n_sample[0]]
                
                
                color_clust = []

                
                if np.size(n_sample) > 0:
                    
                    for i in range(len(y_clust[n_sample[0]])):
                        color_clust.append(color[y_clust[n_sample[0]][i]]) 
                        
                    plt.scatter(pos_cell[:,1]+15,pos_cell[:,0]+15, c = color_clust, cmap='RdYlGn', s = msize)
                    if n_cells==0:
                        X_pos=[pos_cell]
                    else:
                        X_pos.append(pos_cell)
                    n_cells += 1
                        
            plt.axis("off")
            plt.title("Clustering results for %s_%s" % (filename,n_clusters))
            plt.savefig("Sub_cellular_study/Clusters/%s_%s.png" %  (filename,n_clusters))
            plt.show()
                
    print("   There is", n_cells, "cells")
        
        

        
    cell_pop = np.zeros((n_cells,n_clusters))
    cell_label = np.zeros(n_cells)
    cell_date = np.zeros(n_cells)
    cell_img = np.zeros(n_cells)
    cell_line = np.zeros(n_cells)
    cell_file = []
    i=0
    
    plt.figure(figsize=(7,4))
    for img_index in range(0,nimg):
        n_img = np.where(yim == img_index)
        if np.size(n_img) > 0:
            ncells = int(np.max(ycell[n_img]))+1
            for cell_index in range(0,ncells):
                n_sample = np.where(np.logical_and(ycell == cell_index, yim == img_index))
                if np.size(n_sample) > 0:
                    for icl in range(n_clusters):
                        cell_pop[i,icl] = np.size(np.where(y_clust[n_sample]==icl))/np.size(n_sample)
                    cell_label[i]= yc[n_sample][0]
                    cell_date[i]= yi[n_sample][0]
                    cell_img[i]= yim[n_sample][0]
                    cell_line[i] = yl[n_sample][0]
                    cell_file.append(yfile[n_sample][0])
                    i+=1
                    
    X_clust = np.zeros((n_clusters,X.shape[1]))
    for i in range(n_clusters):
        n_clust = np.where(y_clust == i)
        X_clust[i,:] = np.mean(X[n_clust], axis = 0)
        
                        
    data = {"population": cell_pop, "cancer_label": cell_label, "date_label":cell_date, "image_label":cell_img, "line_label":cell_line, "filename":cell_file, "class_av":X_clust, "wavenumber":wn, "cell_position":X_pos}
    return data
    



def plot_clusters(data):
    
    #plot the cells individually to check the clusters
    X_clust = data["class_av"]
    wn = data["wavenumber"]
    X = data["population"]
    yc = data["cancer_label"]
    yl = data["line_label"]
    n_clusters = X_clust.shape[0]
    
    if n_clusters >= 6:
        color = ["blue", "darkviolet", "green", "gold","darkorange","red", "cadetblue", "orange", "crimson", "navy", "olive", "darkcyan", "darkorange","blue", "darkviolet", "green", "gold","darkorange","red", "cadetblue", "orange", "crimson", "navy", "olive", "darkcyan", "darkorange"]
    if n_clusters == 5:    
        color = ["blue", "darkviolet", "green", "gold","red"]
    if n_clusters == 4:    
        color = ["blue", "green", "gold","red"]
    if n_clusters == 3:    
        color = ["blue", "green","red"]
    if n_clusters == 2:
        color = ["blue", "red"]    
    
    
    #broken plot
    plt.rcParams.update({'font.size': 12})
    w1 = 1800
    w2 = 2800
    i1 = np.argmin(abs(wn-w1))
    i2 = np.argmin(abs(wn-w2))     
    wn1 = wn[:i1]
    wn2 = wn[i2:]
    
    plt.figure(figsize=(12,4))
    bax = brokenaxes(xlims=((min(wn1),max(wn1)), (min(wn2)-20,max(wn2))), hspace=0.15)
    for i in range(n_clusters):
        bax.plot(wn, X_clust[i,:], color = color[i])
    bax.set_ylabel("Raman Intensity [au]")  
    bax.set_xlabel('Wavenumber [cm-1]')
    plt.title("Average spectrum of each spectral class")
    plt.savefig("Sub_cellular_study/spectral_class.png")
    plt.show()     
    
  
    
    
    #generate histogram by malignancy:    
    nftc = np.where(yc == 1)
    nthy = np.where(yc == 0)
    ftc_pop = np.median(X[nftc],axis=0)
    nthy_pop = np.median(X[nthy],axis=0)
    nthy_err = np.concatenate(((nthy_pop - np.percentile(X[nthy],25,axis=0)-0.005).reshape(1,-1),(np.percentile(X[nthy],75,axis=0) - nthy_pop).reshape(1,-1)),axis=0)
    ftc_err =  np.concatenate(((ftc_pop  - np.percentile(X[nftc],25,axis=0)-0.005).reshape(1,-1),(np.percentile(X[nftc],75,axis=0) -  ftc_pop).reshape(1,-1)),axis=0)
    
    
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(7,4))
    n, bins, patches = plt.hist(np.arange(n_clusters),bins=n_clusters, weights=nthy_pop,alpha=0.7, rwidth=0.85, color='blue')
    for i, p in enumerate(patches):
        plt.setp(p, 'facecolor', color[i]) 
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Class (Ascending Lipid Intensity)')
    plt.ylabel('Frequency')
    plt.title('Median NT cells class population') 
    bincenters = 0.5*(bins[1:]+bins[:-1])
    width      = 0.05
    plt.bar(bincenters, n, width=width, color='r', yerr=nthy_err, alpha = 0, error_kw=dict(lw=3, capsize=5, capthick=3))
    plt.ylim([-0.05,0.75])
    plt.show()
    

    plt.figure(figsize=(7,4))
    n, bins, patches = plt.hist(np.arange(n_clusters),bins=n_clusters, weights=ftc_pop,alpha=0.7, rwidth=0.85, color="red")
    for i, p in enumerate(patches):
        plt.setp(p, 'facecolor', color[i]) 
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Class (Ascending Lipid Intensity)')
    plt.ylabel('Frequency')
    plt.title('Median FTC cells class population')
    bincenters = 0.5*(bins[1:]+bins[:-1])
    width      = 0.05
    plt.bar(bincenters, n, width=width, color='r', yerr=ftc_err, alpha = 0, error_kw=dict(lw=3, capsize=5, capthick=3))
    plt.ylim([-0.05,0.75])
    plt.show()
    
    
    
    #gini importance
    Ntry = 1000
    gini_score = np.zeros((Ntry,X.shape[1]))
    for i in range(Ntry):
        model = RandomForestClassifier()
        model.fit(X, yc)
        gini_score[i,:] = model.feature_importances_
        
    importance = np.mean(gini_score, axis = 0)
    
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(7,4))
    n, bins, patches = plt.hist(np.arange(n_clusters),bins=n_clusters, weights=importance,alpha=0.7, rwidth=0.85, color="red")
    for i, p in enumerate(patches):
        plt.setp(p, 'facecolor', color[i]) 
    plt.xlabel("Class (Ascending Lipid Intensity)")
    plt.ylabel("Gini importance")
    plt.ylim([0,0.6])
    
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    plt.show()  
    
    
    
    #Univariate Selection
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X, yc)
    importance = fit.scores_
    
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(7,4))
    n, bins, patches = plt.hist(np.arange(n_clusters),bins=n_clusters, weights=importance,alpha=0.7, rwidth=0.85, color="red")
    for i, p in enumerate(patches):
        plt.setp(p, 'facecolor', color[i]) 
    plt.xlabel("Class (Ascending Lipid Intensity)")
    plt.ylabel("K importance")
    #plt.ylim([0,0.6])
    
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    plt.show()  
    
    
    
    

def classify_subclust(data,labels, plot = False): 
    
    ##try classifying with different classifier   
    X = data["population"]
    
    N = X.shape[0]
    f = X.shape[1]            
    print("    n =", N, "cells" )
    print("    f  =", f, "spectral classes" )    
    
    print("\n   Calssification with different classifiers...")
    
    plot = False
    c1,_,_,acc1 = test_classifier(data,plot, "knn", labels, plot_final = plot)
    c2,_,_,acc2 = test_classifier(data,plot, "lda", labels, plot_final = plot)
    c3,c,d,acc3 = test_classifier(data,plot, "svm", labels, plot_final = plot)
    
    
    color = ["red","green"]
    color2 = ["blue", "red", "gold", "darkviolet", "cadetblue", "navy", "olive", "darkcyan", "darkorange", "brown", "coral", "chartreuse", "orange", "crimson", "chocolate", "maroon","dodgerblue","goldenrod", "darkred","darkblue","blue", "red", "gold", "darkviolet", "cadetblue", "navy", "olive", "darkcyan", "darkorange", "brown", "coral", "chartreuse", "orange", "crimson", "chocolate", "maroon","dodgerblue","goldenrod", "darkred","darkblue"] 
    n_cell = np.size(c1)
    colors_c = []; colors_d = []
    colors_p1 = []; colors_p2 = []; colors_p3 = []
    date_c = []
    for i in range(n_cell):
        colors_p1.append(color[c1[i]])
        colors_p2.append(color[c2[i]])
        colors_p3.append(color[c3[i]])
        colors_c.append(color2[c[i]])
        colors_d.append(color2[d[i]+2])
        if labels[d[i]] not in date_c:
            date_c.append(labels[d[i]])
        
        
    print("\n\n  Classification results:")
    print("    line 1 = date [", end='')
    for i in range(len(date_c)-1):
        print("%s, " % date_c[i] , end='')
    print("%s]"% date_c[-1])
    print("    line 2 = FTC/NT (red/blue)")
    print("    line 3 (kNN acc) = %.2f" %acc1 )
    print("    line 4 (LDA acc) = %.2f" %acc2 )
    print("    line 5 (SVM acc) = %.2f" %acc3)    
        
    plt.figure(figsize=(13,2))
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,2),c=colors_d,s = 7)
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,1.5),c=colors_c,s = 7)    
    
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,0.5),c=colors_p1,s = 7)
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,0),c=colors_p2,s = 7)
    plt.scatter(np.linspace(0,n_cell,n_cell),np.full(n_cell,-0.5),c=colors_p3,s = 7)
    plt.axis("off")
    #plt.tick_params(axis ="both", bottom = False, labelbottom=False, left = False, labelleft=False)
    plt.ylim([-2.2,3.7])
    plt.show()
    
    

    


def test_classifier(data,plot,classifier, labels, plot_final = False, imshow_results = False):
    
    plot_final = True
    X = data["population"]
    yc = data["cancer_label"]
    yi = data["date_label"]
    
    print("\n\n  Predictions on testset with %s: " % classifier)
    acc_ = []
    n_cell_ = []
    y_score = np.array([])
    y_true = np.array([])
    y_date  = np.array([])
    ndates = int(np.max(yi)+1)
    
    
    for j in range(ndates):
        acc0,n_cell0,pred0,true0 = test_date(j,X,yi,yc, plot, classifier)
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
        
    if classifier in {"lda","LDA"}:
        imshow_results = False
    
    if imshow_results:
        X_pos = data["cell_position"]
        yfile = data["filename"]
        yim = data["image_label"]
        
        nimg = int(np.max(yim)+1)
        ndate = int(np.max(yi)+1)
        color = ["red","green"]
        back = np.zeros((400,450))
        
        ind = 0
        for date_index in range(0,ndate):
            n_date = np.where(yi == date_index)
            if np.size(n_date) > 0:        
                for img_index in range(0,nimg):
                    n_img = np.where(np.logical_and(yim == img_index,yi == date_index))
                    if np.size(n_img) > 0:
                        filename = yfile[int(n_img[0][0])]
                        
                        plt.imshow(back, origin='lower')
                        xs, ys = np.where(back==0)
                        plt.scatter(ys, xs, label="background", s=0.3, c = "black")
                                
                        #ncells = np.size(n_img)
                        for cell_index in n_img[0]:
                            pos_cell = X_pos[cell_index]
                            if cell_index < np.size(correct):
                                plt.scatter(pos_cell[:,1]+15,pos_cell[:,0]+15, c = color[correct[ind]], cmap='RdYlGn', s = 3.5)
                                ind += 1
        
                                                
                        plt.title("Classification results for image %s" % filename)
                        plt.axis("off")
                        plt.savefig("Sub_cellular_study/Clusters/%s_class.png" %  (filename))
                        plt.show()    
    
    
    
    
    
    print(" ----------------------------------------------------------------------------------------------------------- ")    
    
    
    return correct,y_true.astype(int),y_date.astype(int),acc



    
    
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
    
    if plot:
        print("\n  Training set:")
        print("   ", NTHY, "Nthy spectra")
        print("   ", NFTC, "FTC spectra")
        print("    Ratio FTC/Nthy = 1.00:%.2f" % (ratio))
        
        
    #train a kNN classifier for nearest neighboor classification:
    n_neighbors = 5
    kNN = KNeighborsClassifier(n_neighbors=n_neighbors)
    kNN.fit(X_train, yc_train)
    y_pred_train = kNN.predict(X_train)
    acc = accuracy_score(yc_train,y_pred_train)
    print("   kNN training accuracy is", acc)
    
    
    #test: classify each spectra of a test cell with kNN, account for the ratio correction
    y_nei_nc = kNN.predict_proba(X_test)[:,0]*n_neighbors
    y_nei_can = kNN.predict_proba(X_test)[:,1]*n_neighbors
    y_pred_proba = y_nei_can*ratio/(y_nei_can*ratio+y_nei_nc)
    y_pred = y_pred_proba > 0.5
    acc = accuracy_score(yc_test, y_pred)

    if plot:     
        print("  Prediction accuracy with %s cells is %s/%s = %.3f \n" % (n_cell, int(round(acc*n_cell,0)), n_cell, acc))
    
    return acc,n_cell,y_pred_proba,yc_test






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
    
    SVM = svm.SVC(class_weight = class_weight)
    SVM.fit(X_train, yc_train)
    y_pred_train = SVM.predict(X_train)
    acc = accuracy_score(yc_train,y_pred_train)
    print("   SVM training accuracy is", acc)
    
    y_pred_proba = SVM.predict(X_test)
    y_pred = y_pred_proba > 0.5
    acc = accuracy_score(yc_test, y_pred)

    if plot:     
        print("  Prediction accuracy with %s cells is %s/%s = %.3f \n" % (n_cell, int(round(acc*n_cell,0)), n_cell, acc))
    
    return acc,n_cell,y_pred_proba,yc_test
                    
                
 
    

    

def classify_LDA(X,n_test,n_train,yc, plot):  
      
    
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
        
    
    
    mda = LinearDiscriminantAnalysis()
    X_reduced_train = mda.fit_transform(X_train, yc_train) 
    y_pred_train = X_reduced_train[:,0]>0
    acc = accuracy_score(yc_train,y_pred_train)
    print("   LDA training accuracy is", acc)
    
    X_reduced_test = mda.transform(X_test)
    y_pred = X_reduced_test[:,0]>0
    acc = accuracy_score(yc_test, y_pred)

    if plot:     
        print("  Prediction accuracy with %s cells is %s/%s = %.3f \n" % (n_cell, int(round(acc*n_cell,0)), n_cell, acc))
    
    return acc,n_cell,y_pred,yc_test







    
