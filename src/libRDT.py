
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform






def RDT_Clustering(X,n_clusters,distance,Nrand = 10000,plot=False):

    """
    Perform Clustering using Rate distortion theory [1].
    
    Inputs:
    ----------
    X: 2D array
        set of area normalized histograms for a set of segments from a
        time-series segmentation. Histograms bins along the columns and segments
        along the rows.
        
    n_clusters: int scalar
        desired number of clusters
        
    distance: string
        distance metric used for the pairwize distance computation
        
    Nrand: int scalar
        Number of instance to use for generating the clusters to avoid memory error, it is not advised to go higher than 10 000
        
        
    Outputs:
    ----------
    clusters: 1D array
        array containing the index of each example
        
        
    [1] Tavakoli M, Taylor JN, Li CB, Komatsuzaki T, Pressé S. 
        Single molecule data analysis: an introduction.
        arXiv preprint arXiv:1606.00403. 2016 Jun 1.    
    """
    
    random.seed(a=None)
    
    # Input parameters
    beta = 0.1                       # beta parameter for RDT    - recommended is 100/max(dij(:))
    eps = 1e-5                     # convergence tolerance for RDT (1e-3 - small)
    nrs = 5                        # number of times to initialize RDT
    
    """
    Some notes on beta:
      - Large beta will return harder clustering results, psi -> 0.1
      - If beta is too small, the psi will be roughly equal across all clusters and segments
      - If beta is too large, the exponential vanishes. rdclust will return NaNs for most outputs
    """
    
    
    #random sampling of the spectra set to avoid memory error
    if X.shape[0] > Nrand:
        nkeep = np.random.randint(X.shape[0],size=Nrand)
        Xkeep = X[nkeep,:]
    
    else:
        Xkeep = X
        Nrand = X.shape[0]

    print("    Computing pairwise distances for %s spectra..." % Nrand)
    dij = squareform(pdist(Xkeep, distance))
    
    
    # Cluster the segments. Run RDT with nrs random restarts
    ni = np.ones((Xkeep.shape[0]))       # ni: vector containing the number of samples in each of the N segments.
    pi = ni/np.sum(ni)                # frequentist probabilities of the segments
    print("    Running RDT clustering...")
    Ck = []
    fk = np.zeros(nrs)
    for k in range(nrs):
        C = rdclust(dij,pi,beta,n_clusters,eps)
        Ck.append(C)
        fk[k] = C.F

    i = np.argmin(fk)
    C = Ck[i]             # choose the restart that minimizes RDT functional
    CLUSTER = C.psi
    
    if plot:
        plt.imshow(CLUSTER, extent=[0,Xkeep.shape[0],0,n_clusters], aspect='auto')
        plt.show()
    
    
    clusters_keep = np.zeros(Nrand)
    for ni in range(Nrand):
        clusters_keep[ni] = np.argmax(CLUSTER[:,ni])
    clusters_keep = clusters_keep.astype("int")
    
    
    Xmean = np.zeros((n_clusters,X.shape[1]))
    for k in range(n_clusters):
        nclust = np.where(clusters_keep == k)
        Xmean[k,:] = np.mean(Xkeep[nclust],axis=0)
        
    N = X.shape[0]    
    clusters = np.zeros(N)
    for ni in range(N):
        clusters[ni] = np.argmin(np.sum(np.abs(np.tile(X[ni,:],(n_clusters,1)) - Xmean),axis=1))
    
    return clusters.astype("int")






def rdclust(D,pi,beta,n_clusters,ep):

    """
    Outputs
    ----------
    C: structure with several fields
    C.psi: matrix Pr(state|segment). States on rows, segments on columns.
    C.ps: marginals of states, Pr(state)
    C.I: mutual information, I(states, segments)
    C.ud: mean distortion
    C.F: F = I + beta*ud
    C.Fp: vector containing F at each iteration. useful for convergence checks
    """
    
    N = D.shape[0]  # number of data points to be clustered
    
    
    # initialize
    psi = np.random.rand(n_clusters,N) 
    psi = psi / np.tile(np.sum(psi,axis=0), (n_clusters, 1))
    
    
    ps = np.sum(psi,axis = 1)/N
    ps = ps[:,None]
    pis = psi * np.tile(pi,(n_clusters,1)) / np.tile(ps,(1,N))
    dis = np.zeros(psi.shape)
    for i in range(N):
        di = D[i,:]
        dis[:,i] = np.sum(pis*np.tile(di,(n_clusters,1)),axis = 1)

    Z = np.sum(np.tile(ps,(1,N)) * np.exp(-beta*dis),axis=0)
    I = (1/N) * np.sum(np.sum(psi*np.log(psi/np.tile(ps,(1,N))),axis=0),axis=0)
    ud = np.sum(np.sum(pis*np.tile(ps,(1,N))*dis,axis=0),axis=0)
    Fi = I + beta*ud
    
    
    # iterate until F converges within ep
    Fp = [Fi]
    Ip = [I]
    udp = [ud]
    iter = 0
    converge = False
    while not converge:
        iter += 1
        psi = (np.tile(ps,(1,N))/np.tile(Z,(n_clusters,1)))*np.exp(-beta*dis)
        psi[np.isnan(psi)] = 0
        ps = np.sum(psi*np.tile(pi,(n_clusters,1)), axis = 1)
        ps = ps[:,None]
        pis = psi*np.tile(pi,(n_clusters,1))/np.tile(ps,(1,N))
        for i in range(N):
            di = D[i,:]
            dis[:,i] = np.sum(pis*np.tile(di,(n_clusters,1)),axis = 1)

        Z = np.sum(np.tile(ps,(1,N))*np.exp(-beta*dis),axis=0)
        Isi = np.zeros((n_clusters,N))
        for s in range(n_clusters):
            for i in range(N):
                if psi[s,i] == 0:
                    Isi[s,i] = 0
                else:
                    Isi[s,i] = psi[s,i]*pi[i]*np.log(psi[s,i]/ps[s,0])


        I = np.sum(np.sum(Isi,axis=0),axis=0)
        ud = np.sum(np.sum(pis*np.tile(ps,(1,N))*dis,axis=0),axis=0)
        F = I + beta*ud
        dF = Fi - F
        Fp.append(F)
        Ip.append(I)
        udp.append(ud)
        
        if abs(dF) < ep:
            converge = True
            #print('ep')
        elif dF < 0:
            converge = True
            #print('neg')
        elif np.isnan(F):
            converge = True
            #print('nan')
        elif iter > 200:
            converge = True
            #print('iter')
        else:
            Fi = F

        if converge:
            C = structtype()
            C.psi = psi
            C.ps = ps
            C.pis = pis
            C.I = I
            C.ud = ud
            C.F = Fp[-1]
            C.Fp = Fp
            
    return C



class structtype():
    pass



if __name__ == '__main__':
    
    filename = "processed_spectras_100_raw.npz"
    data = np.load(filename)
    X = data["data"]
    color = ["blue", "darkviolet", "green", "gold","darkorange","red", "cadetblue", "orange", "crimson", "navy", "olive", "darkcyan", "darkorange","blue", "darkviolet", "green", "gold","darkorange","red", "cadetblue", "orange", "crimson", "navy", "olive", "darkcyan", "darkorange"]
    
    #nkeep = np.random.randint(X.shape[0],size=1000)
    #X = X[nkeep,:]
    
    print("\n   ", filename)
    print("    n = ", X.shape[0], " spectra" )
    print("    f  = ", X.shape[1], " wavenumbers\n" ) 
    
    distance = "cityblock"
    n_clusters = 6                         # desired number of clusters
    y_clust = RDT_Clustering(X,n_clusters,distance,Nrand = 1000,plot = True)
    
    
    #reanrange the clusters by descending lipid intensity
    lip_val = np.zeros(n_clusters)
    for i in range(n_clusters):
        nclust = np.where(y_clust == i)
        lip_val[i] = np.max(np.mean(X[nclust],axis = 0))  
    ind_order = np.argsort(np.argsort(lip_val)) #need to call it twice for float somehow, see forum
    for i in range(np.size(y_clust)):
        y_clust[i]=ind_order[y_clust[i]]
    
    
    
    for i in range(n_clusters):
        nclust = np.where(y_clust == i)
        if np.size(nclust) !=0:
            plt.plot(np.mean(X[nclust],axis=0), color = color[i])
    plt.show()







