"""
Created on Wed May 19 16:08:56 2021

@author: gowthas

"""

from sklearn.datasets import make_classification
import sklearn.cluster as cl
import sklearn.mixture as mixture
import matplotlib.pyplot as plt
from numpy import where, unique

"""
Different clustering techniques
https://scikit-learn.org/stable/modules/clustering.html
https://machinelearningmastery.com/clustering-algorithms-with-python/
"""

# x, y = make_classification(1000, 2, 2, 0, n_clusters_per_class=1, random_state=4)
# for c in range(2):
#     row_ix = where(y == c)
#     plt.scatter(x[row_ix, 0], x[row_ix, 1])
#     plt.plot()
    
# plt.show()

x, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=4)


"""
Affinity Propagation
Affinity Propagation involves finding a set of exemplars that best summarize the data

https://towardsdatascience.com/unsupervised-machine-learning-affinity-propagation-algorithm-explained-d1fef85f22c8
"""

def affinity_propagation(x):
    model = cl.AffinityPropagation(damping=.9)
    model.fit(x)
    yhat = model.predict(x)
    clusters = unique(yhat)
    print(len(clusters))
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(x[row_ix, 0], x[row_ix, 1])
        
    plt.plot()
    plt.show()

 
"""
Agglomerative clustering

Agglomerative clustering involves merging examples until the desired number of clusters 
is achived

It is hierarchical clustering

"""

def agglomerative_clustering(x):
    model = cl.AgglomerativeClustering(n_clusters=3)
    # model = cl.AgglomerativeClustering(n_clusters=3, affinity='l1', linkage='average')
    # model = cl.AgglomerativeClustering(n_clusters=3, affinity='l1', linkage='single')
    # model = cl.AgglomerativeClustering(n_clusters=3, affinity='l1', linkage='complete')
    
    # model = cl.AgglomerativeClustering(n_clusters=3, affinity='l2', linkage='average')
    # model = cl.AgglomerativeClustering(n_clusters=3, affinity='l2', linkage='single')
    # model = cl.AgglomerativeClustering(n_clusters=3, affinity='l2', linkage='complete')
    
    # model = cl.AgglomerativeClustering(n_clusters=3, affinity='manhattan', linkage='average')
    # model = cl.AgglomerativeClustering(n_clusters=3, affinity='manhattan', linkage='single')
    # model = cl.AgglomerativeClustering(n_clusters=3, affinity='manhattan', linkage='complete')
    
    # model = cl.AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='average')
    # model = cl.AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='single')
    # model = cl.AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='complete')
    
    
    yhat = model.fit_predict(x)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(x[row_ix, 0], x[row_ix, 1])
    
    plt.plot()
    plt.show()


    
"""
BIRCH - Balanced Iterative Reducing and clustering using Hierarchies 

It constructs a tree data structure with the cluster centroids being read off the leaf.
These can be either the final cluster centroids or can be provided as input 
to another clustering algorithm such as AgglomerativeClustering.

Hyperparameters: n_clusters, thereshold

https://www.ques10.com/p/9298/explain-birch-algorithm-with-example/

Only useful for metric data(Data that can be represented in Euclidian space)

"""

def birch_clustering(x):
    model = cl.Birch(threshold=0.01, n_clusters =3)
    yhat = model.fit_predict(x)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(x[row_ix, 0], x[row_ix, 1])
    
    plt.plot()
    plt.show()
    


"""
DBSCAN: Density Based Spatial clustering of Applications with Noise

-1 label in returned value represents noise

"""
def dbscan_clustering(x): 
    model = cl.DBSCAN(eps=0.3, min_samples=9)
    yhat = model.fit_predict(x)
    clusters = unique(yhat)
    for cluster in clusters:
        # if cluster != -1:
        # print(cluster)
        row_ix = where(yhat == cluster)
        plt.scatter(x[row_ix, 0], x[row_ix, 1])
        
    plt.plot()
    plt.show()



"""
OPTICS: Ordering Points To Identify Clustering Structure

Modified version of DBSCAN
Hyperparameters
    eps, min_samples, cluster_method: xi or dbscan

"""

def optics_clustering(x):
    model = cl.OPTICS(min_samples=9, eps=0.3, cluster_method="dbscan")
    yhat = model.fit_predict(x)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(x[row_ix, 0], x[row_ix, 1])
        
    plt.plot()
    plt.show()



"""
K-Mean clustering
Aims in partitioning n observations to k-clusters in which each observation
belongs to a cluster with nearest mean serving as prototype for the cluster.
It minimizes the variance with in cluster
"""
def kmean_clustering(x):
    model = cl.KMeans(n_clusters=3, n_init=20)
    yhat = model.fit_predict(x)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(x[row_ix, 0], x[row_ix, 1])
        
    plt.plot()
    plt.show()
    

"""
mini-batch K-Mean clustering

Modified version of K-Mean, which makes updates to the cluster centroids using
mini-batches of samples rather than the entire dataset. 
* Faster for large dataset
* Most robust for statistical noise

"""
def mini_batch_k_mean_clustering(x):
    model = cl.MiniBatchKMeans(n_clusters = 3)
    yhat = model.fit_predict(x)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(x[row_ix, 0], x[row_ix, 1])
        
    plt.plot()
    plt.show()


"""
Mean shift clustering

Mean shift clustering involves finding and adapting centroids based on the 
density of examples in the feature space.

hyperparameters: bandwidth 
"""

def mean_shift_clustering(x):
    model = cl.MeanShift()
    yhat = model.fit_predict(x)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(x[row_ix, 0], x[row_ix, 1])
        
    plt.plot()
    plt.show()


"""
Spectral clustering

Clustering uses the top eigenvectors of a matrix derived from the distance 
between points.

hyperparameters: n_clusters

"""
def spectral_clustering(x):
    model = cl.SpectralClustering(n_clusters= 3)
    yhat = model.fit_predict(x)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        plt.scatter(x[row_ix, 0], x[row_ix, 1])
        
    plt.plot()
    plt.show()


"""
Gaussian mixture model

Hyperparameter: n_components
"""
def gaussian_mixture_model(x):
    model = mixture.GaussianMixture(n_components=2)
    yhats = model.fit_predict(x)
    clusters = unique(yhats)
    for cluster in clusters:
        row_ix = where(yhats == cluster)
        plt.scatter(x[row_ix, 0], x[row_ix, 1])
        
    plt.plot()
    plt.show()
