from sklearn import metrics, datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # LDA
import pandas as pd

iris = datasets.load_iris()
X = iris.data
Y = iris.target

"Partie A: K-Moyenne"
print("Partie A\n")

"1"
#Fonction kmeans
def baricentre(X, Y):                                                   
    b_s = []
    for k in np.unique(Y):
        A = X[np.where(Y == k)]
        b_s.append(np.mean(A, axis=0))
    return np.array(b_s)


def kmoyenne(data, k):
    bar_i = np.random.randint(0, len(data), k)
    ls_bar = data[bar_i]
    last_pred=np.zeros(len(data))
    for i in range(100):
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(ls_bar, np.array(list(range(k))))
        prediction = knn.predict(data)
        if np.all(prediction == last_pred):
            return prediction
        last_pred = np.copy(prediction)
        ls_bar = baricentre(data, prediction)
        
        
#Test sur Iris avec notre fonction
prediction = kmoyenne(X, 3)
plt.title("Representation des données, reduction de dimension avec kmoyenne")
plt.scatter(X[:, 0], X[:, 1], c=prediction)
plt.scatter(baricentre(X,prediction)[:,0],baricentre(X,prediction)[:,1],c='red')
plt.show()

# #Test sur Iris avec kmeans sklearn
kmeans=KMeans(n_clusters=3).fit(X)
centroid=kmeans.cluster_centers_
plt.title("Representation des données avec Kmeans")
plt.scatter(X[:,0],X[:,1],c=kmeans.labels_)


plt.scatter(centroid[:,0],centroid[:,1],c='red')
plt.show()

"2"
#Rapport

"3"

score={}
for k in range(2,10):
    for i in range(10):
        km=KMeans(n_clusters=k,init='k-means++', n_init=10,  random_state=10).fit(X)
        score[k]=silhouette_score(X,km.labels_)
plt.figure(figsize=(8,8))
plt.plot(list(score.keys()),list(score.values()))
plt.xlabel("number of cluster")
plt.ylabel("silhouette score")
plt.show()
        

"4"
pca = PCA(n_components=2)
X = pca.fit_transform(X)

plt.figure(figsize=(4, 3))
plt.scatter(X[:, 0], X[:, 1],c=Y)
plt.title("Iris PCA")

lda = LinearDiscriminantAnalysis(n_components=2)
IrisLDA=lda.fit(iris.data,iris.target).transform(iris.data)

plt.figure(figsize=(4, 3))
plt.scatter(IrisLDA[:, 0], IrisLDA[:, 1],c=iris.target)
plt.title("Iris LDA")





"PARTIE B"

"1"
proj = pd.read_csv('choixprojetstab.csv', sep = ';')
print(proj)
proj.info()
C = proj['étudiant·e'] 
M = proj.values[:, 1:]          #Ici, c'est toutes les lignes et toutes les colonnes sauf la première
print(M)
proj.dtypes                     #Attribut de pandas :) Différence fonction / attribut : pas de parenthèse avec un attribut. 
for i in range(1,len(M)), 
    print(proj.dtypes[i]==1)    #on veut juste vérifier que les valeurs dans M soient bien numériques. Ici, on a pas besoin de les changer avec astype. 

"2"
#Affinity propagation
af = AffinityPropagation(preference=3).fit(M)
cluster_centers_indices1 = af.cluster_centers_indices_
plt.title("Representation des données avec AffinityPropagation")
plt.scatter(M[:,0],M[:,1],c=af.labels_)
plt.show()

#Mean-shift
bandwidth = estimate_bandwidth(M, quantile=0.2)
ms = MeanShift(bandwidth=3, bin_seeding=True)
ms.fit(M)
cluster_centers2 = ms.cluster_centers_
plt.title("Representation des données avec Meanshift")
plt.scatter(M[:,0],M[:,1],c = ms.labels_)
plt.show()

#Spectral clustering
sc = SpectralClustering(n_clusters = 3)
sc.fit(M)
plt.title("Representation des données avec Special Clustering")
plt.scatter(M[:,0],M[:,1],c = sc.labels_)
plt.show()

#Birch
B = Birch(n_clusters = 3)
B.fit(M)
plt.title("Representation des données avec Birch")
plt.scatter(M[:,0],M[:,1],c = B.labels_)
plt.show()


#DBSCAN
db = DBSCAN(eps=0.078)
db.fit(M)
plt.title("Representation des données avec DBSCAN")
plt.scatter(M[:,0],M[:,1],c = db.labels_)
plt.show()


#Gaussian mixtures
G = GaussianMixture(n_components=3, covariance_type='full')
y_pred = G.fit_predict(M)
plt.title("Representation des données avec Gaussian mixtures")
plt.scatter(M[:,0],M[:,1],c = y_pred)
plt.show()

