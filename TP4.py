from sklearn import metrics, datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd

iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Partie A: K-Moyenne
print("Partie A\n")

# 1
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

lda = LDA(n_components=2)
IrisLDA=lda.fit(iris.data,iris.target).transform(iris.data)

plt.figure(figsize=(4, 3))
plt.scatter(IrisLDA[:, 0], IrisLDA[:, 1],c=iris.target)
plt.title("Iris LDA")
plt.show()





# PARTIE B

# 1
proj = pd.read_csv('choixprojetstab.csv', sep = ';')
print(proj)
proj.info()
C = proj['étudiant·e'] 
M = proj.values[:, 1:]          #Ici, c'est toutes les lignes et toutes les colonnes sauf la première
print(M)
proj.dtypes                     #Attribut de pandas :) Différence fonction / attribut : pas de parenthèse avec un attribut. 
for i in range(1,len(M)):
    print(proj.dtypes[i])    #on veut juste vérifier que les valeurs dans M soient bien numériques. Ici, on a pas besoin de les changer avec astype. 

# 2

pca = PCA(n_components=2)
M_pca = pca.fit_transform(M)
plt.title("Representation des données")
plt.scatter(M_pca[:,0],M_pca[:,1])
plt.show()

models = {
     "AffinityPropagation" : AffinityPropagation(),
     "Meanshift"           : MeanShift(bandwidth=3, bin_seeding=True),
     "SpectralClustering"  : SpectralClustering(n_clusters=3),
     "Birch"               : Birch(n_clusters=3),
     "OPTICS"              : OPTICS(min_samples=5),
     "GaussianMixture"     : GaussianMixture(n_components=3, covariance_type='full')
    }

silhouettes = []
silhouettes_pca = []
names = []

for name, model in models.items():
    label = model.fit_predict(M)
    label_PCA = model.fit_predict(M_pca)

    plt.figure()
    
    plt.subplot(1, 2, 1)
    plt.scatter(M_pca[:, 0], M_pca[:, 1], c=label)
    plt.title("données originales")
    
    plt.subplot(1, 2, 2)
    plt.scatter(M_pca[:, 0], M_pca[:, 1], c=label_PCA)
    plt.title("données après PCA")
    
    plt.suptitle("Clustering avec " + name)
    plt.show()
    
    silhouettes.append(silhouette_score(M, label))
    silhouettes_pca.append(silhouette_score(M, label_PCA))
    names.append(name)

bar_width = 0.3
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(x,             silhouettes,     bar_width, label="NO PCA")
ax.barh(x + bar_width, silhouettes_pca, bar_width, label="PCA")
ax.set_yticks(x + bar_width / 2)
ax.set_yticklabels(names)
ax.legend()
plt.title("indices de silhouette")
plt.show()



# #Affinity propagation)
# af = AffinityPropagation()
# label = af.fit_predict(M)
# print(np.unique(label))
# plt.title("Representation des données avec AffinityPropagation")
# lda = LDA(n_components=2)
# M_lda = lda.fit(M, label).transform(M)
# plt.scatter(M_red[:, 0], M_red[:, 1], c=label)
# plt.show()
# plt.scatter(M_lda[:, 0], M_lda[:, 1], c=label)
# plt.show()

# #Mean-shift
# bandwidth = estimate_bandwidth(M, quantile=0.2)
# ms = MeanShift(bandwidth=3, bin_seeding=True)
# label = ms.fit_predict(M)
# plt.title("Representation des données avec Meanshift")
# plt.scatter(M_red[:,0], M_red[:,1], c = label)
# plt.show()

# #Spectral clustering
# sc = SpectralClustering(n_clusters = 3)
# label = sc.fit_predict(M)
# plt.title("Representation des données avec Special Clustering")
# plt.scatter(M_red[:,0],M_red[:,1],c = label)
# plt.show()

# #Birch
# B = Birch(n_clusters = 3)
# label = B.fit_predict(M)
# plt.title("Representation des données avec Birch")
# plt.scatter(M_red[:,0],M_red[:,1],c = label)
# plt.show()


# #DBSCAN
# db = DBSCAN(eps=0.078)
# label = db.fit_predict(M)
# plt.title("Representation des données avec DBSCAN")
# plt.scatter(M_red[:,0],M_red[:,1],c = label)
# plt.show()


# #Gaussian mixtures
# G = GaussianMixture(n_components=3, covariance_type='full')
# label = G.fit_predict(M)
# plt.title("Representation des données avec Gaussian mixtures")
# plt.scatter(M_red[:,0],M_red[:,1],c = label)
# plt.show()

