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
# Fonction kmeans


def baricentre(X, Y):
    b_s = []
    for k in np.unique(Y):
        A = X[np.where(Y == k)]
        b_s.append(np.mean(A, axis=0))
    return np.array(b_s)


def kmoyenne(data, k):
    bar_i = np.random.randint(0, len(data), k)
    ls_bar = data[bar_i]
    last_pred = np.zeros(len(data))
    for i in range(100):
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(ls_bar, np.array(list(range(k))))
        prediction = knn.predict(data)
        if np.all(prediction == last_pred):
            return prediction
        last_pred = np.copy(prediction)
        ls_bar = baricentre(data, prediction)


# Test sur Iris avec notre fonction
prediction = kmoyenne(X, 3)
plt.title("Representation des données, reduction de dimension avec kmoyenne")
plt.scatter(X[:, 0], X[:, 1], c=prediction)
plt.scatter(baricentre(X, prediction)[:, 0],
            baricentre(X, prediction)[:, 1], c='red')
plt.show()

# #Test sur Iris avec kmeans sklearn
kmeans = KMeans(n_clusters=3).fit(X)
centroid = kmeans.cluster_centers_
plt.title("Representation des données avec Kmeans")
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)


plt.scatter(centroid[:, 0], centroid[:, 1], c='red')
plt.show()

"2"
# Rapport

"3"

score = {}
for k in range(2, 10):
    for i in range(10):
        km = KMeans(n_clusters=k, init='k-means++',
                    n_init=10,  random_state=10).fit(X)
        score[k] = silhouette_score(X, km.labels_)
plt.figure(figsize=(8, 8))
plt.plot(list(score.keys()), list(score.values()))
plt.xlabel("number of cluster")
plt.ylabel("silhouette score")
plt.show()


"4"
pca = PCA(n_components=2)
X = pca.fit_transform(X)

plt.figure(figsize=(4, 3))
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.title("Iris PCA")

lda = LDA(n_components=2)
IrisLDA = lda.fit(iris.data, iris.target).transform(iris.data)

plt.figure(figsize=(4, 3))
plt.scatter(IrisLDA[:, 0], IrisLDA[:, 1], c=iris.target)
plt.title("Iris LDA")
plt.show()


# PARTIE B

# 1
proj = pd.read_csv('choixprojetstab.csv', sep=';')
print(proj)
proj.info()
C = proj['étudiant·e']
# Ici, c'est toutes les lignes et toutes les colonnes sauf la première
M = proj.values[:, 1:]
print(M)
# Attribut de pandas :) Différence fonction / attribut : pas de parenthèse avec un attribut.
proj.dtypes
for i in range(1, len(M)):
    # on veut juste vérifier que les valeurs dans M soient bien numériques. Ici, on a pas besoin de les changer avec astype.
    print(proj.dtypes[i])

# 2

pca = PCA(n_components=26)
M_pca = pca.fit_transform(M)
plt.title("Representation des données")
plt.scatter(M_pca[:, 0], M_pca[:, 1])
plt.show()

pca = PCA().fit(M)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.cumsum(pca.explained_variance_ratio_))

x = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0]
xy = (x, np.cumsum(pca.explained_variance_ratio_)[x])
ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
ax.scatter(*xy, marker="X", color="red")

plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

models = {
    "AffinityPropagation": AffinityPropagation(random_state=42, damping=0.5),
    "Meanshift": MeanShift(bandwidth=4, bin_seeding=False),
    "SpectralClustering": SpectralClustering(n_clusters=8),
    "Birch": Birch(n_clusters=None, threshold=1),
    "OPTICS": OPTICS(min_samples=5),
    "GaussianMixture": GaussianMixture(n_components=12)
}

silhouettes = []
silhouettes_pca26 = []
silhouettes_pca2 = []

num_cluster = []
num_cluster_pca26 = []
num_cluster_pca2 = []

names = []

for name, model in models.items():
    label = model.fit_predict(M)
    label_PCA26 = model.fit_predict(M_pca)
    label_PCA2 = model.fit_predict(M_pca[:, 0:2])

    fig, ax = plt.subplots(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.scatter(M_pca[:, 0], M_pca[:, 1], c=label)
    plt.title("données originales")
    plt.xlabel(str(len(np.unique(label))) + " clusters")

    plt.subplot(2, 2, 2)
    plt.scatter(M_pca[:, 0], M_pca[:, 1], c=label_PCA2)
    plt.title("données après PCA (n=2)")
    plt.xlabel(str(len(np.unique(label_PCA2))) + " clusters")

    plt.subplot(2, 2, 3)
    plt.scatter(M_pca[:, 0], M_pca[:, 1], c=label_PCA26)
    plt.title("données après PCA (n=26)")
    plt.xlabel(str(len(np.unique(label_PCA26))) + " clusters")

    plt.suptitle("Clustering avec " + name)
    plt.show()

    silhouettes.append(silhouette_score(M, label))
    silhouettes_pca26.append(silhouette_score(M, label_PCA26))
    silhouettes_pca2.append(silhouette_score(M, label_PCA2))

    num_cluster.append(len(np.unique(label)))
    num_cluster_pca26.append(len(np.unique(label_PCA26)))
    num_cluster_pca2.append(len(np.unique(label_PCA2)))

    names.append(name)

bar_width = 0.25
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(12, 8))
bar1 = ax.barh(x,               silhouettes,       bar_width, label="NO PCA")
bar2 = ax.barh(x + bar_width,   silhouettes_pca2,  bar_width, label="PCA n=2")
bar3 = ax.barh(x + bar_width*2, silhouettes_pca26, bar_width, label="PCA n=26")

def print_nb_cluster(bar, nb_cluster):
    for i, b in enumerate(bar):
        
        width = b.get_width()
        x_coor = 0.01
        if width < 0:
            alignement = "left"
        else:
            x_coor *= -1
            alignement = "right"
        ax.text(x_coor, b.get_y() + b.get_height()/4., 
        str(nb_cluster[i]) + " clusters",
        ha=alignement, va='bottom', rotation=0)

print_nb_cluster(bar1, num_cluster)
print_nb_cluster(bar2, num_cluster_pca2)
print_nb_cluster(bar3, num_cluster_pca26)

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
