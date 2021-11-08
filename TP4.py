from sklearn import metrics, datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris.data
Y = iris.target

"Partie A: K-Moyenne"
print("Partie A\n")
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

"1"
prediction = kmoyenne(X, 3)
pca = PCA(n_components=2)
X = pca.fit_transform(X)
plt.title("Representation des données, reduction de dimension avec PCA")
plt.scatter(X[:, 0], X[:, 1], c=prediction)
plt.show()
