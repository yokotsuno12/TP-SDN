
from numpy import *
from sklearn import *
import sklearn.preprocessing as sp #pour iris 

iris = datasets.load_iris()
X = iris.data
Y = iris.target

def PPV(X,Y):
    Ypred=[]
    for i,e in enumerate(X):
        L= metrics.pairwise.euclidean_distances(X,e[np.newaxis])
        L = np.delete(L,i)
        Ypred.append(Y[np.argmin(L)])
    return np.array(Ypred)

a=PPV(X, Y)
print(a)
print(Y)
