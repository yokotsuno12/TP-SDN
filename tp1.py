
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
        np.argmin(L)
        L =np.delete(L,i)
        Ypred.append(Y[np.argmin])
    return Ypred

        


def PPV2(X,Y):
    s=0
    for i in range(0, len(Y)):
        if PPV(X,Y)[i] == Y[i]:
            s+=0
        else : 
            s+=1
    print(s*100/len(Y), "%")

    
