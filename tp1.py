
from numpy import *
from sklearn import *
import sklearn.preprocessing as sp #pour iris 

iris = datasets.load_iris()
X = iris.data
Y = iris.target

def PPV(X,Y):
    R= 0,8*X.shape[0] #Nombre de données reference
    Ref = X[R:]
    Test= X[:R]
    for i in Test:
        L= metrics.pairwise.euclidean_distances(Ref,i)
        argmin(L)
        Ypred.append( Y[argmin])
        


def PPV2(X,Y):
    s=0
    for i in range(0, len(Y)):
        if PPV(X,Y)[i] == Y[i]:
            s+=0
        else : 
            s+=1
    print(s*100/len(Y), "%")

    
