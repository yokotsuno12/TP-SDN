
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

def PPV2(X,Y):
    s=0
    for i in range(0, len(Y)):
        if PPV(X,Y)[i] == Y[i]:
            s+=0
        else : 
            s+=1
    print(s*100/len(Y), "%")

# Question 3 :     
#print(PPV(X,Y))
#print(PPV2(X,Y))

# Question 4 : 

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X,Y)
Ychapeau = neigh.predict(X)
print(Ychapeau)








