
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


def PPV3(Ychapeau,Y) : 
    s=0
    for i in range(0, len(Y)):
        if Ychapeau[i] == Y[i]:
            s+=0
        else : 
            s+=1
    print(s*100/len(Y), "%")

PPV3(Ychapeau,Y) #On obtient 0% parce que l'algorithme KN prend comme PPV d'une donnée la donnée elle-même (de distance 0 par rapport à elle-même)
PPV3(PPV(X,Y),Y) #En revanche, dans notre algorithme, nous avons supprimé la possibilité de prendre comme PPV d'une donnée la donnée elle-même (en supprimant la diagonale de la matrice des distances).




