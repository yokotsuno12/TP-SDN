from sklearn import*
import matplotlib.pyplot as plt
import numpy as np


iris = datasets.load_iris()
X = iris.data
Y = iris.target

"Plus Proche Voisin"

"1)"
def PPV(X,Y):
    Ypred=[]
    for i,e in enumerate(X):
        L= metrics.pairwise.euclidean_distances(X,e[np.newaxis])
        L = np.delete(L,i)
        Ypred.append(Y[np.argmin(L)])
    return np.array(Ypred)


"2)"

def Erreur(X,Y):
    L = PPV(X,Y)
    N = sum(L==Y)
    return ((L.size-N)/L.size)*100 


"3)"
print('\n')
print('PPV pour iris :\n',PPV(X,Y),'\n')
print('Erreur PPV pour iris\n',Erreur(X,Y),'\n')

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

neigh2 = KNeighborsClassifier(n_neighbors=2)
neigh2.fit(X,Y)
Ychapeau2 = neigh2.predict(X)
print(Ychapeau2)
print(PPV3(Ychapeau2,Y))
neigh3 = KNeighborsClassifier(n_neighbors=3)
neigh3.fit(X,Y)
Ychapeau3 = neigh3.predict(X)
print(Ychapeau3)
print(PPV3(Ychapeau3,Y))

