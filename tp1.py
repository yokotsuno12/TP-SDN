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

# Question 5, BONUS : 

def PPV_mod(k, X, Y) :
    Ypred=[]
    Ypred2= []
    for i,e in enumerate(X):
        L= metrics.pairwise.euclidean_distances(X,e[np.newaxis])
        L=L.reshape(Y.size)
        L2 = np.argsort(L)
        G = [L2[j] for j in range(1,k+1)]
        M = list(Y[G])
        Ypred.append(max(M, key=M.count))
    return np.array(Ypred)

# Classifieur Bayesien Naïf

def P(Y,i):
    a=Y.size
    b=sum(Y==i)
    return a/b

def baricentre(X, Y):
    b_s = []
    for e in np.unique(Y):
        A = X[np.where(Y == e)]
        b_s.append(np.mean(A, axis=0))
    return np.array(b_s)


def Bar2(X,Y,k) :
    G = X[np.where(Y==k)]
    H = np.mean(G, axis=0)
    return H

def d(x,X,k): #On a une donnée x, on ne connait pas son emplacement dans X, donc on ne sait pas à quelle classe elle appartient. On définit cette fonction pour calculer la distance de cette donnée avec le barycentre de la classe k. 
    b= np.linalg.norm(x - Bar2(X, Y, k))
    return b

def P2(x, X, i, k) :
    IT = np.unique(Y)
    u = Y[i]
    Q = np.sum(d(x, X, j) for j in IT)
    a = (1-d(x, X ,u))/Q
    return a
    
def CBN(X,Y) : 
    IT = np.unique(Y)
    T =[]
    for j in range (0,len(Y)):
        L = [P(Y, Y[j])*P2(X[j], X, i, Y[j]) for i in range(0,len(Y)) ]
        T.append(Y[argmax(L)])
    return T
print(CBN(X,Y))    

# Question 2:

def Erreur3(X,Y):
    L = CBN(X,Y)
    N = sum(L==Y)
    return ((L.size-N)/L.size)*100 

print('\n')
print('CBN pour iris :\n',CBN(X,Y),'\n')
print('Erreur CBN pour iris\n',Erreur3(X,Y),'\n')

# Question 3 : 

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, Y)
MultinomialNB()
print(clf.predict(X))


