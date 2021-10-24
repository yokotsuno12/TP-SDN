from sklearn import metrics, datasets
import numpy as np


iris = datasets.load_iris()
X = iris.data
Y = iris.target

# 1)
def PPV(X,Y):
    """
    Evalue l'algorithme du Plus Proche Voisin avec la cross validation.

    Parameters
    ----------
    X : numpy.ndarray
        Une matrice de dimension 2 où chaque ligne correspond à une valeur
    Y : array-like
        Les labels

    Returns
    -------
    numpy.array
        Les predictions

    """
    Ypred = []
    for i, e in enumerate(X):
        L = metrics.pairwise.euclidean_distances(X, e[np.newaxis])
        L = np.delete(L, i)
        Ypred.append(Y[np.argmin(L)])
    return np.array(Ypred)


# 2)

def Erreur(X, Y):
    """
    Calcule le nombre de valeur pour lesquelles X est different de Y.

    Parameters
    ----------
    X : numpy.array
        Le premier vecteur
    Y : numpy.array
        Le second vecteur

    Returns
    -------
    float
        Le pourcentage de valeurs différentes

    """
    N = X.size - sum(X == Y)
    return N / X.size * 100


def ErreurPPV(X, Y):
    return Erreur(PPV(X, Y), Y)


def PPV3(Ychapeau,Y) :
    """
    Un autre algorithme pour calculer l'erreur
    """
    s = 0
    for i in range(0, len(Y)):
        if Ychapeau[i] == Y[i]:
            s += 0
        else:
            s += 1
    return s*100 / len(Y)


# 3)

print('PPV pour iris :\n', PPV(X, Y), '\n')
print('Erreur PPV pour iris\n\t', ErreurPPV(X, Y), '\n')

# Question 4 :

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, Y)
Ychapeau = neigh.predict(X)
print(Ychapeau)

print(Erreur(Ychapeau, Y))
# On obtient 0% parce que l'algorithme KN prend comme PPV d'une donnée
# la donnée elle-même (de distance 0 par rapport à elle-même)

print(ErreurPPV(X, Y))
# En revanche, dans notre algorithme, nous avons supprimé la possibilité
# de prendre comme PPV d'une donnée la donnée elle-même
# (en supprimant la diagonale de la matrice des distances).

neigh2 = KNeighborsClassifier(n_neighbors=2)
neigh2.fit(X, Y)
Ychapeau2 = neigh2.predict(X)
print(Ychapeau2)

print(Erreur(Ychapeau2, Y))
neigh3 = KNeighborsClassifier(n_neighbors=3)
neigh3.fit(X, Y)
Ychapeau3 = neigh3.predict(X)
print(Ychapeau3)
print(Erreur(Ychapeau3, Y))

# Question 5, BONUS :


def PPV_mod(k, X, Y):
    """
    Implémentation de l'algorithme des k plus proches voisins.

    Parameters
    ----------
    k : int
        Le nombre de voisins à prendre en considération
    X : numpy.ndarray
        Une matrice de dimension 2 où chaque ligne correspond à une valeur
    Y : array-like
        Les labels

    Returns
    -------
    numpy.array
        Les predictions
    """
    Ypred = []
    for i, e in enumerate(X):
        L = metrics.pairwise.euclidean_distances(X, e[np.newaxis])
        L = L.reshape(Y.size)
        L2 = np.argsort(L)
        G = [L2[j] for j in range(1, k+1)]
        M = list(Y[G])
        Ypred.append(max(M, key=M.count))
    return np.array(Ypred)

# Classifieur Bayesien Naïf


def baricentre(X, Y):
    """
    Baricentre de toutes les classes.
    Calcule le baricentre de toutes les classes comme étant la moyenne des
    points composant cette classe.

    Parameters
    ----------
    X : numpy.ndarray
        Tableau à deux dimension, chaque ligne est un point
    Y : numpy.array
        Les labels

    Returns
    -------
    numpy.array
        La liste des baricentre de chaque classes (l'ordre des classes est 
        donnée par la fonction numpy.unique).

    """
    b_s = []
    for k in np.unique(Y):
        A = X[np.where(Y == k)]
        b_s.append(np.mean(A, axis=0))
    return np.array(b_s)


def d(a, b):
    """
    Calcule la distance entre deux point.

    Parameters
    ----------
    a : numpy.array
        Un vecteur representant le premier point
    b : numpy.array
        Un vecteur representant le second point

    Returns
    -------
    float
        La distance euclidienne entre les deux points.

    """
    return np.linalg.norm(a - b)


def P1(Y):
    """
    Pour chaque classe k dans Y, calcule la probabilité d'obtenir un élement
    de la classe k si on tire un element au hasard.

    Parameters
    ----------
    Y : numpy.array
        Un vecteur représentant les labels

    Returns
    -------
    p_s : numpy.array
        La liste des probabilités pour toutes les classes.

    """
    p_s = []
    for e in np.unique(Y):
        p = sum(Y == e)
        p_s.append(p/Y.size)
    return p_s


def P2(x, k, baricentre):
    """
    Calcule la probabilité d'avoir le point x sachant qu'il appartient à
    la classe k.

    Parameters
    ----------
    x : numpy.array
        Un vecteur représentant point x
    k : int
        La classe à laquelle appartient le point x
    baricentre : numpy.array
        Un vecteur representant les baricentres de toutes les classes.

    Returns
    -------
    a : float
        Un nombre entre 0 et 1 représentant la probabilité.

    """
    sum_dist = np.sum(d(x, bar) for bar in baricentre)
    a = 1 - (d(x, baricentre[k]) / sum_dist)
    return a


def CBN(X, Y):
    """
    Implémente L'agorithme du classifieur bayesien Naif.

    Calcule la prédiction pour chaque données de X en fonction de toutes les
    autres.

    Parameters
    ----------
    X : numpy.array de dimension (num_sample, num_features)
        Les données
    Y : numpy.array
        Vecteur représentant les labels

    Returns
    -------
    numpy.array
        Les prédictions de tous les points en prenant en données
        d'entrainement tous les autre points.

    """
    ls_p = P1(Y)
    ls_baricentre = baricentre(X, Y)
    classes = np.unique(Y)

    T = []
    for j in range(0, len(Y)):
        L = [ls_p[k-1] * P2(X[j], k, ls_baricentre) for k in classes]
        T.append(classes[np.argmax(L)])

    return np.array(T)


print(CBN(X, Y))

# Question 2:


def ErreurCBN(X, Y):
    return Erreur(CBN(X, Y), Y)


print()
print('CBN pour iris :\n', CBN(X, Y), '\n')
print('Erreur CBN pour iris\n', ErreurCBN(X, Y), '\n')

# Question 3 :

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
# GaussianNB()
K = clf.predict(X)
CBN(X, Y) == K
# la réponse est non!!! Ce n'est pas la même manière de calculer
# des probabilités!! Donc pas le même résultat


def Erreur4(X, Y):
    L = K
    N = sum(L == Y)
    return ((len(L)-N) / len(L)) * 100


print()
print('classificateur gaussien pour iris :\n', K, '\n')
print('Erreur classificateur gaussien pour iris\n', Erreur4(X, Y), '\n')
