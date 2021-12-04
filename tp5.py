# TP5 - SDN

#### Ce TP présente ce qu’est une Descente de Gradient (DG). La DG s’applique lorsquel’on cherche le minimum d’une fonction dont on connaît l’expression analytique, qui estdérivable, mais dont le calcul direct du minimum est difficile. C’est un algorithme fondamental à connaître car utilisé partout sous des formes dérivées. Nous n’étudions ici que la version de base.

## Descente de gradient

#### Ce paragraphe présente la DG sur la minimisation de la fonction E(x)quelconque. Le problème est de trouver la valeur de x qui minimise E(x). Pour trouver analytiquement le minimum de la fonction E, il faut trouver les racines de l’équation E′(x) = 0, donc trouver ici les racines d’un polynôme de degré 3, ce qui est des fois “difficile". Donc on va utiliser la DG. La DG consiste à construire une suite de valeurs xi(avec x0 fixé au hasard) de manière itérative : xi+1=xi−ηE′(xi).On peut donner un critère de fin à la DG par exemple si xi+1−xi< epsilon ou si i > nombremax. Pour ce problème, nous utilisons epsilon = 0.01 et nombremax = 1000.

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.datasets import make_regression


# 1.\\
# $E(x)=(x-1)(x-2)(x-3)(x-5)\\
# =(x^2-2x-x+2)(x^2-5x-3x+15)\\
# =(x^2-3x+2)(x^2-8x+15)\\
# =x^4-8x^3+15x^2-3x^3+24x^2-45x+2x^2-16x+30 \\
# =x^4-11x^3+41x^2-61x+30 $
# \\
# $E'(x)=4x^3-33x^2 +82x-61$


def E(x):
    return (x-1)*(x-2)*(x-3)*(x-5)


def Eprime(x):
    return 4*x**3 - 33*x**2 + 82*x - 61


epsilon = 0.01
nb_max = 1000


def DG_E(x_0, nu):
    L = []
    L.append(x_0)
    for i in range(nb_max):
        a = L[-1] - nu*Eprime(L[-1])
        L.append(a)
        if epsilon > abs(L[-1] - L[-2]):
            return L
        else:
            pass
    return L


X = np.arange(0.5, 5.3, 0.01)
min_locaux_X = []
Y_Eprime = Eprime(X)

for i, (a, b) in enumerate(zip(Y_Eprime[1:], Y_Eprime[:-1])):
    if a > 0 and b < 0 or a == 0:
        min_locaux_X.append(X[i])
min_locaux_X = np.array(min_locaux_X)

fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(X, E(X))
plt.scatter(min_locaux_X, E(min_locaux_X), marker="X", c="red")
for x in min_locaux_X:
    ax.annotate('x = %s' % round(x, 2), xy=(x, E(x)-0.5), textcoords='data', ha="left")
plt.title("Visualisation de la fonction E")
plt.xlabel("X")
plt.ylabel("E(X)")
plt.show()

DG_a = DG_E(5, 0.001)
DG_b = DG_E(5, 0.01)
DG_c = DG_E(5, 0.1)
DG_d = DG_E(5, 0.17)
#DG_e = DG_E(5, 1)
DG_f = DG_E(0, 0.001)
visualisation = [DG_a, DG_b, DG_c, DG_d, DG_f]

# Autre manière pour les visualiser toutes d'un coup ;)
i = 0
plt.figure()
plt.figsize = (20, 20)
for j in visualisation:
    i += 1
    plt.subplot(1, 6, i)
    plt.scatter(list(range(len(j))), j, color='blue')
    plt.plot(list(range(len(j))), j, color='red')
    plt.xlabel("epoch")
    plt.ylabel("x trouvé")
    if visualisation.index(j) == 0:
        plt.title('DG_E(5, 0.001) ' + str(len(j)) + " iterations")
    elif visualisation.index(j) == 1:
        plt.title('DG_E(5, 0.01) ' + str(len(j)) + " iterations")
    elif visualisation.index(j) == 2:
        plt.title('DG_E(5, 0.1) ' + str(len(j)) + " iterations")
    elif visualisation.index(j) == 3:
        plt.title('DG_E(5, 0.17) ' + str(len(j)) + " iterations")
    elif visualisation.index(j) == 4:
        plt.title('DG_E(5, 1) ' + str(len(j)) + " iterations")
    else:
        plt.title('DG_E(0, 0.001) ' + str(len(j)) + " iterations")

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=3.5,
                    top=1,
                    wspace=0.4,
                    hspace=0.4)

for j in visualisation:
    print(len(j))

plt.plot(list(range(len(DG_a))), DG_a, color='blue')
plt.title("x0 = 5 et nu = 0.001")
plt.xlabel("epoch")
plt.ylabel("x trouvé")
plt.show()

plt.plot(list(range(len(DG_b))), DG_b, color='green')
plt.title("x0 = 5 et nu = 0.01")
plt.xlabel("epoch")
plt.ylabel("x trouvé")
plt.show()

plt.plot(list(range(len(DG_c))), DG_c, color='yellow')
plt.title("x0 = 5 et nu = 0.1")
plt.xlabel("epoch")
plt.ylabel("x trouvé")
plt.show()

plt.plot(list(range(len(DG_d))), DG_d, color='orange')
plt.title("x0 = 5 et nu = 0.17")
plt.xlabel("epoch")
plt.ylabel("x trouvé")
plt.show()

#plt.plot(list(range(len(DG_e))), DG_e, color='red')
#plt.show()

plt.plot(list(range(len(DG_f))), DG_f, color='purple')
plt.title("x0 = 0 et nu = 0.001")
plt.xlabel("epoch")
plt.ylabel("x trouvé")
plt.show()


#### Maintenant, on va tester tout ça pour des valeurs différentes :

for nu in (10**-i for i in reversed(range(1, 4))):
    plt.figure()
    plt.figsize = (20, 20)
    for j in range(5):
        j += 1
        plt.subplot(1, 6, j)
        plt.scatter(list(range(len(DG_E(j, nu)))), DG_E(j, nu), color='blue')
        plt.plot(list(range(len(DG_E(j, nu)))), DG_E(j, nu), color='red')
        plt.xlabel("epoch")
        plt.ylabel("x trouvé")
        plt.title('DG_E(' + str(j) + ',' + str(nu) + ") / itérations :" + str(len(DG_E(j, nu))))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=3.5,
                        top=1,
                        wspace=0.4,
                        hspace=0.4)

# i=0
# plt.figure()
# plt.figsize = (20, 20)
# for j in range(5):
#     i += 1
#     plt.subplot(1, 6, i)
#     plt.scatter(list(range(len(DG_E(j,0.001)))), DG_E(j,0.001), color='blue')
#     plt.plot(list(range(len(DG_E(j,0.001)))), DG_E(j,0.001), color='red')
#     plt.xlabel("epoch")
#     plt.ylabel("x trouvé")
#     plt.title(j)
# plt.subplots_adjust(left=0.1, 
#                     bottom=0.1,  
#                     right=3.5,  
#                     top=1,  
#                     wspace=0.4,  
#                     hspace=0.4)

# i=0
# plt.figure()
# plt.figsize = (20, 20)
# for j in range(5):
#     i += 1
#     plt.subplot(1, 6, i)
#     plt.scatter(list(range(len(DG_E(j,0.01)))), DG_E(j,0.01), color='blue')
#     plt.plot(list(range(len(DG_E(j,0.01)))), DG_E(j,0.01), color='red')
#     plt.xlabel("epoch")
#     plt.ylabel("x trouvé")
#     plt.title(j)
# plt.subplots_adjust(left=0.1, 
#                     bottom=0.1,  
#                     right=3.5,  
#                     top=1,  
#                     wspace=0.4,  
#                     hspace=0.4)

# i=0
# plt.figure()
# plt.figsize = (20,20)
# for j in range(5) : 
#     i+=1
#     plt.subplot(1,6,i)
#     plt.scatter(list(range(len(DG_E(j,0.1)))), DG_E(j,0.1), color='blue')
#     plt.plot(list(range(len(DG_E(j,0.1)))), DG_E(j,0.1), color='red')
#     plt.xlabel("epoch")
#     plt.ylabel("x trouvé")
#     plt.title(j)
# plt.subplots_adjust(left=0.1, 
#                     bottom=0.1,  
#                     right=3.5,  
#                     top=1,  
#                     wspace=0.4,  
#                     hspace=0.4)

# i=0
# plt.figure()
# plt.figsize = (20,20)
# for j in range(5) : 
#     i+=1
#     plt.subplot(1,6,i)
#     plt.scatter(list(range(len(DG_E(j,0.1)))), DG_E(j,0.1), color='blue')
#     plt.plot(list(range(len(DG_E(j,0.1)))), DG_E(j,0.1), color='red')
#     plt.xlabel("epoch")
#     plt.ylabel("x trouvé")
#     plt.title(j)
# plt.subplots_adjust(left=0.1, 
#                     bottom=0.1,  
#                     right=3.5,  
#                     top=1,  
#                     wspace=0.4,  
#                     hspace=0.4)

#### !! Attention aux échelles !! Il faut bien regarder 

# DESCENTE DE GRADIENT POUR LA REGRESSION LINEAIRE


## Descente de gradient pour la régression linéaire

def Ychapeau(X, a, b) : 
    Ychap = []
    for i in range(len(X)):
        Ychap.append(a * X[i] + b)
    return Ychap


def F(X, Y, a, b):
    s = 0
    for i in range(len(X)):
        # (ax_i -b - y_i)^2 = (ax_i)^2 -2*a*x_i(b+y_i) +(b+y_i)^2
        s += (Ychapeau(X, a, b)[i] - Y[i])**2
    return s


def F_prim_a(X,Y,a,b) :
    s = 0
    for i in range(len(X)):
        s += 2 * (a * X[i]**2 - X[i] * (b + Y[i]))
    return s


def F_prim_b(X,Y,a,b) :
    s = 0
    for i in range(len(X)):
        s += 2 * (b * Y[i]**2 - a * X[i])
    return s


def DG_F(X,Y,a_0, b_0, nu): 
    A = []
    B = []
    C = [(a_0, b_0)]
    A.append(a_0)
    B.append(b_0)
    for i in range(1, nb_max + 1):
        a = A[-1] - nu * F_prim_a(X, Y, A[-1], B[-1])
        b = B[-1] - nu * F_prim_b(X, Y, A[-1], B[-1])
        A.append(a)
        B.append(b)
        C.append((a, b))
        if epsilon > distance.euclidean((A[-1], B[-1]), (A[-2], B[-2])):
            return C
        else:
            pass
    return np.array(C)
#### Note pour les autres : Ici on va rencontrer un problème : Dès qu'on va changer la valeur de nb_max (en 500 par ex), d'epsilon (en 0.1 par ex je crois) ou de nu, l'algorithme ne va pas marcher. Je ne comprends pas encore pourquoi (ici Pépita). Ca m'affiche "array must not contain infs or NaNs". :/ Pb avec la distance euclidienne, je crois. 


nb_max = 100
epsilon = 0.01

X, Y = make_regression(n_features=1)
X = np.concatenate(X)
C = DG_F(X,Y, 1,1, 0.001)

plt.figure()
plt.title("evolution de a")
plt.scatter(X, Y)
plt.plot(np.arange(len(C)), C[:,0], color='RED')
plt.show()

plt.figure()
plt.title("evolution de b")
plt.scatter(X, Y)
plt.plot(np.arange(len(C)), C[:,1], color='RED')
plt.show()
