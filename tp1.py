
from numpy import *
from sklearn import *
import sklearn.preprocessing as sp #pour iris 

iris = datasets.load_iris()
X = iris.data
Y = iris.target

def PPV(X,Y):
    R= 0,8*X.shape[0] #Nombre de données reference
    T =0,2*X.shape[0] #Nombre de données test
    
