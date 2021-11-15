
1.\\

$E(x)=(x-1)(x-2)(x-3)(x-5)\\
=(x^2-2x-x+2)(x^2-5x-3x+15)\\
=(x^2-3x+2)(x^2-8x+15)\\
=x^4-8x^3+15x^2-3x^3+24x^2-45x+2x^2-16x+30 \\
=x^4-11x^3+41x^2-61x+30 $
\\
$E'(x)=4x^3-33x^2 +82x+61$



def E(x):
    return (x-1)*(x-2)*(x-3)*(x-5)
def Eprime(x):
    return 4*x**3-33*x**2 +82*x+61

epsilon = 0.01
nb_max = 1000

def DG_E(x_0, nu): 
    L = []
    L.append(x_0)
    for i in range(nb_max) :
        a = L[-1]-nu*Eprime(L[-1])
        L.append(a)
        if epsilon > L[-1]-L[-2] :
            return L
        else :
            pass
    return L

plt.plot(DG_E(5, 0.001))
plt.plot(DG_E(5, 0.01))
plt.plot(DG_E(5, 0.1))
plt.plot(DG_E(5, 0.17))
plt.plot(DG_E(5, 1))
plt.plot(DG_E(0, 0.001))


