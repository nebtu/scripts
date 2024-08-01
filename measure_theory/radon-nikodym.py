"""Solution to the following exercise:

Let λ be the lebesgue-measure on the intervall (0,1]. Calculate for a given monotonously increasing
and right-continuous functiono F: [0,1] -> R the Radon-Nikody-derivative of the absolutely continously part of the masure induced by F.
Test your code on the function 
    F(x) = c(x) for x ∈ [0,1/2) (the cantor function)
           |x+3/4| + 2x^2 for x ∈ [1/2, 1]

This derivative is calculated with simple numerical approximation, not derivable parts are simply taken away as outlier values."""
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
RECURSION_DEPTH = 30

def cantor(x, n=RECURSION_DEPTH):
    if n == 0:
        return x
    elif x <= 1/3:
        return 1/2 * cantor(3*x, n-1)
    elif 1/3 < x < 2/3:
        return 1/2
    else:
        return 1/2 + 1/2 * cantor(3*x - 2, n-1)

def f(x):
    if x < 1/2:
        return cantor(x)
    else:
        return np.abs(x-3/4) + 2 * x**2


# %%

def radon_nikodym(f, lnsp, diff):
    d = np.array([(f(x+diff) - f(x-diff))/ (2*diff) for x in lnsp])
    #note that this is just (numerically computed) derivative of f
    #it is however the same as the radon-nikodyn-derivative almost everywhere according to Bem 2.1.11
    d = np.where(d < 10000, d, 0) #big values are probably non-differantiable points
                                  #in praxis this takes good enough care of the singular points of F
                                  #although one could ofc construct a steep enough function where points are wrongly ignored
    return d

#%%

diff = 1e-12

lnsp = np.linspace(0,1, 1000)
deriv = radon_nikodym(f, lnsp, diff)

#%%

plt.plot(lnsp, [f(x) for x in lnsp], label="$F$")
plt.plot(lnsp, deriv, label="$\\frac{d\mu_A}{dλ}$")
plt.legend()
