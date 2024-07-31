"""Solution to the following exercise (paraphrased):

Let p ∈ [1, ∞), Write a program that for a given function f ∈ L^p([0,1)) and a given error ε ∈ (0, ∞)
returns a periodic smooth function g with period 1 and ||f-g|| < ε with ||.|| being the L^p-norm.
Test the code with the function
    f: [0,1) -> R, f(x) :=  (x-1/2)^(-1/2p)  for x > 1/2
                            1/2 - x          for x <= 1/2

The solution uses the fact that the convolution of a periodic function with a smooth function gives a smooth periodic function under the right constraints,
and that, when the smooth function gets scaled appropriately, the convolution approaches the periodic function in L^p"""

#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#%%

def make_periodic(f):
    """"Transforms a given fuction [0,1] -> R into a periodic function R -> R"""
    return lambda x: f(x - np.floor(x))


#%%
def lp_norm(f, p, i=(0,1)):
    """Computes the (approximate) L^p-Norm of a given function on the given intervall"""
    return sp.integrate.quad(lambda x:np.abs(f(x))**p, i[0], i[1])[0] ** (1/p)

def approximation(f, g, epsilon, detail=1000):
    """Gives the approximation of f, using the convolution for example 4.
    If f is periodic and g C_c^infty, then the result is both periodic, and smooth.
    Note: the result can only be computed pointwise, not with a numpy array."""
    #g_epsilon = lambda x: 1/epsilon * g(x/epsilon)
    f_linspace = np.linspace(0, epsilon, detail)
    g_linspace = np.linspace(0,1,detail)
    #return np.vectorize(lambda x: sp.integrate.quad(lambda y:(f(x) - y)*g_epsilon(y), 0, epsilon)[0])
    #return np.vectorize(lambda x: sp.integrate.quadrature(lambda y:(f(x) - y)*g_epsilon(y), 0, epsilon, tol=1e-5)[0])
    return lambda x: 1/epsilon *np.trapz(f(x - f_linspace) * g(f_linspace/epsilon),f_linspace)

def find_approx(f, g, p, max_norm, detail=1000):
    approx = lambda x:0
    k = 1
    current_norm = lp_norm(lambda x:f(x) - approx(x), p)
    while current_norm > max_norm:
        approx = approximation(f, g, 1/k, detail)
        k *= 2
        current_norm = lp_norm(lambda x:f(x) - approx(x), p)
        print(current_norm)
    return approx

#%%
def g(x):
    """C_c^oo function that is used as one of the factors for the convolution, and is nonzero only in (0,1)"""
    integ = 0.22199690808403932 #Integral value of not normalized g, used to achieve an integral of 1 for g
    if type(x) == float:
        if 1e-15 < x < 1 - 1e-15:
            val = np.exp(1/((2*x-1)**2-1))
        else:
            val = 0
        print("Using g as float function!")
        return val/integ
    else:
        mask = np.logical_and(1e-15 < x, x < 1 - 1e-15)
        val = np.zeros(x.shape)
        val[mask] = np.exp(1/ ((2*x[mask]-1)**2 - 1))
        return val/integ


p = 2
max_norm = 0.5
f = lambda x: np.where(x > 1/2 + 1e-15, np.power(x-1/2, -1/(2*p),where=(x > 1/2 + 1e-15)), 1/2-x)
f = make_periodic(f)

f_approx = find_approx(f, g, p, max_norm)

#%%
linspace = np.linspace(0.75,1.25, 1000)
plt.plot(linspace, [f_approx(i) for i in linspace], '.',label="approx")
plt.plot(linspace, f(linspace), '',label="f")
#plt.plot(linspace, g(linspace), label="g")
plt.legend()
#plt.axis([None, None, -2,10])

# %%
