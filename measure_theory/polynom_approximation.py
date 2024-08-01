"""Solution to the following exercise (paraphrased):
Write a program that, for a given f∈L^2([-1,1]) and given Degree n, finds the unique polynom p with
    ||f-p|| <= ||f-q|| (in the L^2-Norm)
for all Polynomials q with degree <= n. Test it for different n on the function f(x)=log(|x|)

This is done via the legendre polynomials, see https://en.wikipedia.org/wiki/Legendre_polynomials"""
#%%
import sympy as sp

x, y, z = sp.symbols('x y z')
oo = sp.oo

#%%
def dot(f, g, x):
    """The inner product of sympy functions f and g with variable x, computed symbolically using sympy"""
    return sp.integrate(f * g, (x,-1,1))

def norm(f, x):
    """The 2-norm of f, with variable x"""
    return dot(f,f,x)**(1/2)

def opt_poly(f, x, n):
    """The optimal approximation of f in the 2-norm with polynomials of at most degree n,
    which are exactly the first n summands if f is given as  a linear combination of legendre polynomials.
    (For a proof of this fact see the worksheet)"""
    leg_pol = [sp.legendre(i,x) for i in range(n+1)] #Gives the first n legendre polynomials
    coeff = [norm(leg_pol[i],x) **-2 * dot(leg_pol[i], f, x) for i in range(n+1)]
    #The coefficients for the linear combination of f in legendre polynomials are given by
    # 1/||leg_pol[i] * <leg_pol[i], f> for the i-th legendre polynomial,
    # the factor 1/||leg_pol[i] is necessary because legendre polynomials are not normalized
    return sum([coeff[i] * leg_pol[i] for i in range(n+1)])

#%%
f = sp.log(abs(x))
#f = abs(x)**1/2
#f = sp.sin(x*5)

opt = [opt_poly(f,x,n) for n in range(8)]
#%%

for pol in opt:
    print(pol)
    print(norm(pol - f, x))
# %%

sp.plot(*[i for i in opt], f, xlim=(-1,1), ylim=(-6,1))


# %%