"""Solution to a rather long-winded exercise that in essence reconstructs
Bertrands paradoxon (https://en.wikipedia.org/wiki/Bertrand_paradox_(probability)) in a measure-theoretic sense.
Here we define the 3 appropriate measures and see that the measure of the inner triangle is numerically actually equal to 1/3, 1/2 and 1/4, respectively.
"""
# %%
import numpy as np
import sympy as sp
import scipy as scp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# %%

def plot3dgraph(func, xrange, yrange, num=200):
    ax = plt.figure().add_subplot(projection='3d')
    X = np.repeat(np.linspace(*xrange, num=num)[:,np.newaxis], num, axis=1)
    Y = np.repeat(np.linspace(*yrange, num=num)[:,np.newaxis], num, axis=1).T
    Z = func(X, Y)

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    ax.contour(X, Y, Z, zdir='z', offset=np.nanmin(Z), cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='x', offset=xrange[0], cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='y', offset=yrange[1], cmap='coolwarm')

    #ax.set(xlim=xrange, ylim=yrange, zlim=(-1000,1000),

    ax.set(xlim=xrange, ylim=yrange, zlim=(np.nanmin(Z), np.nanmax(Z)),
        xlabel='X', ylabel='Y', zlabel='Z')

    plt.show()

# %%
def f1(x,y):
    return np.where(x**2 + y**2 > 1, 0 , x/x * 1/np.pi)

def f2(x,y):
    return np.where(x**2 + y**2 > 1, np.nan, 1/(2*np.pi) * 1/np.sqrt(x**2+y**2))

x, y, z, u, v = sp.symbols('x y z u v', positive=True)
    #the symbols technically wont be all positive, but it turns that we only care about absolut values in the end,
    #and the simplifications work better and give more numerically stable results like this
oo = sp.oo

a1 = sp.atan(y/x) + sp.acos(y/(sp.sin(sp.atan(y/x))))
a2 = sp.atan(y/x) - sp.acos(y/(sp.sin(sp.atan(y/x))))
 #φ_3 is nearly bijective, since every midpoint is generated from exactly one pair of points, only the order varies
 #So there exists a local inverse function φ^-1 almost everywhere, and we just need to fix a factor of 2 somewhere

 #therefore we can use the fact that D(φ^-1)(x) = (D φ)^-1 (φ^-1(x)), since an inverse of φ exists locally
j = 1/abs(sp.Matrix.jacobian(sp.Matrix([1/2 * (sp.cos(u)+sp.cos(v)), 1/2 * (sp.sin(u)+sp.sin(v))]), sp.Matrix([u,v])).det().subs(u, a1).subs(v, a2))
 #This  is |det(φ^-1(x,y))|
j = j.simplify()
f3 = sp.lambdify([x,y], j)
#%%
plot3dgraph(f1, (-1,1), (-1,1), num=200)

# %%
plot3dgraph(f2, (-1,1), (-1,1), num=200)

# %%
plot3dgraph(f3, (-1,1), (-1,1), num=300)

# %%
def restrict_domain(f, r):
    """Sets the function f in two variables to 0 outside of a disk with center 0 and radius r"""
    def restricted_f(x,y):
        if np.sqrt(x**2+y**2) > r:
            return 0
        return f(x,y)
    return restricted_f

#For quadrature we need to avoid division by zero, but dont need the function to be numpy-compatible
def f1_quad(x,y):
    return 1/np.pi 

def f2_quad(x,y):
    if np.sqrt(x**2+y**2) < 1e-15 :
        return 0
    return 1/(2*np.pi) * 1/np.sqrt(x**2+y**2)

def f3_quad(x,y):
    if (x**2 + y**2) < 1e-18:
        return 0
    return 1/(2*(np.pi**2)) * f3(x,y)
            # Factor 2 compared to definition of the measure because we need to adjust for φ_3 not being bijektive,
            # every midpoint gets pointed to exactly twice

#%%
for f, label in [(f1_quad, "φ_1"), (f2_quad, "φ_2"), (f3_quad, "φ_3")]:
    f_whole = restrict_domain(f, 1)
    res = scp.integrate.dblquad(f_whole, -1, 1, -1, 1)[0]
    print(f"∫{label} over B_1 = {res}")

    f_half = restrict_domain(f, 1/2)
    res = scp.integrate.dblquad(f_half, -1, 1, -1, 1)[0]
    print(f"∫{label} over B_1/2 = {res}\n")


# %%
