"""Solution for the following exercise (paraphrased):
Let μ be a finite Measure on [0, 2π) of the form
    μ = fλ + aδ_z,
where λ is the Lebesgue-Measure, f: R -> [0,∞) a C¹ 2π-periodic Function, a a nonnegative real number and δ_z the Dirac-Measure at z (z between 0 and 2π)
Write a program that reconstructs the parameters a and z and the function f from the Fourier-Coefficients
μ_k := Integral_0^2π e^(-ikz)dμ(x)

The solution first constructs those coefficents from a given function and given parameters, and then reconstructs the parameters using fast fourier transforms."""

#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

PADD = 10_000
x, y, z = sp.symbols('x y z')
#%%

def get_coeff(f, z, a, max_k):
    """Constructs the first max_k μ_k from given f, a, z as given in the excercise."""

    s = sp.fourier_series(f, (x, 0, 2*sp.pi))

    real_fourier = np.array([1] + [s.an.coeff(i).subs(x,0).evalf() for i in range(1, max_k)])
    imag_fourier = np.array([s.a0] + [s.bn.coeff(i).subs(x, sp.pi/(2*i)).evalf() for i in range(1, max_k)])
    z_k = a * np.array([np.exp(1j *i * z) for i in range(0, max_k)])

    coeff = np.array(z_k + real_fourier + 1j * imag_fourier, dtype=np.complex_)
    return coeff

def reconstruct(coeff):
    """Reconstructs an approximation of the function f, and the parameters a and z from the given coefficients"""
    fft = np.fft.fft(coeff, max_k*PADD)
                                #^ padding the array with a big constant PADD allows for better approximation of the actual peak
    z_guess = 2*np.pi *np.argmax(fft)/(max_k*PADD) #The location of the biggest peak of the fourier transform should be the period of the e^ikz part
    a_guess = np.max(np.abs(fft))/max_k            # And its magnitude is the value of a
    s_guess = coeff - a*np.exp(1j * z_guess * np.arange(len(coeff)))
    s_guess = s_guess[:int(np.floor(max_k/2))] #since the guess for z is not exact,
                                    #there is a sort of diffraction pattern, and the fourier series doesnt exactly converge to zero like it should

    f_guess = lambda x: np.imag(coeff[0]) -1  + np.sum([np.real(s_guess[i]) * np.cos(i*x) + np.imag(s_guess[i]) * np.sin(i*x) for i in range(len(s_guess))], 0)
    #                       ^ The first of the coefficients is constant, and only formally in the imaginary part, since sin(0)=0
    return f_guess, a_guess, z_guess

# %%
#### This defines the function used for the reconstruction and builds the μ_k terms
f = (x-sp.pi)**2 
#f = sp.exp(-(x-sp.pi)**2)

#f = x**2       #Not continuous and periodic, since f(0) != f(2pi)
#f = sp.exp(x)  #They still work, but the approximations are a lot less good


max_k = 50 #number of μ_k terms that are constructed
z = 1.5
a = 1

coeff = get_coeff(f, z, a, max_k)

#### Actual reconstruction of the function happens here
f_guess, a_guess, z_guess = reconstruct(coeff)

#%%
#### Show results in a plot
lnsp = np.linspace(0, 2* np.pi, 1000)
plt.plot(lnsp, f_guess(lnsp), linewidth=2, label="Approximation for f")
plt.plot(lnsp, sp.lambdify(x, f)(lnsp), linewidth=1, label="f")
plt.legend()
plt.show()

print(f"The guess for z was {z_guess:.5f}, vs real value {z:.5f}")
print(f"The guess for a was {a_guess:.5f}, vs real value {a:.5f}")

#%%
#########################################################################
#Those are just some test I used for figuring out the way the reconstruction could work
#########################################################################
def test_for_magnitude():
    """Proof of concept for finding a, the magnitute of (the periodic part of) a function"""
    f = lambda x, k: np.cos(k*x) +1j* np.sin(k*x)
    l = int(np.floor(np.pi* 10))
    lnsp = np.linspace(0,l, l)
    alist = np.linspace(0,10, 11)
    z=0.5
    fft = [np.fft.fft(a*f(z, lnsp), l*1000) for a in alist]
    for i in range(11):
        print((max(abs(fft[i])))/(l), "is magnitude for a", alist[i])
        plt.plot(np.arange(31000), fft[i], label=alist[i])
    plt.legend()
    plt.show()
# %%
def test_for_period():
    """Proof of concept for finding z, the (most prominent) period of a function"""
    f = lambda x, k: np.cos(k*x) +1j* np.sin(k*x)
    l = int(np.floor(np.pi* 10))
    lnsp = np.linspace(0,l, l)
    zeds = np.linspace(0,1, 11)
    fft = [np.fft.fft(f(z, lnsp), l*1000) for z in zeds]

    for i in range(11):
        print((2*np.pi *np.argmax(fft[i])/(l*1000)), "is period for z", zeds[i])
        plt.plot(np.arange(31000), fft[i], label=zeds[i])
    plt.legend()
    plt.show()
