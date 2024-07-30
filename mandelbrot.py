"""A short scripts which provides a function to generate a part of the mandelbrot fractal at some given coordinates.
The colors are taken from the gruvbox colorin scheme"""

#%%
import numpy as np
from PIL import Image

# COLORS = np.array([(251, 73,  52),
#           (184, 187, 38),
#           (250, 189, 47),
#           (131, 165, 152),
#           (211, 134, 155),
#           (142, 192, 124),
#           (213, 196, 161),
#           (102, 92,  84),
#           (0,  0,  0)])
COLORS = np.array([(251, 73,  52),
                   (251, 73,  52),
          (250, 189, 47),
          (131, 165, 152),
          (211, 134, 155),
          (142, 192, 124),
          (213, 196, 161),
          (102, 92,  84),
          (0,  0,  0)])
COLORS = COLORS.astype(np.int8)

#%%

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter):
    a = np.stack((np.linspace(ymin, ymax, width),)*height, axis=0)
    b = np.stack((np.linspace(xmax, xmin, height),)*width, axis=1)
    z = a + 1j * b
    c = z[::]
    iter = np.zeros((height, width))
    for i in range(1, maxiter):
        z = z**2 + c
        iter = np.where(np.logical_and(np.abs(z) > 2, iter==0), i, iter)
    iter = np.where(iter==0, maxiter, iter)
    return iter 

#%%

def mandelbrot_image(x, y, radius, width, height, maxiter):
    if height < width:
        width_distance = (radius * width * 0.5) / height #Distance from Center to left/right side
        height_distance = radius * 0.5
    else:
        height_distance = (radius * height * 0.5) / width #Distance from Center up/down side
        width_distance = radius * 0.5
    xmin = x - width_distance
    xmax = x + width_distance
    ymin = y - height_distance
    ymax = y + height_distance
    mandelbrot = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter)
    mandelbrot_colors = COLORS[np.round((mandelbrot / maxiter) * (len(COLORS)-1)).astype(int)]
    im = Image.fromarray(mandelbrot_colors, 'RGB')
    return im
# %%

mandelbrot_image(0.,-0.7,2.8, 512, 512, 50).save('mandelbrot.png')
# %%
