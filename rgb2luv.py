import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt

def rgb_to_luv_man(image):
    
    R = image[...,0]/255.0
    G = image[...,1]/255.0
    B = image[...,2]/255.0

    X = (0.412453*R) + (0.35758*G) + (0.180423*B)
    Y = (0.212671*R) + (0.715160*G) + (0.072169*B)
    Z = (0.019334*R) + (0.119193*G) + (0.950227*B)
    
    L = np.zeros_like(Y)
    L[Y > 0.008856] = 116*((Y[Y > 0.008856])**(1/3))-16
    L[Y <= 0.008856] = (903.3)*Y[Y <= 0.008856]
    
    u_n = 0.19793943
    v_n = 0.46831096
    denom = X + 15*Y + 3*Z
    u_m = 4 * X / denom
    v_m = 9 * Y / denom

    u = 13 * L * (u_m - u_n)
    v = 13 * L * (v_m - v_n)

    luv_image = np.zeros_like(image)
    luv_image[..., 0] = L
    luv_image[..., 1] = u
    luv_image[..., 2] = v
    
    return luv_image