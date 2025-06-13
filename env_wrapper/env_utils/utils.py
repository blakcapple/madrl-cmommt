
import numpy as np 
import math 

def l2norm(x, y):  return math.sqrt(l2normsq(x,y))

def l2normsq(x, y): return (x[0]-y[0])**2 + (x[1]-y[1])**2

def wrap(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle