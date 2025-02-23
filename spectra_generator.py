#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 17:15:15 2021

@author: mohammadkazemzadeh
"""

import pybaselines as pbs
import matplotlib.pyplot as plt
import numpy as np
from pybaselines import utils
from random import uniform
from astropy.modeling.models import Moffat1D, Lorentz1D, Voigt1D, Trapezoid1D
import pandas as pd
from scipy.signal import savgol_filter
import pybaselines

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def roll(u,b):
    
    a=[]
    for i in range(len(u)):
       a.append(u[i]) 
    v=a
    
    if b>0:        
        v[b:]=a[:-b]
        v[:b]=[a[0] for k in range(b)]
    if b<0:
        b=abs(b)
        v[:-b]=a[b:]
        v[-b:]=[a[-b] for k in range(b)]
    return np.array(v)

mi=100
ma=4000

L=1024

sigm=[]
sigi=[]
sigd=[]
sigc=[]

spectralnumber=20000


## okay so basically for the range of the spectra, we are adding artificial peaks here - modeled using diff functional forms
# the peak positions, heights, widths -> randomized
for ii in range(spectralnumber):
    x = np.linspace(mi, ma, L)
    signal=0*x
    
    #gaussian
    npeaks= int(uniform(0,9))
    pp=[int(uniform(mi+100,ma-100)) for i in range(npeaks)]
    ph=[uniform(100,2000) for i in range(npeaks)]
    pw=[uniform(4,80) for i in range(npeaks)]

    for i in range(npeaks):
        signal=signal+utils.gaussian(x, ph[i], pp[i], pw[i])
        
    #Moffat
    npeaks= int(uniform(0,9))
    pp=[int(uniform(mi+100,ma-100)) for i in range(npeaks)]
    ph=[uniform(100,2000) for i in range(npeaks)]
    pw=[uniform(4,80) for i in range(npeaks)]
    s1 = Moffat1D()
    
    for i in range(npeaks):
        s1.x_0=pp[i]
        s1.width=pw[i]
        s1.amplitude=ph[i]
        s1.gamma=uniform(.5*pw[i],1.5*pw[i])
        s1.alpha=uniform(.5,1.5)
        signal=signal+s1(x)
           
    #Lorantz
    npeaks= int(uniform(0,9))
    pp=[int(uniform(mi+100,ma-100)) for i in range(npeaks)]
    ph=[uniform(100,2000) for i in range(npeaks)]
    pw=[uniform(4,80) for i in range(npeaks)]
    s1=Lorentz1D()
    
    for i in range(npeaks):
        s1.x_0=pp[i]
        s1.fwhm=pw[i]
        s1.amplitude=ph[i]
        signal=signal+s1(x)
    
    #Voigt
    npeaks= int(uniform(0,9))
    pp=[int(uniform(mi+100,ma-100)) for i in range(npeaks)]
    ph=[uniform(100,2000) for i in range(npeaks)]
    pw=[uniform(4,80) for i in range(npeaks)]
    
    for i in range(npeaks):
        V=Voigt1D(x_0=pp[i], amplitude_L=ph[i],
                              fwhm_L=pw[i], fwhm_G=uniform(.5,1)*pw[i]) 
        signal=signal+V(x)


    # introduce random noise here and like other artifacts
    nl=uniform(10,70) 
    noise = np.random.normal(0, nl, x.size)

    # adding cosmic rays ---> so basically sharp spikes
    ncosmicray=int(uniform(0,3))
    pp=[int(uniform(mi+10,ma-10)) for i in range(ncosmicray)]
    ph=[uniform(1000,10000) for i in range(ncosmicray)]
    pw=[int(uniform(1,3)) for i in range(ncosmicray)]
    cosm=0*x
    for q in range(ncosmicray):
        cosm=cosm+utils.gaussian(x, ph[q], pp[q], pw[q])
    
    
    ncosmicray=int(uniform(0,4))
    pp=[int(uniform(mi+10,ma-10)) for i in range(ncosmicray)]
    pw=[int(uniform(1,3)) for i in range(ncosmicray)]
    z=0*x
    for q in range(ncosmicray):
        z[pp[q]:int(pp[q]+pw[q])]=uniform(2000,10000)
    
    cosm=cosm+z

    # distorting the baseline based on polynomial trends
    poly=10
    poly=int(uniform(0,poly+1))
    base=0*x
    for w in np.array(range(poly+1)):
        base=base+uniform(-1,1)*(x/max(x))**w
                        
    base=base*uniform(1000,10000)                    
    base=base-min(base)                  
    base=base+(x*0+uniform(0,6000))

    sigd.append(signal+noise+base+cosm)
    sigc.append((signal+noise+cosm))
    sigi.append((signal))
    
    if ii%5000==0:
        print(str(ii/spectralnumber*100)[:4]+"% finished")
        plt.plot(sigd[-1])
        
        plt.savefig("spectra"+str(ii)+".png")
        plt.show()

sigi=np.array(sigi)  # pure signal --> ideal; no noise
sigd=np.array(sigd)  # final spectrum --> signal, noise, and baseline, and cosmic rays
sigc=np.array(sigc)  # spectrum w/o baseline

np.save("sigi", sigi)
np.save("sigd", sigd)
np.save("sigc", sigc)


