import matplotlib.pyplot as plt
import numpy as np


X=np.load('sigd.npy')[:120000]  # input signals from generated spectra
y1=np.load('sigc.npy')[:120000] # first target
y=np.load('sigi.npy')[:120000] # second target

plt.plot(y[1])
plt.show()
