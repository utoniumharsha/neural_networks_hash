#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 09:24:31 2020

@author: s0v00kg
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams.update({'font.size': 18})

#define Domain
dx = 0.001
L = np.pi
x = L * np.arange(-1+dx, 1+dx, dx)
n = len(x)
nquart = int(np.floor(n/4))

#define hat function
f = np.zeros_like(x)
f[nquart:2*nquart] = (4/n)*np.arange(1, nquart+1)
f[2*nquart:3*nquart] = np.ones(nquart) - (4/n)*np.arange(0, nquart)

fig, ax = plt.subplots()
ax.plot(x, f, '-', color='k', LineWidth=2)

# Compute Fourier Series
name = "Accent"
cmap = get_cmap('tab10')
colors = cmap.colors
ax.set_prop_cycle(color=colors)

A0 = np.sum(f) * dx
fFS = A0/2

A = np.zeros(20)
B = np.zeros(20)
for k in range(20):
    A[k] = np.sum( f * np.cos(np.pi*(k+1)*x/L )) * dx
    B[k] = np.sum( f * np.sin(np.pi*(k+1)*x/L )) * dx
    fFS += A[k]*np.cos((k+1) * np.pi * x/L) + B[k]*np.sin((k+1) * np.pi * x/L)
    ax.plot(x, fFS, '-')

# Plot amplitudes
fFS = (A0/2) * np.ones_like(f)
kmax = 100
A = np.zeros(kmax)
B = np.zeros(kmax)
err = np.zeros(kmax)

A[0] = A0/2
err[0] = np.linalg.norm(f-fFS)/np.linalg.norm(f)

for k in range(1, kmax):
    A[k] = np.sum(f * np.cos(np.pi*k*x/L)) * dx
    B[k] = np.sum(f * np.sin(np.pi*k*x/L)) * dx
    fFS += A[k] * np.cos(k*np.pi*x/L) + B[k] * np.sin(k*np.pi*x/L)
    err[k] = np.linalg.norm(f-fFS)/np.linalg.norm(f)
    
thresh = np.median(err) * np.sqrt(kmax) * (4/np.sqrt(3))
r = np.max(np.where(err > thresh))

fig, axs = plt.subplots(2, 1)
axs[0].semilogy(np.arange(kmax), A, color='k', LineWidth=2)
axs[0].semilogy(r, A[r], 'o', color='b', MarkerSize=10)
plt.sca(axs[0])
plt.title('Fourier Coefficients')

axs[1].semilogy(np.arange(kmax), err, color='k', LineWidth=2)
axs[1].semilogy(r, err[r], 'o', color='b', MarkerSize=10)
plt.sca(axs[1])
plt.title('Error')

plt.show()

