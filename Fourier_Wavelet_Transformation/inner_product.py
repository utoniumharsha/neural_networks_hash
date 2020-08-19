#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 01:53:34 2020

@author: s0v00kg
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})


f = np.array([0, 0, .1, .2, .25, .2, .25, .3, .35, .43, .45, .5, .55, .5, .4, .425, .45, .425, .4, .35, .3, .25, .225, .2, .1, 0, 0])
g = np.array([0, 0, .025, .1, .2, .175, .2, .25, .25, .3, .32, .35, .375, .325, .3, .275, .275, .25, .225, .225, .2, .175, .15, .15, .05, 0, 0])
g = g - 0.025 * np.ones_like(g)

x = 0.1 * np.arange(1,len(f)+1)
xf = np.arange(0.1,x[-1],0.01)

f_interp = interpolate.interp1d(x, f, kind='cubic')
g_interp = interpolate.interp1d(x, g, kind='cubic')

ff = f_interp(xf)  
gf = g_interp(xf)

plt.plot(xf[10:-10],ff[10:-10],color='k',LineWidth=2)
plt.plot(x[1:-2],f[1:-2],'o',color='b')

plt.plot(xf[10:-10],gf[10:-10],color='k',LineWidth=2)
plt.plot(x[1:-2],g[1:-2],'o',color='r')

# inner product of f and g
totes = f*g
totes_interp = interpolate.interp1d(x, totes, kind='cubic')
totesf = totes_interp(xf)
area = np.sum(totes)

plt.plot(xf[10:-10], totesf[10:-10],color = 'k', LineWidth=2)
plt.plot(x[1:-2], totes[1:-2], 'o', color='g')

plt.show()
