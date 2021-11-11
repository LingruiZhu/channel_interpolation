#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:50:22 2021

@author: zhu
"""
import numpy as np
from scipy.interpolate import interp1d

x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f = interp1d(x, y, fill_value='extrapolate')
f2 = interp1d(x, y, kind='cubic', fill_value='extrapolate')
xnew = np.linspace(0, 11, num=51, endpoint=True)
ynew = np.cos(-xnew**2/9.0)

import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-x', xnew, f2(xnew), '--', xnew, ynew, '--')
plt.legend(['data', 'linear', 'cubic' , 'new_data'], loc='best')
plt.show()


