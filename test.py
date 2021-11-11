#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:22:15 2021

@author: zhu
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

n_experiments = 1000
n_taps = 64
n_fft = 64

samples_time = np.zeros([n_experiments, n_taps], dtype=complex)
samples_freq = np.zeros([n_experiments, n_fft], dtype=complex)

for i in range(n_experiments):
    rndn = np.random.randn(n_taps) + np.random.randn(n_taps)*(1j)
    samples_time[i, :] = rndn
    samples_freq[i, :] = np.fft.fft(rndn, n_fft)

mean_time = np.mean(samples_time)
mean_freq = np.mean(samples_freq)

n_fft = 64
cir_mean = np.linspace(1, 0.0001, 40)
channel_len = len(cir_mean)
cir = 0.01*(np.random.randn(channel_len) + (1j)*np.random.randn(channel_len)) + cir_mean
cfr = np.fft.fft(cir, n=n_fft)
plt.stem(cfr)
plt.ylim([0,1])

a = np.array([0, 1, 2, 2, 4, 5, 7, 7, 8, 9])*0.1
locs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
a_ds_loc = np.array([0, 3, 6, 9])
a_ds_value = a[a_ds_loc]
test_intpl = interp1d(a_ds_loc, a_ds_value, kind='linear', fill_value='extrapolate')
a_intpl = test_intpl(locs)
