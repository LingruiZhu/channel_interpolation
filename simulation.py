#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:45:18 2021

@author: zhu
"""

import numpy as np
import commpy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

symbols_per_frame = 1
modulation_order = 2
bits_per_frame = symbols_per_frame*modulation_order
n_subcarriers = 51

n_frames = 100
SNRs = np.linspace(0, 15, 16)
pilot_loc = np.linspace(0, 60, 13, endpoint=True, dtype=int)
subcarrier_loc = np.linspace(0, 63, 64, endpoint=True, dtype=int)
data_loc = [i for i in subcarrier_loc if i not in pilot_loc]
qam = commpy.modulation.QAMModem(np.power(2,modulation_order))
n_fft = 64

NMSE_cubic = np.zeros([len(SNRs)])
NMSE_linear = np.zeros([len(SNRs)])
BERs = np.zeros([len(SNRs)])

for index, snr in enumerate(SNRs):
    tx_bits = (-1) * np.ones([n_fft, n_frames, bits_per_frame])
    rx_bits = (-1) * np.ones([n_fft, n_frames, bits_per_frame])
    cfr_real = np.zeros([n_fft, n_frames], dtype=complex)
    cfr_est = np.zeros([n_fft, n_frames], dtype=complex)
    cfr_est_interp_linear = np.zeros([n_fft, n_frames], dtype=complex)
    cfr_est_interp_cubic = np.zeros([n_fft, n_frames], dtype=complex)
    for i in range(n_frames):
        cir_mean = [1, 0.8, 0.5, 0.2, 0.1, 0.05]
        cir = (np.random.randn(6) + (1j)*np.random.randn(6)) * cir_mean
        cfr = np.fft.fft(cir, n=64)
        cfr_real[:, i] = cfr
        for j in pilot_loc:
            pilot = 1+1j
            pilot_symbols = np.ones([symbols_per_frame, 1])
            ch_out = pilot_symbols * cfr[j]
            rx_pilots = commpy.awgn(ch_out, snr)
            ls_est = rx_pilots / pilot_symbols
            cfr_est[j, i] = ls_est
        freq_interp_linear = interp1d(pilot_loc, cfr_est[pilot_loc, i], kind='linear', fill_value='extrapolate')
        freq_interp_cubic = interp1d(pilot_loc, cfr_est[pilot_loc, i], kind='cubic', fill_value='extrapolate')
        cfr_interp_linear = freq_interp_linear(subcarrier_loc)
        cfr_interp_cubic = freq_interp_cubic(subcarrier_loc)
        cfr_est_interp_linear[:,i] = cfr_interp_linear   
        cfr_est_interp_cubic[:,i] = cfr_interp_cubic
        for j in data_loc:    
            msg_bits = np.random.randint(0, 2, bits_per_frame)
            symbols = qam.modulate(msg_bits)
            ch_out = symbols * cfr[j]
            rx_symbols = commpy.awgn(ch_out, snr)
            equalized_rx = rx_symbols / cfr_interp_linear[j]
            estimate_bits = qam.demodulate(equalized_rx, demod_type='hard')
            tx_bits[j, i, :] = msg_bits
            rx_bits[j, i, :] = estimate_bits
        
    BERs[index] = np.sum(tx_bits != rx_bits) / (n_frames*n_subcarriers*bits_per_frame)
    
    NMSE_linear[index] = (np.sum(np.abs(cfr_est_interp_linear - cfr_real)) / cfr_real.size) / np.mean(np.abs(cfr_real))
    NMSE_cubic[index] = (np.sum(np.abs(cfr_est_interp_cubic - cfr_real)) / cfr_real.size) / np.mean(np.abs(cfr_real))
        
        
plt.figure(1)
plt.plot(SNRs, NMSE_cubic, 'r-+', SNRs, NMSE_linear, 'b-o')
plt.legend(['cubic interpolation', 'linear interpolation'], loc='best')
plt.ylabel('Channel estimation NMSE')
plt.xlabel('SNR in dB')
plt.grid()
plt.show()

plt.figure(2)
plt.semilogy(SNRs, BERs)
plt.xlabel('SNR in dB')
plt.ylabel('Bit Error Rate')
plt.grid()
plt.show()