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

def collect_training_data(ch_est, pilot_interval, pilot_number_per_group):
    data_length = len(ch_est)
    input_dim = pilot_number_per_group
    output_dim = (pilot_interval - 1) * (pilot_number_per_group - 1)
    window_length = pilot_interval * (pilot_number_per_group - 1) + 1
    set_number = data_length - window_length + 1
    training_output = np.zeros([output_dim, set_number], dtype=ch_est.dtype)
    training_input = np.zeros([input_dim, set_number], dtype=ch_est.dtype)
    index = np.linspace(0, window_length, window_length, endpoint=False)
    pilot_location = np.array([j for j in index if j%pilot_interval==0], dtype=int)
    data_location = np.array([j for j in index if j not in pilot_location], dtype=int)
    for i in range(set_number):
        window_data = ch_est[i: i+window_length]
        training_input[:, i] = window_data[pilot_location]
        training_output[:, i] = window_data[data_location]
    return training_input, training_output


def interpolate_channel(w, ch_est_pilot, pilot_interval, pilot_per_group, n_groups):
    data_length = len(ch_est_pilot)
    window_length = pilot_interval * (pilot_per_group - 1) + 1
    index = np.linspace(0, window_length, window_length, endpoint=False)
    pilot_location = np.array([j for j in index if j%pilot_interval==0], dtype=int)
    data_location = np.array([j for j in index if j not in pilot_location], dtype=int)
    ch_intpl_est = ch_est_pilot.copy()
    for i in range(n_groups):
        window_data = ch_est_pilot[i*pilot_interval: i*pilot_interval+window_length].copy()
        input_data = window_data[pilot_location]
        output_data = np.dot(w, input_data)
        window_data[data_location] = output_data
        ch_intpl_est[i*pilot_interval: i*pilot_interval+window_length] = window_data
    return ch_intpl_est


def mmse_interpolate():
    return 0  #TODO: add mmse interpolation as benchmarks


def read_channel_data(file, n_subcarriers, n_time_stamps):
    channel_data_raw = np.loadtxt(file, dtype=complex, delimiter=',')
    assert channel_data_raw.shape[0] >= n_time_stamps
    channel_data_output = channel_data_raw[:n_time_stamps, :n_subcarriers]
    return channel_data_output.transpose()
    

symbols_per_frame = 1
modulation_order = 2
bits_per_frame = symbols_per_frame*modulation_order
file_path = '/home/zhu/Codes/channel_estimation/toy_example_interporlation/data/Cost_channel_doppler_0.csv'

pilot_per_group = 2
pilot_interval = 3
n_group = 170
n_total_subcarriers = pilot_interval * (pilot_per_group - 1) * n_group + 1                   # here need to be careful
n_data_subcarriers = (pilot_interval - 1) * (pilot_per_group - 1) * n_group    # number of data subcarriers
n_fft = 512

n_frames = 20
symbol_per_frame = 10                        # here just set this parameter but not added in the simulation yet.
SNRs = np.linspace(0, 20, 21)                
pilot_loc = np.linspace(0, n_total_subcarriers-1, n_group*(pilot_per_group-1)+1, endpoint=True, dtype=int)
subcarrier_loc = np.linspace(0, n_total_subcarriers, n_total_subcarriers, endpoint=False, dtype=int)
data_loc = np.array([i for i in subcarrier_loc if i not in pilot_loc])
qam = commpy.modulation.QAMModem(np.power(2,modulation_order))

NMSE_cubic = np.zeros([len(SNRs)])
NMSE_linear = np.zeros([len(SNRs)])
NMSE_olml = np.zeros([len(SNRs)])
NMSE_ls_est = np.zeros([len(SNRs)])
BERs = np.zeros([len(SNRs)])

for index, snr in enumerate(SNRs):
    tx_bits = (-1) * np.ones([n_total_subcarriers, n_frames, bits_per_frame])
    rx_bits = (-1) * np.ones([n_total_subcarriers, n_frames, bits_per_frame])
    cfr_real = read_channel_data(file_path, n_total_subcarriers, n_frames)
    cfr_est = np.zeros([n_total_subcarriers, n_frames], dtype=complex)
    cfr_est_ls = np.zeros([n_total_subcarriers, n_frames], dtype=complex)
    cfr_est_interp_linear = np.zeros([n_total_subcarriers, n_frames], dtype=complex)
    cfr_est_interp_cubic = np.zeros([n_total_subcarriers, n_frames], dtype=complex)
    cfr_est_interp_olml = np.zeros([n_total_subcarriers, n_frames], dtype=complex)
    for i in range(n_frames):
        cfr = cfr_real[:, i]
        
        for j in pilot_loc:
            pilot = 1+1j
            pilot_symbols = np.ones([symbols_per_frame, 1])
            ch_out = pilot_symbols * cfr[j]
            rx_pilots = commpy.awgn(ch_out, snr)
            ls_est = rx_pilots / pilot_symbols
            cfr_est[j, i] = ls_est
            cfr_est_ls[j, i] = ls_est
        
        # estimate channel on pilot position
        freq_interp_linear = interp1d(pilot_loc, cfr_est[pilot_loc, i], kind='linear', fill_value='extrapolate')
        freq_interp_cubic = interp1d(pilot_loc, cfr_est[pilot_loc, i], kind='cubic', fill_value='extrapolate')
        freq_interp_quadratic = interp1d(pilot_loc, cfr_est[pilot_loc, i], kind='quadratic', fill_value='extrapolate')
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
            tx_symbols = qam.modulate(estimate_bits)
            ls_est = rx_symbols / tx_symbols
            cfr_est_ls[j, i] = ls_est    # dataset for training interpolation matrix
            tx_bits[j, i, :] = msg_bits
            rx_bits[j, i, :] = estimate_bits
        # update weighting matrix
        data_input, data_output = collect_training_data(cfr_real[:,i], pilot_interval, pilot_per_group)   # here should be cfr_est_ls, not cfr_real
        w = np.dot(data_output, np.linalg.pinv(data_input))
        cfr_est_interp_olml[:,i] = interpolate_channel(w, cfr_est[:,i], pilot_interval, pilot_per_group, n_group)
    
    BERs[index] = np.sum(tx_bits != rx_bits) / (n_frames*n_data_subcarriers*bits_per_frame)
    
    NMSE_linear[index] = np.mean(np.square(cfr_est_interp_linear - cfr_real)) / np.mean(np.square(cfr_real))
    NMSE_cubic[index] = np.mean(np.square(cfr_est_interp_cubic - cfr_real)) / np.mean(np.square(cfr_real))
    NMSE_olml[index] = np.mean(np.square(cfr_est_interp_olml - cfr_real)) / np.mean(np.square(cfr_real))
    NMSE_ls_est[index] = np.mean(np.square(cfr_est_ls - cfr_real)) / np.mean(np.square(cfr_real))    
        
plt.figure(1)
plt.semilogy(SNRs, NMSE_cubic, 'r-+', SNRs, NMSE_linear, 'b-o', SNRs, NMSE_olml, 'g-x', SNRs, NMSE_ls_est, 'm-d')
plt.legend(['Cubic interpolation', 'Linear interpolation', 'Online-learning', 'LS estimation'], loc='best')
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