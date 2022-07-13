# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:07:01 2022
creates the negative training examples containing measurement 
(used as background noise) data before the gravitational waves
specified in the list "events" in line 14 of this code
happened. 
@author: Daniel Blatman
"""
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design

events=['GW150914','GW151012','GW151226','GW170814','GW170608','GW170809','GW170818',
        'GW190408_181802','GW190412','GW190413_052954','GW190413_134308','GW200306_093714','GW200311_115853','GW200316_215756']  
        
for i in range(len(events)):    
    event = events[i]
    gps = event_gps(event)
    long_segment = (int(gps) - 29, int(gps) - 2)
    hdata_long = TimeSeries.fetch_open_data('H1', *long_segment, verbose=True, cache=True)
    long_data_bp = hdata_long.bandpass(50, 250, filtfilt=True)
    long_data_bp = long_data_bp.crop(int(gps)- 28, int(gps) - 3)
    notches = [filter_design.notch(f, long_data_bp.sample_rate) for f in (60, 120, 180)]
    powernotch = filter_design.concatenate_zpks(*notches)
    hclean = long_data_bp.filter(powernotch, filtfilt=True)
    hclean.write('different_noise_long_training_data_noise_bp_and_notch_filtered_'+events[i]+'.txt')