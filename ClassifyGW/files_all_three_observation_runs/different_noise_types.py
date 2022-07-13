#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel Blatman

Created on Sun Feb 20 19:59:49 2022

File creates simulated signals that are injected into background noise before
the gravitational wave events:
'GW150914','GW151012','GW151226','GW170814','GW170608','GW170809','GW170818',
'GW190408_181802','GW190412','GW190413_052954','GW190413_134308'
,'GW200306_093714','GW200311_115853','GW200316_215756'
The simulated signals from binary coalescences have masses: 
    1. massesM1+ integer 
    2. massesM2 + integer, thereby resulting in a 10x 10 grid of masses.
The background noise is preprocessed before the signals are injected into these
background signals. Overall, the simulated signals are cropped in
order to be 250 millisecond long.  
The script should be run in Ubuntu with the libraries pycbc and gwpy installed.
"""

from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
import gwpy
from scipy import signal
from pycbc.waveform import get_td_waveform
import numpy as np
massesM1=31
massesM2=23
events = ['GW150914','GW151012','GW151226','GW170814','GW170608','GW170809','GW170818',
          'GW190408_181802','GW190412','GW190413_052954','GW190413_134308'

,'GW200306_093714',
                'GW200311_115853','GW200316_215756']
s_values=[1E-1,1E-2,5E-3,1E-3,5E-4,1E-4]
for s_value in s_values:
    for event_id in events:
        gps = event_gps(event_id)
        segment = (int(gps) - 81, int(gps) -9)
        hdata = TimeSeries.fetch_open_data('H1', *segment, verbose=True, cache=True)
    
        data_bp = hdata.bandpass(50, 250, filtfilt=True)
        data_bp = data_bp.crop(int(gps)- 80, int(gps) -10)
    
        notches = [filter_design.notch(f, data_bp.sample_rate) for f in (60, 120, 180)]
        powernotch = filter_design.concatenate_zpks(*notches)
        hclean = data_bp.filter(powernotch, filtfilt=True)
    
        fs= int(hclean.sample_rate.value)
        start_index = 0
        end_index = int(fs)
        data_segments_test = []
        # split data into 0.25 windows with 0.125 seconds overlap
        test_window = signal.windows.tukey(hclean[start_index:int(end_index)].size)
        for i in range(109):
            data_test= hclean[start_index:end_index]
            data_test_window = data_test * test_window
            start_index = start_index + int(fs/2)
            end_index = end_index +int(fs/2)
            data_segments_test.append(data_test_window)
        noise = data_segments_test
        step=0
        for i in range(10):
            for j in range(10):
            
                hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                                     mass1=massesM1+i,
                                     mass2=massesM2+j,
                                     delta_t=1.0/(4096),
                                     f_lower=35)
                
                hc= hc*s_value
                noise_temp = TimeSeries(np.array(noise[step].value),sample_rate=4096)
                hc.start_time=0.25
                data = noise_temp.inject(gwpy.timeseries.TimeSeriesBase.from_pycbc(hc))
                data=data.crop(start=0.25,end=0.5,copy=False)
                step=step+1
                data.write("gwpy_hc_mass1_"+str(massesM1+i)+"_mass2_"+str(massesM2+j)+"real_data_noise_s"+str(s_value)+"_from_"+event_id+"_.txt")
