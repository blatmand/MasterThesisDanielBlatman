#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel Blatman
Created on Sun Feb 20 19:59:49 2022

This script shows an example of creating simulated signals which are then 
injected into background noise from before the event 'GW150914' after the 
background noise has been preprocessed. The resulting simulated time series are
plotted for the parameter s being 
    1. s=1E-1
    2. s=1E-2
    3. s=1E-3
    4. s=1E-4
The plots are saved as png files.    
"""

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
import gwpy
from scipy import signal
from pycbc.waveform import get_td_waveform
import numpy as np
import matplotlib.pyplot as plt
massesM1=31
massesM2=23
events = ['GW150914']
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
    for s in [1E-1,1E-2,1E-3,1E-4] :
        for i in range(1):
            for j in range(1):
            
                hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                                     mass1=massesM1+i,
                                     mass2=massesM2+j,
                                     delta_t=1.0/(4096),
                                     f_lower=35)
                
                hc= hc*s
                noise_temp = TimeSeries(np.array(noise[step].value),sample_rate=4096)
                hc.start_time=0.25
                data = noise_temp.inject(gwpy.timeseries.TimeSeriesBase.from_pycbc(hc))
                data=data.crop(start=0.25,end=0.5,copy=False)
                step=step+1

                plot = data.plot()
                axes=plot.gca()
                plt.ylabel("Strain []")
                plt.title("injected signal = simulated signal * "+str(s)+", Background noise: "+event_id)
                plt.savefig("injected_signal_into_GW150914_m1_31_m2_23_s_equals_"+str(s)+".png")