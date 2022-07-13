# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 22:53:03 2022

@author: Daniel Blatman
Example shows the true labels given to GW150914 (Hanford H1 time series) shown in chapter 3 section 3.5
of the thesis ("Form of test data").
"""
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
import tsfel
event='GW150914'
gps = event_gps(event)
segment = (int(gps) - 2, int(gps) + 10)
hdata = TimeSeries.fetch_open_data('H1', *segment, verbose=True, cache=True)

data_bp = hdata.bandpass(50, 250, filtfilt=True)
data_bp = data_bp.crop(int(gps)- 1, int(gps) +9)

notches = [filter_design.notch(f, data_bp.sample_rate) for f in (60, 120, 180)]
powernotch = filter_design.concatenate_zpks(*notches)
hclean = data_bp.filter(powernotch, filtfilt=True)

long_data_test = hclean

cfg_file = tsfel.get_features_by_domain("temporal")   
fs= int(hclean.sample_rate.value)
start_index = 0
end_index = int(fs/4)

data_segments_test = []

def get_label(index1,index2):
    t1=long_data_test.xindex[index1].value
    t2=long_data_test.xindex[index2].value
    if(gps >= t1 and gps <= t2):
        return 1
    return -1
y_test_long_signal=[]
from scipy import signal
# split data into 0.25 windows with 0.125 seconds overlap
test_window = signal.windows.tukey(long_data_test[start_index:int(end_index)].size)
for i in range(79):
    data_test= long_data_test[start_index:end_index]
    data_test_window = data_test * test_window            
    y_test_long_signal.append(get_label(start_index,end_index-1)) # [0,2047]
    start_index = start_index + int(fs/8)
    end_index = end_index +int(fs/8)
    data_segments_test.append(data_test_window)
x_test_long_signal = data_segments_test
print(y_test_long_signal)


