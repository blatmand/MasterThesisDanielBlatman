# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:08:29 2022
Plots the Q transform of the event GW150914 of the Hanford H1 time series
after preprocessing. The time series is plotted as well. The plots are saved. 
@author: Daniel Blatman
"""
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
event='GW150914'

gps = event_gps(event)

segment = (int(gps) - 2, int(gps) + 10)

hdata = TimeSeries.fetch_open_data('H1', *segment, verbose=True, cache=True)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

plot = hdata.plot()
ax = plot.gca()
ax.set_epoch(gps)
ax.set_title(event)
ax.set_ylabel('Strain []')
plt.axvline(x=event_gps(event), color="red", linestyle="--")
ax.set_xlim(gps-0.2,gps+0.2)
plt.title(event+" without filters")
plot.tight_layout()
plt.savefig(event+"NoFilters")
plt.close()

data_bp = hdata.bandpass(50, 250, filtfilt=True)
data_bp = data_bp.crop(int(gps)- 1, int(gps) +9)

notches = [filter_design.notch(f, data_bp.sample_rate) for f in (60, 120, 180)]
powernotch = filter_design.concatenate_zpks(*notches)
hclean = data_bp.filter(powernotch, filtfilt=True)

plot = hclean.plot()
ax = plot.gca()
ax.set_epoch(gps)
ax.set_title(event+ " bandpassed (50,250) Hz and notch filtered (60,120,180) Hz")
ax.set_ylabel('Strain []')
plt.axvline(x=event_gps(event), color="red", linestyle="--")
ax.set_xlim(gps-0.2,gps+0.2)
plot.tight_layout()
plt.savefig(event+"BandpassedAndNotchFiltered")
plt.close()

segment = (int(gps) - 2, int(gps) + 2)
hdata = TimeSeries.fetch_open_data('H1', *segment, verbose=True, cache=True)

hq = hdata.q_transform(frange=(50, 250),outseg=(gps-0.25,gps+0.25))
plot = hq.plot()
plot.colorbar(label="Normalised energy")

ax = plot.gca()
ax.set_title("Q transform of bandpassed and notch filtered signal "+event)
ax.set_epoch(gps)
ax.axvline(x=gps, linewidth=1.5, color="r")
plot.savefig("Qtransform"+event)
plt.close()











