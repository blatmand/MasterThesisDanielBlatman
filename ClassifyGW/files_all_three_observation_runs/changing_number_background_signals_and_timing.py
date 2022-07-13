# -*- coding: utf-8 -*-
"""
Created on Tue May 17 21:58:26 2022
This script compares the average balanced accuracy for different parameters s
as a function of the number of background noise signals into which simulated signals
were injected to form the positive training samples. Therefore the number of 
background signals is proportional to the number of training signals used to train the 
classical SVM with RBF kernel. Using a multiplicity of 1 result in an equal number of 
positive and negative training signals. The SVM uses temporal features only
extracted using the library TSFEL. The output are 2 plots saved as png files:
    1. Time needed to extract features and 
    train models for the different amount of background 
    training signals "timing_adding_train_events.png" with a linear fit.
    2. Average balanced accuracy for different s 
    "classical_average_balanced_accuracy_changing_background_signals_"+attribute+"C_"+str(C)+"gamma_"+str(gamma)
    where attribute is "temporal" C=2 and gamma=1.


@author: Daniel Blatman
"""
import matplotlib.pyplot as plt
plt.rc('font', size=14) 
times=[]
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from scipy import signal
import numpy as np
import tsfel
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics



train_multiplicity=1

def get_label(index1,index2,long_data_test,gps):
    '''
    

    Parameters
    ----------
    index1 : integer
        sampling point of start of window.
    index2 : integer
        sampling point of end of window.
    long_data_test : list
        list of gwpy.timeseries.timeseries.TimeSeries objects. 
    gps : float
        gps time of the LIGO gravitational wave event.

    Returns
    -------
    integer
    +1 if window contains a gravitsational wave based on the gps time of the 
    event.
    -1 otherwise.
    '''
    t1=long_data_test.xindex[index1].value
    t2=long_data_test.xindex[index2].value
    if(gps >= t1 and gps <= t2):
        return 1
    return -1
attribute = "temporal"
cfg_file = tsfel.get_features_by_domain(attribute)
events=['GW150914','GW151012','GW151226','GW170814','GW170823'
        ,'GW170729','GW170104','GW170608','GW170809','GW170817'
        ,'GW170818']
def create_filename(i,j,file_id,train_event):
    '''
    

    Parameters
    ----------
    i : integer
        mass 1 of simulated binary black hole coalescing system.
    j : integer
        mass 2 of simulated binary black hole coalescing system.
    file_id : string
    The file_id that contains the information about the s-factor in the 
    training file
    train_event: string
    train_event gives information about from which train event the
    background noise originates from.
    Returns
    -------
    filename : string
        filename of the simulated signal with mass 1 being i 
        and mass 2 being j.

    '''
    filename = "gwpy_hc_mass1_"+str(i)+"_mass2_"+str(j)+"real_data_noise_"+file_id+"_from_"+train_event+"_.txt"
    return filename 
def process_event(event_id):
    '''
    Preprocesses LIGO data.
    1. bandpass filter (50,250) Hz
    2. Crop 1 second of each edge
    3. Notch filter (60,120,180) Hz to remove the influence of the US powerline
    frequency
    4. Divide data into 0.25 second windows with overlap of 0.125 seconds
    5. Multiply each window with a Tukey window function.
    6. Label each window using the function get_label

    Parameters
    ----------
    event_id : string
        The event_id of the real gravitational wave signal from LIGO data.

    Returns
    -------
    hclean : gwpy.timeseries.timeseries.TimeSeries object
        bandpassed and notch filtered TimeSeries object.
    gps : float
        gps time of the LIGO gravitational wave event.
    fs : int
        sample rate frequency.
    x_test_long_signal : list 
        list of gwpy.timeseries.timeseries.TimeSeries objects. 
        Time series test data
    y_test_long_signal : list 
        list of integers. Labels of windows in test data

    '''
    gps = event_gps(event_id)
    segment = (int(gps) - 2, int(gps) + 10)
    hdata = TimeSeries.fetch_open_data('H1', *segment, verbose=True, cache=True)

    data_bp = hdata.bandpass(50, 250, filtfilt=True)
    data_bp = data_bp.crop(int(gps)- 1, int(gps) +9)

    notches = [filter_design.notch(f, data_bp.sample_rate) for f in (60, 120, 180)]
    powernotch = filter_design.concatenate_zpks(*notches)
    hclean = data_bp.filter(powernotch, filtfilt=True)
    long_data_test = hclean
    y_test_long_signal=[]
    fs= int(hclean.sample_rate.value)
    start_index = 0
    end_index = int(fs/4)
    data_segments_test = []
    # split data into 0.25 windows with 0.125 seconds overlap
    test_window = signal.windows.tukey(long_data_test[start_index:int(end_index)].size)
    for i in range(79):
        data_test= long_data_test[start_index:end_index]
        data_test_window = data_test * test_window            
        y_test_long_signal.append(get_label(start_index,end_index-1,long_data_test,gps)) 
        start_index = start_index + int(fs/8)
        end_index = end_index +int(fs/8)
        data_segments_test.append(data_test_window)
    x_test_long_signal = data_segments_test
    return hclean, gps, fs, x_test_long_signal, y_test_long_signal

import time


total_train_events=[]
total_average_classical_accuracies=[]
total_train_events.append(['GW150914']) 
total_train_events.append(['GW150914','GW151012'])
total_train_events.append(['GW150914','GW151012','GW151226']) 
total_train_events.append(['GW150914','GW151012','GW151226','GW170814']) 
total_train_events.append(['GW150914','GW151012','GW151226','GW170814','GW170608']) 
total_train_events.append(['GW150914','GW151012','GW151226','GW170814','GW170608','GW170809']) 
total_train_events.append(['GW150914','GW151012','GW151226','GW170814','GW170608','GW170809','GW170818']) 

for index in range(len(total_train_events)):
    tic = time.perf_counter()
    filename_ids = ["s0.1","s0.01","s0.005","s0.001","s0.0005","s0.0001"]
    legend=["1E-1","1E-2","5E-3","1E-3","5E-4","1E-4"]
    total_balanced_accuracies = []
    average_classical_accuracies=[]

    for file_id in filename_ids:
        results_hclean = []
        results_gps = []
        results_fs= []
        results_x_test_long_signal =[]
        results_y_test_long_signal = []
        for i in range(len(events)):
            hclean, gps, fs, x_test_long_signal, y_test_long_signal = process_event(events[i])
            results_hclean.append(hclean)
            results_gps.append(gps)
            results_fs.append(fs)
            results_x_test_long_signal.append(x_test_long_signal)
            results_y_test_long_signal.append(y_test_long_signal)
        
        hclean1 = results_hclean[0]
        # using the time series feature extraction library tsfel
        data_segments_train = []
        
        filenames_train=[]
        
        train_events= total_train_events[index] 
        
        for iii in range(len(train_events)):
            
            data_train = TimeSeries.read('different_noise_long_training_data_noise_bp_and_notch_filtered_'+train_events[iii]+'.txt')
            length = 25.0
            
            fs = int(hclean1.sample_rate.value)
        
            start_index = 0
            end_index = int(fs*0.25)
            train_window = signal.windows.tukey(data_train[start_index:int(end_index)].size)
        
            for i in range(int(length*4)):
                data_segment_train= data_train[start_index:int(end_index)]
                hwin_train = data_segment_train * train_window
                
                start_index = start_index + int(0.25*fs)
                end_index = end_index +int(0.25*fs)
                data_segments_train.append(hwin_train)
            
        
        num=10
        
        for train_event in train_events:
            for i in range(num):
                for j in range(num):
                    filenames_train.append(create_filename(31+i,23+j,file_id,train_event))
                        
        names_train = filenames_train        
    
    
        total_train_data = []
        train_labels=[]
    
        windowed_total_train_data=[]
    
        for i in range(len(names_train)):
            read_data = TimeSeries.read(names_train[i])
            total_train_data.append(read_data)
    
        for i in range(len(data_segments_train)):
            total_train_data.append(data_segments_train[i])  
    
        windowed_simulated_data_array = []
        simulated_data_window = signal.windows.tukey(len(total_train_data[0]))
        for j in range(len(names_train)):
        
            windowed_simulated_data = total_train_data[j]*simulated_data_window
            windowed_simulated_data_array.append(windowed_simulated_data)
    
        for i in range(len(windowed_simulated_data_array)):
            windowed_total_train_data.append(windowed_simulated_data_array[i])
            train_labels.append(1)
    
        for i in range(len(data_segments_train)):
            windowed_total_train_data.append(data_segments_train[i])  
            train_labels.append(-1)
    
        x_train_sig = windowed_total_train_data
        X_train = tsfel.time_series_features_extractor(cfg_file, x_train_sig, fs=fs)
    
        results_X_test_long_signal=[]
        for i in range(len(events)):
            results_X_test_long_signal.append(tsfel.time_series_features_extractor(cfg_file, results_x_test_long_signal[i], fs=results_fs[i]))
    
        # Highly correlated features are removed
        corr_features = tsfel.correlated_features(X_train)
        X_train.drop(corr_features, axis=1, inplace=True)
        tempvars=[]
        for i in range(len(events)):
            tempvars.append(results_X_test_long_signal[i])
            results_X_test_long_signal[i].drop(corr_features, axis=1, inplace=True)
    
    
        # Remove low variance features
        selector = VarianceThreshold()
        tempvars2=[]
        for i in range(len(events)):
            tempvars2.append(results_X_test_long_signal[i])
        X_train = selector.fit_transform(X_train)
        for i in range(len(events)):
            results_X_test_long_signal[i] = selector.transform(results_X_test_long_signal[i])
    
        # Normalising Features
        scaler = preprocessing.StandardScaler()
        nX_train = scaler.fit_transform(X_train)
        results_nX_test_long_signal = []
        for i in range(len(events)):
            results_nX_test_long_signal.append(scaler.transform(results_X_test_long_signal[i]))
        
        y_train = train_labels 
    
        C=2
        gamma=1
        model=SVC(C=C,gamma=gamma,kernel='rbf')
        model.fit(nX_train,y_train)
        results_labels_test_long_signal = []
        results_accuracy_test_svc_long_signal = []
        for i in range(len(events)):
            results_labels_test_long_signal.append(model.predict(results_nX_test_long_signal[i]))
            results_accuracy_test_svc_long_signal.append(metrics.balanced_accuracy_score(y_true=results_y_test_long_signal[i], y_pred=results_labels_test_long_signal[i]))
        
        total_balanced_accuracies.append(results_accuracy_test_svc_long_signal)
        
        average_accuracy=sum(results_accuracy_test_svc_long_signal)/len(results_accuracy_test_svc_long_signal)
        
        average_classical_accuracies.append(average_accuracy)
    total_average_classical_accuracies.append(average_classical_accuracies)  
    toc = time.perf_counter()
    times.append(toc-tic)
    
#calculate the elapsed times:
colors = ["red"]

num_train_events=[]
for i in range(len(total_train_events)):
    num_train_events.append(len(total_train_events[i]))
plt.figure(figsize=(12,6))
plt.plot(num_train_events, times,'.',color=colors[0])

polynomial = np.polyfit(num_train_events, times, 1)
p=np.poly1d(polynomial)
x=np.linspace(1,7,100)
plt.plot(num_train_events, times,'.',color=colors[0],label="measured times")
plt.plot(x, p(x),'.',label="linear fit")
plt.title("Time count for adding Bg. signals with train events")
plt.ylabel("time[s]")
plt.xlabel("Number Bg. signals with train events used")
plt.legend()
plt.savefig("timing_adding_Bg_signals_with_train_events")     
plt.close()    

colors=["red","orange","green","violet","cyan","grey","black"]    
plt.figure(figsize=(14,6))
for index in range(len(total_average_classical_accuracies)):
    plt.plot(legend,total_average_classical_accuracies[index] ,'.',label="RBF kernel: "+str(index+1) +" Bg. signals",color=colors[index])
plt.legend(loc="center right",bbox_to_anchor=(1.35, 0.5))
plt.xlabel("parameter s")
plt.ylabel("average balanced accuracy")
plt.ylim(0,1)
plt.title("Average balanced accuracy "+ str(attribute)+" features"+ " C = "+str(C)+" gamma = "+str(gamma))
plt.tight_layout()
plt.savefig("classical_average_balanced_accuracy_changing_background_signals_"+attribute+"C_"+str(C)+"gamma_"+str(gamma))      
plt.close()        