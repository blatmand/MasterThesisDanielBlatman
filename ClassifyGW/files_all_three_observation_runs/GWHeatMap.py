# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:58:12 2022

@author: Daniel Blatman
This script draws a heat map of the performance of a classical SVM with
a RBF kernel for different kernel parameters C and gamma for the 
classification of gravitational wave events. In the feature
extraction step using the library TSFEL temporal featrures and 1 wavelet 
feature are used. The heat map drawn is saved under the name
'balanced_accuracies_classical_SVC_GW.png'


"""
train_multiplicity=7
import matplotlib.pyplot as plt

CLASSICAL_HEATMAP_ROWS_NUM=10
CLASSICAL_HEATMAP_COL_NUM=10

import seaborn as sns;
def getAccuracySVC(Cparam,gammaParam): 
    '''
    Calculates the balanced accuracy of a SVM with RBF kernel
    having hyperparameters Cparam and gammaParam.

    Parameters
    ----------
    Cparam : float
        Hyperparameter C of the RBF kernel SVM.
    gammaParam : float
        Hyperparameter gamma of the RBF kernel.

    Returns
    -------
    accuracy_test_svc : float
    balanced accuracy of the model    
        

    '''
    C=Cparam
    gamma=gammaParam
    model=SVC(C=C,gamma=gamma,kernel='rbf')
    model.fit(nX_train,y_train)
    results_labels_test_long_signal = []
    results_accuracy_test_svc_long_signal = []
    for i in range(len(events)):
        results_labels_test_long_signal.append(model.predict(results_nX_test_long_signal[i]))
        results_accuracy_test_svc_long_signal.append(metrics.balanced_accuracy_score(y_true=results_y_test_long_signal[i], y_pred=results_labels_test_long_signal[i]))

    return sum(results_accuracy_test_svc_long_signal)/len(results_accuracy_test_svc_long_signal)
def buildHeatMapData(nx,ny):  
    '''
    Puts data into the results_dict global variable.  This function
    creates the data for the classical heat map.

    Parameters
    ----------
    nx : integer
        Number of rows in the classical heat map.
    ny : integer
        Number of columns in the classical heat map.
        
    Returns
    -------
    None.    

    '''
    for i in range(nx):
        
        gamma=0.125*i+0.125
        
        gammaValues.append(gamma)
    for j in range(ny):
        
        Cparam = j/4.0+1.0
        CValues.append(Cparam)
    for i in range(nx):
        for j in range(ny):
            key= str(i)+"-"+str(j)
            result = getAccuracySVC(CValues[j], gammaValues[i])    
            results_dict[key] = result


# process raw data from Ligo
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from scipy import signal
import tsfel
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics

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

attribute=None
cfg_file = tsfel.get_features_by_domain(attribute)
for key in cfg_file['statistical'].keys():
    cfg_file['statistical'][key]['use']='no'   
for key in cfg_file['spectral'].keys():
    cfg_file['spectral'][key]['use']='no'  
                     

cfg_file['spectral']['Wavelet absolute mean']['use']='yes'

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

filename_ids = ["s0.001"]
legend=["1E-3"]
total_balanced_accuracies = []



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
    train_events = ['GW150914','GW151012','GW151226','GW170814','GW170608','GW170809','GW170818'] 
    
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
                for k in range(train_multiplicity):
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
    
    
    gammaValues = []
    CValues = []
    results_dict = {}
    
    y_train = train_labels 
    buildHeatMapData(CLASSICAL_HEATMAP_ROWS_NUM,CLASSICAL_HEATMAP_COL_NUM)    

    
    data=[]
    for i in range(len(gammaValues)):
        row=[]
        for j in range(len(CValues)):
            key= str(j)+"-"+str(i)
            row.append(results_dict[key])
        data.append(row)
                        
    gammaValues_labels=[] 
    length=len(gammaValues)-1
    for ii in range(len(gammaValues)):
        if((ii) % 1 == 0 or ii==length):
            gammaValues_labels.append('{0:.4g}'.format(gammaValues[ii]))
        else:
            gammaValues_labels.append('')     
    CValues_labels=[] 

    length=len(CValues)-1
    for ii in range(len(CValues)):
        if((ii) % 1 == 0 or ii==length):
            CValues_labels.append('{0:.4g}'.format(CValues[ii]))
        else:
            CValues_labels.append('')     
    
    
    fig = plt.figure(figsize=(12,8))
    
    plt.rc('font', size=16)  
    #https://stackoverflow.com/questions/42092218/how-to-add-a-label-to-seaborn-heatmap-color-bar
    ax = sns.heatmap(data,cmap="coolwarm",xticklabels=gammaValues_labels
    ,yticklabels=CValues_labels,vmax=0.8
    ,cbar_kws={'label': 'balanced accuracy'},annot=True, fmt="0.4f")
    ax.invert_yaxis()
    plt.ylabel('C values')
    plt.xlabel('gamma')
    ax.set_title('Average balanced accuracy for gravitational wave events')  
    plt.tight_layout()  
    plt.savefig('balanced_accuracies_classical_SVC_GW')
    #https://stackoverflow.com/questions/332289/how-do-i-change-the-size-of-figures-drawn-with-matplotlib
    plt.close()