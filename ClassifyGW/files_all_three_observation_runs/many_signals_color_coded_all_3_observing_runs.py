# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:21:29 2022

@author: Daniel Blatman

This script trains a classical SVM and calculates the
average balanced accuracies for selected events. It also creates a massplot 
visualizing the performance of the classical algorithm in recognizing
gravitational waves from different advanced LIGO events and plots visualizing the
performance of the classical SVM for different events.
The plots are saved as png files. This script utilizes a "train_multiplicity"
parameter of 1 and temporal as well as wavelet features. The classification is
done on time series containing event from all 3 observation runs of
advanced LIGO Hanford time series. The training is done using simulated signals
with background noise from before events from all 3 observation runs
of advanced LIGO.
"""

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

def get_label(index1,index2,long_data_test,gps):
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
                     
cfg_file['spectral']['Wavelet energy']['use']='yes'
cfg_file['spectral']['Wavelet entropy']['use']='yes'
cfg_file['spectral']['Wavelet standard deviation']['use']='yes'
cfg_file['spectral']['Wavelet variance']['use']='yes'
cfg_file['spectral']['Wavelet absolute mean']['use']='yes'

multiplicity = 1
events=[
        'GW150914','GW151012','GW151226','GW170814','GW170823'
                ,'GW170729','GW170104','GW170608','GW170809','GW170817'
                ,'GW170818'
                ,
                
                'GW190408_181802','GW190412','GW190413_052954','GW190413_134308',
                'GW190421_213856',
                'GW190503_185404','GW190512_180714','GW190513_205428','GW190514_065416','GW190517_055101',
                'GW190519_153544','GW190521','GW190521_074359','GW190527_092055',
               'GW190701_203306','GW190706_222641',
                'GW190707_093326'
                ,'GW190719_215514','GW190720_000836','GW190727_060333',
                'GW190728_064510','GW190731_140936','GW190803_022701',
                'GW190828_063405','GW190828_065509',
                'GW190915_235702','GW190924_021846','GW190929_012149',
                
                'GW191103_012549','GW191105_143521','GW191109_010717','GW191113_071753',
                'GW191126_115259','GW191127_050227','GW191129_134029','GW191204_110529',
                'GW191204_171526','GW191215_223052','GW191216_213338','GW191219_163120',
                'GW191222_033537',
                'GW191230_180458',
                'GW200115_042309','GW200128_022011','GW200129_065458','GW200202_154313',
                'GW200208_130117','GW200208_222617','GW200209_085452','GW200210_092254',
                'GW200216_220804','GW200219_094415','GW200220_061928','GW200220_124850',
                'GW200224_222234','GW200225_060421','GW200302_015811','GW200306_093714',
                'GW200311_115853','GW200316_215756'
        ]

def create_filename(i,j,file_id,train_event):
    filename = "gwpy_hc_mass1_"+str(i)+"_mass2_"+str(j)+"real_data_noise_"+file_id+"_from_"+train_event+"_.txt"
    return filename 
def process_event(event_id):
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

filename_ids = ["s0.1","s0.01","s0.005","s0.001","s0.0005","s0.0001"]
legend=["1E-1","1E-2","5E-3","1E-3","5E-4","1E-4"]
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
    fft_data_segments_train=[]
    filenames_train=[]
    
    train_events = ['GW150914','GW151012','GW151226','GW170814','GW170608','GW170809','GW170818',
                    'GW190408_181802','GW190412','GW190413_052954','GW190413_134308'
                    ,'GW200306_093714','GW200311_115853','GW200316_215756'] 
    for iii in range(len(train_events)):
        
        data_train = TimeSeries.read('different_noise_long_training_data_noise_bp_and_notch_filtered_'+train_events[iii]+'.txt')
        length = 25.0
        
        fs = int(hclean1.sample_rate.value)
    
        start_index = 0
        end_index = int(fs*0.25)
        train_window = signal.windows.tukey(data_train[start_index:int(end_index)].size)
    
        for k in range(int(length*4)):
            data_segment_train= data_train[start_index:int(end_index)]
            hwin_train = data_segment_train * train_window
            fft_data_segments_train.append(hwin_train.fft().abs())
            start_index = start_index + int(0.25*fs)
            end_index = end_index +int(0.25*fs)
            data_segments_train.append(hwin_train)
        
        
        num=10
    for train_event in train_events:   
        for i in range(num):
            for j in range(num):
                for k in range(multiplicity):
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
        
import matplotlib.pyplot as plt  
plt.rc('font', size=15)    
if(attribute==None):
    attribute="temp+wavelet"

masses_dict = {
        'GW150914': [35.6,30.6],
                'GW151012': [23.2,13.6],
                'GW151226': [13.7,7.7],
                'GW170814': [30.6,25.2],
                'GW170823': [39.5,29.0],
                'GW170729': [50.2,34.0],
                'GW170104': [30.8,20.0],
                'GW170608': [11.0,7.6],
                'GW170809': [35.0,23.8],
                'GW170817': [1.46,1.27],
                'GW170818': [35.4,26.7],

                'GW190408_181802':[24.6,18.4],
                'GW190412': [30.1,8.3],
                'GW190413_052954': [34.7,23.7],
                'GW190413_134308': [47.5,31.8],
                'GW190421_213856': [41.3,31.9],
                'GW190503_185404': [43.3,28.4],
                'GW190512_180714': [23.3,12.6],
                'GW190513_205428': [35.7,18.0],
                'GW190514_065416': [39.0,28.4],
                'GW190517_055101': [37.4,25.3],
                'GW190519_153544': [66.0,40.5],
                'GW190521': [95.3,69.0],
                'GW190521_074359': [42.2,32.8],
                'GW190527_092055': [36.5,22.6],
                'GW190701_203306': [53.9,40.8],
                'GW190706_222641': [67.0,38.2],
                'GW190707_093326': [11.6,8.4],
                'GW190719_215514': [36.5,20.8],
                'GW190720_000836': [13.4,7.8],
                'GW190727_060333': [38.0,29.4],
                'GW190728_064510': [12.3,8.1],
                'GW190731_140936': [41.5,28.8],
                'GW190803_022701': [37.3,27.3],
                'GW190828_063405': [32.1,26.2],
                'GW190828_065509': [24.1,10.2],
                'GW190915_235702': [35.3,24.4],
                'GW190924_021846': [8.9,5.0],
                'GW190929_012149': [80.8,24.1],
                
                'GW191103_012549': [11.8,7.9],
                'GW191105_143521': [10.7,7.7],
                'GW191109_010717': [65,47],
                'GW191113_071753': [29,5.9],
                'GW191126_115259': [12.1,8.3],
                'GW191127_050227': [53,24],
                'GW191129_134029': [10.7,6.7],
                'GW191204_110529': [27.3,19.3],
                'GW191204_171526': [11.9,8.2],
                'GW191215_223052': [24.9,18.1],
                'GW191216_213338': [12.1,7.7],
                'GW191219_163120': [31.1,1.17],
                'GW191222_033537': [45.1,34.7],
                'GW191230_180458': [49.4,37],
                'GW200115_042309': [5.9,1.44],
                'GW200128_022011': [42.2,32.6],
                'GW200129_065458': [34.5,28.9],
                'GW200202_154313': [10.1,7.3],
                'GW200208_130117': [37.8,27.4],
                'GW200208_222617': [51,12.3],
                'GW200209_085452': [35.6,27.1],
                'GW200210_092254': [24.1,2.83],
                'GW200216_220804': [51,30],
                'GW200219_094415': [37.5,27.9],
                'GW200220_061928': [87,61],
                'GW200220_124850': [38.9,27.9],
                'GW200224_222234': [40.0,32.5],
                'GW200225_060421': [19.3,14.0],
                'GW200302_015811': [37.8,20.0],
                'GW200306_093714': [28.3,14.8],
                'GW200311_115853': [34.2,27.7],
                'GW200316_215756': [13.1,7.8]
        
        }

prob = total_balanced_accuracies[4]

fig= plt.figure(figsize=(12,6))

ax = plt.gca()
for i in range(len(prob)):
    masses_dict[events[i]].append(prob[i])
    
X= [] 
Y= []
Z = []
for i in range(len(prob)):
    X.append(masses_dict[events[i]][0])
    Y.append(masses_dict[events[i]][1])
    Z.append(masses_dict[events[i]][2])
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)    
cmap = plt.get_cmap('jet', 20)
cmap.set_under('gray')

const_x_left = np.array([31,31,31,31,31,31,31,31,31,31])
y_up= np.array([32,32,32,32,32,32,32,32,32,32])   
const_x_right = np.array([40,40,40,40,40,40,40,40,40,40])
const_y=np.array([23,23,23,23,23,23,23,23,23,23])
y_values = np.arange(23,33,1)
y_values_up = np.arange(31,41,1)
y_values_down = np.arange(23,33,1)


plt.title("Event-overview for s = " +legend[4])
for i in range(len(prob)):
    offset=0
    if(events[i]=='GW151226'):
        offset=-2
    if(events[i]=='GW170608'):
        offset=+1
    p= [masses_dict[events[i]][0],masses_dict[events[i]][1]]
    
    ###################################################################
    plt.annotate(i+1,(p[0],p[1]), fontsize=10) 
    if(i<30):
        plt.annotate(str(i+1)+": "+events[i],(0.5,70-2*i),fontsize=10)   
    elif(i<49): 
        plt.annotate(str(i+1)+": "+events[i],(18.5,69-2*(i-30.5)),fontsize=10) 
    elif(i<63):
        plt.annotate(str(i+1)+": "+events[i],(36.5,69-2*(i-49.5)),fontsize=10) 
    else:     
        plt.annotate(str(i+1)+": "+events[i],(54.5,69-2*(i-63.5)),fontsize=10) 
    ####################################################################
cax=plt.scatter(X,Y,c=Z,cmap=cmap, vmin=0.0, vmax=1.0)    
colorbar = plt.colorbar(cax,ticks=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
colorbar.set_label('balanced accuracy', rotation=270,labelpad=15)
plt.plot(const_x_left, y_values, ':r');  # dotted red    
plt.plot(const_x_right, y_values, ':r');  # dotted red    
plt.plot(y_values_up, y_up, ':r');  # dotted red 
plt.plot(y_values_up, const_y, ':r');  # dotted red

#https://datavizpyr.com/how-to-draw-a-rectangle-on-a-plot-in-matplotlib/    
left, bottom, width, height = (31, 23, num-1, num-1)
from matplotlib import patches
rect = patches.Rectangle((left,bottom),width,height, 
                        alpha=0.1,
                       facecolor="cyan")
plt.gca().add_patch(rect)    
plt.xlabel('$m_1$[Solar masses]')
plt.ylabel('$m_2$[Solar masses]')     
plt.tight_layout()
plt.savefig("All_observing_runs_many_signals_masses_plot_C_"+str(C)+"gamma_"+str(gamma))
plt.close()
    
    
#calculate average accuracy for each file id:   
average_accuracy=[]
for i in range(len(total_balanced_accuracies)):
    average_accuracy.append(sum(total_balanced_accuracies[i])/len(total_balanced_accuracies[i]))
plt.figure(figsize=(12,6))
plt.plot(legend,average_accuracy ,'o')
#https://stackoverflow.com/questions/6282058/writing-numerical-values-on-the-plot-with-matplotlib
for i,j in zip(legend,average_accuracy):
    number= '{0:.3g}'.format(j)
    plt.annotate(number,xy=(i,j))
plt.xlabel("parameter s")
plt.ylabel("average balanced accuracy")
plt.ylim(0,1)
plt.title("Many signals average balanced accuracy "+ str(attribute)+" features"+ " C = "+str(C)+" gamma = "+str(gamma))
plt.savefig("All_observing_runs_many_signals_average_accuracy_"+attribute+"_C_"+str(C)+"gamma_"+str(gamma))  
plt.close()