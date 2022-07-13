# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:51:11 2022

@author: Daniel Blatman
This script demonstrates the classification of elements of the group
of integers with multiplication modulo a prime p subject to a condition
involving discrete logarithms. This problem can be related to the discrete 
logarithm problem (DLOG) which is assumed to be hard to solve classically and
was proven to exhibit a possible quantum advantage provided a suitable
quantum circuit is used using a fault tolerant quantum computer. In this 
script a simpler quantum circuit is constructed based on the
covariant feature map. The classification results of the quantum circuit 
are plotted for different C values and learning rates in a heat map
and are compared to the classification performance using a SVM with a classical
RBF kernel. The classical results for different parameters C and Gamma
are plotted in a heatmap. Both heatmaps are saved to png files.



"""
import matplotlib.pyplot as plt
QUANTUM_HEATMAP_ROWS_NUM = 3
QUANTUM_HEATMAP_COL_NUM = 3 
CLASSICAL_HEATMAP_ROWS_NUM = 10 
CLASSICAL_HEATMAP_COL_NUM  = 10


import numpy as np
from sklearn.svm import SVC

from qiskit_machine_learning.kernels import QuantumKernel
from MySVCLoss import MySVCLoss 


from sympy.ntheory import discrete_log

seed = 12345 
random_seed = seed
from sklearn import metrics
# using discrete logarithm function from the sympy library
#https://docs.sympy.org/latest/modules/ntheory.html?highlight=discrete#sympy.ntheory.residue_ntheory.discrete_log
prime=7
base=3

s=1
numbers=list(range(1,prime))
   
labels=[]
for number in numbers:
    if(discrete_log(prime, number, base) in range(s, 1+s+int((prime-3)/2))):
        labels.append(1)    
    else: 
        labels.append(-1)  
###############################################
# copied from the stackoverflow website example 
#https://stackoverflow.com/questions/33510979/generator-of-evenly-spaced-points-in-a-circle-in-python
def circle_points(r, n):
    '''
    

    Parameters
    ----------
    r : integer
        radius of a circle.
    n : integer
        number of points on the circle.

    Returns
    -------
    circles : list
        list of points on the circle.

    '''
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles     

    
r = [1]
n = [prime-1]
circles = circle_points(r, n)
fig, ax = plt.subplots()
for circle in circles:
    ax.scatter(circle[:, 0], circle[:, 1])
###############################################    
n=list(range(1,prime))       
logn=[]
for i in n:
    logn.append(discrete_log(prime, i, base))
for i, txt in enumerate(n):
    if (labels[i]==1):
        color=(0,0,1)
    else: color=(1,0,0)
    ax.annotate(txt, (circle[(i), 0], circle[(i),1]),color=color)
ax.set_aspect('equal')
if(prime==7):
    plt.title('$\mathbb{Z}_{7}^*$')
elif(prime==37):
    plt.title('$\mathbb{Z}_{37}^*$')     
name= "Elements_in_Z_" +str(prime)
plt.savefig(name)
plt.close()

sorted_logn = np.sort(logn)
new_labels=[]
for i in sorted_logn:
    index = logn.index(sorted_logn[i])
    new_labels.append(labels[index])
fig, ax = plt.subplots()
for circle in circles:
    ax.scatter(circle[:, 0], circle[:, 1])
for i, txt in enumerate(sorted_logn):
    if (new_labels[i]==1):
        color=(0,0,1)
    else: color=(1,0,0)
    ax.annotate(txt, (circle[(i), 0], circle[(i),1]),color=color)
ax.set_aspect('equal')
if(prime==7):
    plt.title("Discrete logarithm " +'$\mathbb{Z}_{7}^*$')
elif(prime==37):
    plt.title("Discrete logarithm " +'$\mathbb{Z}_{37}^*$')   

name_log= "Log of Elements_in_Z_"+ str(prime)
plt.savefig(name_log)
plt.close()

import math
number_bits= math.ceil(np.log2(prime))
format_prefix='{:0'+str(number_bits)+'b}'


N=[]
for i in range(len(n)):
    formated_string=format_prefix.format(i+1)

    feature_array=[]
    for element in formated_string:
        feature_array.append(int(element))
    N.append(feature_array)    

x_train=np.array(N[0:int(len(n)*0.7)])
y_train = np.array(labels[0:int(len(n)*0.7)])
x_test= np.array(N[int(len(n)*0.7):len(n)])
y_test = np.array(labels[int(len(n)*0.7):len(n)])


from qiskit.utils import algorithm_globals
seed = 12345 
algorithm_globals.random_seed = seed

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# Qiskit imports
from qiskit import BasicAer
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.algorithms import QSVC

# Put this repository on the Python path and import qkt pkgs
module_path = os.path.abspath(os.path.join('../prototype-quantum-kernel-training'))
sys.path.append(module_path)
from qkt.feature_maps import CovariantFeatureMap
from qkt.utils import QKTCallback

num_columns=np.shape(x_train)[1]
# copy last feature for an odd number of features
if(num_columns % 2 == 1):
    last_column_train=x_train[:,-1]
    x_train=np.insert(x_train,num_columns,last_column_train,axis=1)
    
    last_column_test = x_test[:,-1]
    x_test = np.insert(x_test,num_columns,last_column_test,axis=1)
    
num_features = np.shape(x_train)[1]
entangler_map =[]
for index in range(int(num_features/2)-1):
    entangler_map.append([index,index+1])

fm = CovariantFeatureMap(
    feature_dimension=num_features,
    entanglement=entangler_map,
    single_training_parameter=False
)
# Use the state vector simulator as a backend
backend = BasicAer.get_backend('statevector_simulator')


learning_rates = []

results_dict = {}
Cvalues=[]

learning_rates=[]
perturbation=0.2


def getAccuracyQSVC(Cvalue,learning_rate): 
    '''
    Calculates the balanced accuracy of a SVM with the global variable fm,
    which is the feature map used.

    Parameters
    ----------
    Cvalue : float
        Hyperparameter C of the QKA algorithm.
    learning_rate : float
        Learning rate of the SPSA class.

    Returns
    -------
    balanced_accuracy_test : flaot
        Balanced accuracy of the quantum model.

    '''
    seed = 12345 
    algorithm_globals.random_seed = seed
    
    # Instantiate quantum kernel
    quant_kernel = QuantumKernel(fm,
                                 user_parameters=fm.user_parameters,
                                 quantum_instance=backend)
    # Set up the optimizer
    cb_qkt = QKTCallback()
    spsa_opt = SPSA(maxiter=1000,
                    callback=cb_qkt.callback,
                    learning_rate=learning_rate,
                    perturbation=perturbation)
    # Instantiate a quantum kernel trainer.
    qkt = QuantumKernelTrainer(
        quantum_kernel=quant_kernel,
        loss=MySVCLoss(C=Cvalue),
        optimizer=spsa_opt,
        initial_point=[0.1]*len(fm.user_parameters)
        )
    # Train the kernel using QKT directly
    qka_results = qkt.fit(x_train, y_train)
    optimized_kernel = qka_results.quantum_kernel

    # Use QSVC for classification
    qsvc = QSVC(C=Cvalue,quantum_kernel=optimized_kernel)

    # Fit the QSVC
    qsvc.fit(x_train, y_train)

    # Predict the labels
    labels_test = qsvc.predict(x_test)

    # Evalaute the test accuracy
    balanced_accuracy_test = metrics.balanced_accuracy_score(y_true=y_test, y_pred=labels_test)
    # plot the convergence plot
    plot_data = cb_qkt.get_callback_data() # callback data
    plt.figure(figsize=(12,6))
    plt.rcParams['font.size'] = 24
    plt.plot([i+1 for i in range(len(plot_data[0]))],
               np.array(plot_data[2]),
               c='k',
               marker='o'
    )
    learning_rate_without_dot = str(learning_rate).replace(".", "dot")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss evolution, p = "+str(prime)+" , C = "+str(Cvalue)+", learning rate = "+str(learning_rate))
    
    fig.tight_layout()
    plt.savefig("DLOG_loss evolution_p_"+str(prime)+"_C_"+str(Cvalue)+"_learning rate_"+str(learning_rate_without_dot))
    plt.close()
    return balanced_accuracy_test


def buildHeatMapQData(nx,ny): 
    '''
    Puts data into the quantum_results_dict global variable. This function
    creates the data for the quantum heat map.

    Parameters
    ----------
    nx : integer
        Number of rows in the quantum heat map.
    ny : integer
        Number of columns in the quantum heat map.

    Returns
    -------
    None.

    '''
    for i in range(nx):
        learning_rates.append(0.01*(i+1))
    for j in range(ny):        
        Cvalues.append(j+1)
    for i in range(nx):
        for j in range(ny):
            
            result = getAccuracyQSVC(Cvalues[j], learning_rates[i])
            key= str(i)+"-"+str(j)
            results_dict[key] = result
            print("i,j is: "+str(i)+","+str(j))
            print("result is: "+str(result))
buildHeatMapQData(QUANTUM_HEATMAP_ROWS_NUM,QUANTUM_HEATMAP_COL_NUM)  

data=[]
for i in range(len(learning_rates)):
    row=[]
    for j in range(len(Cvalues)):
        key= str(j)+"-"+str(i)
        row.append(results_dict[key])
    data.append(row)
                    
learning_rate_labels=[] 
length=len(learning_rates)-1
for ii in range(len(learning_rates)):
    if((ii) % 1 == 0 or ii==length):
        learning_rate_labels.append('{0:.4g}'.format(learning_rates[ii]))
    else:
        learning_rate_labels.append('')     
Cvalues_labels=[] 
length=len(Cvalues)-1
for ii in range(len(Cvalues)):
    if((ii) % 1 == 0 or ii==length):
        Cvalues_labels.append('{0:.4g}'.format(Cvalues[ii]))
    else:
        Cvalues_labels.append('')   
        
import seaborn as sns
# labels in seaborn plot follow the following example from stackoverflow:
#https://stackoverflow.com/questions/42092218/how-to-add-a-label-to-seaborn-heatmap-color-bar         
fig = plt.figure(figsize=(14,8))
plt.rc('font', size=16)  
ax = sns.heatmap(data,cmap="coolwarm",xticklabels=learning_rate_labels,
                 yticklabels=Cvalues_labels,annot=True, fmt="0.4f",cbar_kws={'label': 'balanced accuracy'})
ax.invert_yaxis()
plt.ylabel('C')
plt.xlabel('learning rate')
ax.set_title('QSVC : p = '+str(prime)+", perturbation = "+str(perturbation))  
plt.tight_layout()  
perturbation_str= str(perturbation).replace(".","dot")
plt.savefig('balanced_accuracies_quantum_SVC_'+str(prime)+"_perturbation_"+perturbation_str)   
plt.close()
        
gammaValues = []
CValues = []
results_dict = {}

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
    model = SVC(C=Cparam,gamma=gammaParam,kernel='rbf')
    model.fit(x_train, y_train)
    labels_test = model.predict(x_test)
    accuracy_test_svc = metrics.balanced_accuracy_score(y_true=y_test, y_pred=labels_test)
    return accuracy_test_svc
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
        
        gamma=2*0.09*i+0.02
        
        gammaValues.append(gamma)
    for j in range(ny):
        
        Cparam = 20*j/8.0+0.2
        CValues.append(Cparam)
    for i in range(nx):
        for j in range(ny):
            result = getAccuracySVC(CValues[j], gammaValues[i])
            key= str(i)+"-"+str(j)
            results_dict[key] = result
buildHeatMapData(CLASSICAL_HEATMAP_ROWS_NUM,CLASSICAL_HEATMAP_COL_NUM)    

fig = plt.figure()
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
    
import seaborn as sns;
fig = plt.figure(figsize=(14,8))
plt.rc('font', size=16)  
ax = sns.heatmap(data,cmap="coolwarm",xticklabels=gammaValues_labels,
                 yticklabels=CValues_labels,cbar_kws={'label': 'balanced accuracy'},annot=True, fmt="0.4f")
ax.invert_yaxis()
plt.ylabel('C')
plt.xlabel('gamma')
ax.set_title('Balanced accuracies classical SVC for prime = '+str(prime))  
plt.tight_layout()  
plt.savefig('DLOG_balanced_accuracies_classical_SVC_'+str(prime))
plt.close()