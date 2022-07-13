# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:23:14 2022
Draws covariant feature for p = 7 for the classification task that
is related to the discrete logarithm problem.
@author: Daniel Blatman
"""
import matplotlib.pyplot as plt
QUANTUM_HEATMAP_ROWS_NUM = 3
QUANTUM_HEATMAP_COL_NUM = 3 
CLASSICAL_HEATMAP_ROWS_NUM = 10 
CLASSICAL_HEATMAP_COL_NUM  = 10


import numpy as np

from sympy.ntheory import discrete_log

seed = 12345 
random_seed = seed
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

# Put this repository on the Python path and import qkt pkgs
module_path = os.path.abspath(os.path.join('../prototype-quantum-kernel-training'))
sys.path.append(module_path)
from qkt.feature_maps import CovariantFeatureMap

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

print(fm.draw(output='latex_source'))