README for using the source code of the Master thesis of Daniel Blatman.
The folder contains the source code used to produce the figures in the scripts
and the output obtained when the scripts are run. Additionally a file ml_pipeline.png 
is present in the folder ClassifyGW showing schematically the ML pipeline used.
The thesis was done as part of Master studies in AI at Maastricht University in collaboration with 
IBM Research - Zurich. 
Thesis Committee:
Dr. Georgios Stamoulis, Maastricht University
Dr. Domenica Dibenedetto, Maastricht University
Dr. Ivano Tavernelli, IBM Research - Zurich
Dr. Francesco Tacchino, IBM Research - Zurich
Dr. Panagiotis Barkoutsos, IBM Research - Zurich

The thesis uses the 
Quantum kernel trainer prototype GitHub repository downloaded from 
https://github.com/qiskit-community/prototype-quantum-kernel-training
on the 29th of May 2022. This folder is redistributed with this repository, see the licence 
within the folder prototype-quantum-kernel-training.

The usage of filters in the preprocessing steps used to process gravitational waves are
based on the following tutorial on gravitational wave data processing on the GitHub website: 
https://github.com/gw-odw/odw-2018/blob/master/gwpy/2b%20-
%20Signal%20processing%20with%20GWpy.ipynb 
Accessed on: 31-05-2022

Among other libraries, the thesis uses in particular the following libraries:
tsfel, pyCBC, qiskit, qiskit-machine-learning, seaborn, gwpy and gwosc.


To run the scripts in windows download the repository and 
Unpack the 2 zip files (part1.zip and part2.zip) located in
ClassifyGW\files_all_three_observation_runs and place their content 
into the same folder they where located, that means into the folder
ClassifyGW\files_all_three_observation_runs.
These zip files are the training files 
for the ML pipeline which uses support vector machines.

To run the scripts in Windows >= 10 using Anaconda prompt
and using a virtual environment "GWpyEnv":
1)	conda create -n GWpyEnv python=3.7.12
2)	conda activate GWpyEnv
3)	conda install -c conda-forge gwpy==2.1.2
4)	pip install qiskit==0.36.2
5)	pip install qiskit-machine-learning==0.3.1
6) pip install tsfel==0.1.4
7)	pip install seaborn==0.11.2
8) pip install pylatexenc==2.10 optional and only needed for "draw_covariant_quantum_circuit_GW.py"
 	and "draw_covariant_feature_map_DLOG_p_37.py" and "draw_covariant_feature_map_DLOG_p_7.py"
	 for printing latex-source of quantum feature map circuits output.
With this environment it is possible to run the scripts in the folders
DLOG and ClassifyGW.
 
In windows >= 10:
Run the files in the Anaconda command prompt by going into the directory 
where the files are defined and after activating the virtual environment "GWpyEnv" type:

python filename

**********************
Additional information:

The file "MySVCLoss" is an auxiliary file that is used to be able to change the parameter C 
of the quantum SVM using the quantumkerneltrainer object code. 
It is a modified version of Qiskit code as indicated in the file.

The data files are provided but if wished to recreate these files
one can do using Anaconda in Ubuntu:
1)	conda create -n name_env python=3.7.11
2)	conda activate name_env
3)	conda install -c conda-forge gwpy==2.1.2
4)	pip install pycbc==1.18.3

To recreate the positive simulation files, run the python script different_noise_types.py
in Ubuntu. Then copy the created  txt data files which start with the prefix "gwpy_hc_mass" into windows 
into the folder "files_all_three_observation_runs", which was delivered with the project
in the 2 zip files part1.zip and part2.zip. 

To recreate the negative SVM examples:
In windows >=10 run the python script creating_all_training_noise_files.py 
Copy these files (starting with "different_noise_long") into the folder "files_all_three_observation_runs".
(Also already delivered in the zip files part1.zip and part2.zip)

Small notice:
The few lines of code of the function "circle_points(r,n)"
used in some of the plots of the DLOG examples used in the appendix of the thesis are copied from the 
website "https://stackoverflow.com/questions/33510979/generator-of-evenly-spaced-points-in-a-circle-in-python"
as is indicated in the files that define and use this function in the DLOG folder of the code.