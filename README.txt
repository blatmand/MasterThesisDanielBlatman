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

See the end of the README file for information on the libraries used.

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


Libraries used:

Among other more standard libraries, the thesis uses in particular the following libraries:
tsfel, pyCBC, qiskit, qiskit-machine-learning, seaborn, gwpy and gwosc.

tsfel: @article{barandas2020tsfel,
  title={TSFEL: Time Series Feature Extraction Library},
  author={Barandas, Mar{\'\i}lia and Folgado, Duarte and Fernandes, Let{\'\i}cia and Santos, Sara and Abreu, Mariana and Bota, Patr{\'\i}cia and Liu, Hui and Schultz, Tanja and Gamboa, Hugo},
  journal={SoftwareX},
  volume={11},
  pages={100456},
  year={2020},
  publisher={Elsevier}
}

pyCBC: Nitz, A., Harry, I., Brown, D., Biwer, C. M., Willis, J., Canton, T. D., Capano, C., Dent, T., Pekowsky, L.,
Williamson, A. R., Davies, G. S. C., De, S., Cabero, M., Machenschalk, B., Kumar, P., Macleod, D.,
Reyes, S., dfinstad, Pannarale, F., . . . Gadre, B. U. V. (2021). Gwastro/pycbc: Release v1.18.3 of pycbc
(Version v1.18.3). Zenodo. https://doi.org/10.5281/zenodo.5256134

qiskit: ANIS, M. S., Abby-Mitchell, Abraham, H., AduOffei, Agarwal, R., Agliardi, G., Aharoni, M., Akhalwaya, I. Y.,
Aleksandrowicz, G., Alexander, T., Amy, M., Anagolum, S., Anthony-Gandon, Arbel, E., Asfaw, A.,
Athalye, A., Avkhadiev, A., Azaustre, C., BHOLE, P., . . . ˇCepulkovskis, M. (2021). Qiskit: An opensource
framework for quantum computing. https://doi.org/10.5281/zenodo.2573505

seaborn: @article{Waskom2021,
    doi = {10.21105/joss.03021},
    url = {https://doi.org/10.21105/joss.03021},
    year = {2021},
    publisher = {The Open Journal},
    volume = {6},
    number = {60},
    pages = {3021},
    author = {Michael L. Waskom},
    title = {seaborn: statistical data visualization},
    journal = {Journal of Open Source Software}
 }

@article{gwpy,
    title = "{GWpy: A Python package for gravitational-wave astrophysics}",
   author = {{Macleod}, D.~M. and {Areeda}, J.~S. and {Coughlin}, S.~B. and {Massinger}, T.~J. and {Urban}, A.~L.},
  journal = {SoftwareX},
   volume = 13,
    pages = 100657,
     year = 2021,
     issn = {2352-7110},
      doi = {10.1016/j.softx.2021.100657},
      url = {https://www.sciencedirect.com/science/article/pii/S2352711021000029},
}

gwosc: R. Abbott et al. (LIGO Scientific Collaboration and Virgo Collaboration), "Open data from the first and second observing runs of Advanced LIGO and Advanced Virgo", SoftwareX 13 (2021) 100658

usage of gravitational wave open science center data:
"This research has made use of data or software obtained from the Gravitational Wave Open Science Center (gw-openscience.org), a service of LIGO Laboratory, the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA. LIGO Laboratory and Advanced LIGO are funded by the United States National Science Foundation (NSF) as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the construction of Advanced LIGO and construction and operation of the GEO600 detector. Additional support for Advanced LIGO was provided by the Australian Research Council. Virgo is funded, through the European Gravitational Observatory (EGO), by the French Centre National de Recherche Scientifique (CNRS), the Italian Istituto Nazionale di Fisica Nucleare (INFN) and the Dutch Nikhef, with contributions by institutions from Belgium, Germany, Greece, Hungary, Ireland, Japan, Monaco, Poland, Portugal, Spain. The construction and operation of KAGRA are funded by Ministry of Education, Culture, Sports, Science and Technology (MEXT), and Japan Society for the Promotion of Science (JSPS), National Research Foundation (NRF) and Ministry of Science and ICT (MSIT) in Korea, Academia Sinica (AS) and the Ministry of Science and Technology (MoST) in Taiwan." 




