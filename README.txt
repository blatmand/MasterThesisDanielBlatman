README for using the source code of the Master thesis of Daniel Blatman named
"Classical and quantum kernels and applications to
gravitational waves" (2022).

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
unpack the 2 zip files (part1.zip and part2.zip) located in
ClassifyGW\files_all_three_observation_runs and place their content 
into the same folder they where located, that means into the folder
ClassifyGW\files_all_three_observation_runs.
These zip files are the training files 
for the ML pipeline which uses support vector machines.

To run the scripts in Windows >= 10 using Anaconda prompt
and using a virtual environment "GWpyEnv" do the following:
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

The positive and negative training data files are provided (the 2 zip files part1.zip and part2.zip)
but if wished to recreate these files
one can do the following using Anaconda in Ubuntu:
1)	conda create -n name_env python=3.7.11
2)	conda activate name_env
3)	conda install -c conda-forge gwpy==2.1.2
4)	pip install pycbc==1.18.3

To recreate the positive simulation files, run the python script different_noise_types.py
in Ubuntu located in ClassifyGW\files_all_three_observation_runs.
This will create txt data files which start with the prefix "gwpy_hc_mass".

To recreate the negative SVM examples:
In windows >=10 run the python script creating_all_training_noise_files.py 
located in ClassifyGW\files_all_three_observation_runs.
This will create files starting with "different_noise_long".

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

pyCBC: Alex Nitz, Ian Harry, Duncan Brown, Christopher M. Biwer, Josh Willis, Tito Dal Canton, Collin Capano, Thomas Dent, Larne Pekowsky, Andrew R. Williamson, Gareth S Cabourn Davies, Soumi De, Miriam Cabero, Bernd Machenschalk, Prayush Kumar, Duncan Macleod, Steven Reyes, dfinstad, Francesco Pannarale, … Bhooshan Uday Varsha Gadre. (2021). gwastro/pycbc: Release v1.18.3 of PyCBC (v1.18.3). Zenodo. https://doi.org/10.5281/zenodo.5256134

@misc{ Qiskit,
       author = {MD SAJID ANIS and Abby-Mitchell and H{\'e}ctor Abraham and AduOffei and Rochisha Agarwal and Gabriele Agliardi and Merav Aharoni and Vishnu Ajith and Ismail Yunus Akhalwaya and Gadi Aleksandrowicz and Thomas Alexander and Matthew Amy and Sashwat Anagolum and Anthony-Gandon and Eli Arbel and Abraham Asfaw and Anish Athalye and Artur Avkhadiev and Carlos Azaustre and PRATHAMESH BHOLE and Abhik Banerjee and Santanu Banerjee and Will Bang and Aman Bansal and Panagiotis Barkoutsos and Ashish Barnawal and George Barron and George S. Barron and Luciano Bello and Yael Ben-Haim and M. Chandler Bennett and Daniel Bevenius and Dhruv Bhatnagar and Prakhar Bhatnagar and Arjun Bhobe and Paolo Bianchini and Lev S. Bishop and Carsten Blank and Sorin Bolos and Soham Bopardikar and Samuel Bosch and Sebastian Brandhofer and Brandon and Sergey Bravyi and Nick Bronn and Bryce-Fuller and David Bucher and Artemiy Burov and Fran Cabrera and Padraic Calpin and Lauren Capelluto and Jorge Carballo and Gin{\'e}s Carrascal and Adam Carriker and Ivan Carvalho and Adrian Chen and Chun-Fu Chen and Edward Chen and Jielun (Chris) Chen and Richard Chen and Franck Chevallier and Kartik Chinda and Rathish Cholarajan and Jerry M. Chow and Spencer Churchill and CisterMoke and Christian Claus and Christian Clauss and Caleb Clothier and Romilly Cocking and Ryan Cocuzzo and Jordan Connor and Filipe Correa and Zachary Crockett and Abigail J. Cross and Andrew W. Cross and Simon Cross and Juan Cruz-Benito and Chris Culver and Antonio D. C{\'o}rcoles-Gonzales and Navaneeth D and Sean Dague and Tareq El Dandachi and Animesh N Dangwal and Jonathan Daniel and Marcus Daniels and Matthieu Dartiailh and Abd{\'o}n Rodr{\'\i}guez Davila and Faisal Debouni and Anton Dekusar and Amol Deshmukh and Mohit Deshpande and Delton Ding and Jun Doi and Eli M. Dow and Patrick Downing and Eric Drechsler and Eugene Dumitrescu and Karel Dumon and Ivan Duran and Kareem EL-Safty and Eric Eastman and Grant Eberle and Amir Ebrahimi and Pieter Eendebak and Daniel Egger and ElePT and Emilio and Alberto Espiricueta and Mark Everitt and Davide Facoetti and Farida and Paco Mart{\'\i}n Fern{\'a}ndez and Samuele Ferracin and Davide Ferrari and Axel Hern{\'a}ndez Ferrera and Romain Fouilland and Albert Frisch and Andreas Fuhrer and Bryce Fuller and MELVIN GEORGE and Julien Gacon and Borja Godoy Gago and Claudio Gambella and Jay M. Gambetta and Adhisha Gammanpila and Luis Garcia and Tanya Garg and Shelly Garion and James R. Garrison and Jim Garrison and Tim Gates and Hristo Georgiev and Leron Gil and Austin Gilliam and Aditya Giridharan and Glen and Juan Gomez-Mosquera and Gonzalo and Salvador de la Puente Gonz{\'a}lez and Jesse Gorzinski and Ian Gould and Donny Greenberg and Dmitry Grinko and Wen Guan and Dani Guijo and John A. Gunnels and Harshit Gupta and Naman Gupta and Jakob M. G{\"u}nther and Mikael Haglund and Isabel Haide and Ikko Hamamura and Omar Costa Hamido and Frank Harkins and Kevin Hartman and Areeq Hasan and Vojtech Havlicek and Joe Hellmers and {\L}ukasz Herok and Stefan Hillmich and Hiroshi Horii and Connor Howington and Shaohan Hu and Wei Hu and Chih-Han Huang and Junye Huang and Rolf Huisman and Haruki Imai and Takashi Imamichi and Kazuaki Ishizaki and Ishwor and Raban Iten and Toshinari Itoko and Alexander Ivrii and Ali Javadi and Ali Javadi-Abhari and Wahaj Javed and Qian Jianhua and Madhav Jivrajani and Kiran Johns and Scott Johnstun and Jonathan-Shoemaker and JosDenmark and JoshDumo and John Judge and Tal Kachmann and Akshay Kale and Naoki Kanazawa and Jessica Kane and Kang-Bae and Annanay Kapila and Anton Karazeev and Paul Kassebaum and Tobias Kehrer and Josh Kelso and Scott Kelso and Hugo van Kemenade and Vismai Khanderao and Spencer King and Yuri Kobayashi and Kovi11Day and Arseny Kovyrshin and Rajiv Krishnakumar and Pradeep Krishnamurthy and Vivek Krishnan and Kevin Krsulich and Prasad Kumkar and Gawel Kus and Ryan LaRose and Enrique Lacal and Rapha{\"e}l Lambert and Haggai Landa and John Lapeyre and Joe Latone and Scott Lawrence and Christina Lee and Gushu Li and Tan Jun Liang and Jake Lishman and Dennis Liu and Peng Liu and Lolcroc and Abhishek K M and Liam Madden and Yunho Maeng and Saurav Maheshkar and Kahan Majmudar and Aleksei Malyshev and Mohamed El Mandouh and Joshua Manela and Manjula and Jakub Marecek and Manoel Marques and Kunal Marwaha and Dmitri Maslov and Pawe{\l} Maszota and Dolph Mathews and Atsushi Matsuo and Farai Mazhandu and Doug McClure and Maureen McElaney and Cameron McGarry and David McKay and Dan McPherson and Srujan Meesala and Dekel Meirom and Corey Mendell and Thomas Metcalfe and Martin Mevissen and Andrew Meyer and Antonio Mezzacapo and Rohit Midha and Daniel Miller and Hannah Miller and Zlatko Minev and Abby Mitchell and Nikolaj Moll and Alejandro Montanez and Gabriel Monteiro and Michael Duane Mooring and Renier Morales and Niall Moran and David Morcuende and Seif Mostafa and Mario Motta and Romain Moyard and Prakash Murali and Daiki Murata and Jan M{\"u}ggenburg and Tristan NEMOZ and David Nadlinger and Ken Nakanishi and Giacomo Nannicini and Paul Nation and Edwin Navarro and Yehuda Naveh and Scott Wyman Neagle and Patrick Neuweiler and Aziz Ngoueya and Thien Nguyen and Johan Nicander and Nick-Singstock and Pradeep Niroula and Hassi Norlen and NuoWenLei and Lee James O'Riordan and Oluwatobi Ogunbayo and Pauline Ollitrault and Tamiya Onodera and Raul Otaolea and Steven Oud and Dan Padilha and Hanhee Paik and Soham Pal and Yuchen Pang and Ashish Panigrahi and Vincent R. Pascuzzi and Simone Perriello and Eric Peterson and Anna Phan and Kuba Pilch and Francesco Piro and Marco Pistoia and Christophe Piveteau and Julia Plewa and Pierre Pocreau and Alejandro Pozas-Kerstjens and Rafa{\l} Pracht and Milos Prokop and Viktor Prutyanov and Sumit Puri and Daniel Puzzuoli and Pythonix and Jes{\'u}s P{\'e}rez and Quant02 and Quintiii and Rafey Iqbal Rahman and Arun Raja and Roshan Rajeev and Isha Rajput and Nipun Ramagiri and Anirudh Rao and Rudy Raymond and Oliver Reardon-Smith and Rafael Mart{\'\i}n-Cuevas Redondo and Max Reuter and Julia Rice and Matt Riedemann and Rietesh and Drew Risinger and Pedro Rivero and Marcello La Rocca and Diego M. Rodr{\'\i}guez and RohithKarur and Ben Rosand and Max Rossmannek and Mingi Ryu and Tharrmashastha SAPV and Nahum Rosa Cruz Sa and Arijit Saha and Abdullah Ash- Saki and Sankalp Sanand and Martin Sandberg and Hirmay Sandesara and Ritvik Sapra and Hayk Sargsyan and Aniruddha Sarkar and Ninad Sathaye and Niko Savola and Bruno Schmitt and Chris Schnabel and Zachary Schoenfeld and Travis L. Scholten and Eddie Schoute and Mark Schulterbrandt and Joachim Schwarm and James Seaward and Sergi and Ismael Faro Sertage and Kanav Setia and Freya Shah and Nathan Shammah and Will Shanks and Rohan Sharma and Yunong Shi and Jonathan Shoemaker and Adenilton Silva and Andrea Simonetto and Deeksha Singh and Divyanshu Singh and Parmeet Singh and Phattharaporn Singkanipa and Yukio Siraichi and Siri and Jes{\'u}s Sistos and Iskandar Sitdikov and Seyon Sivarajah and Slavikmew and Magnus Berg Sletfjerding and John A. Smolin and Mathias Soeken and Igor Olegovich Sokolov and Igor Sokolov and Vicente P. Soloviev and SooluThomas and Starfish and Dominik Steenken and Matt Stypulkoski and Adrien Suau and Shaojun Sun and Kevin J. Sung and Makoto Suwama and Oskar S{\l}owik and Hitomi Takahashi and Tanvesh Takawale and Ivano Tavernelli and Charles Taylor and Pete Taylour and Soolu Thomas and Kevin Tian and Mathieu Tillet and Maddy Tod and Miroslav Tomasik and Caroline Tornow and Enrique de la Torre and Juan Luis S{\'a}nchez Toural and Kenso Trabing and Matthew Treinish and Dimitar Trenev and TrishaPe and Felix Truger and Georgios Tsilimigkounakis and Davindra Tulsi and Do{\u{g}}ukan Tuna and Wes Turner and Yotam Vaknin and Carmen Recio Valcarce and Francois Varchon and Adish Vartak and Almudena Carrera Vazquez and Prajjwal Vijaywargiya and Victor Villar and Bhargav Vishnu and Desiree Vogt-Lee and Christophe Vuillot and James Weaver and Johannes Weidenfeller and Rafal Wieczorek and Jonathan A. Wildstrom and Jessica Wilson and Erick Winston and WinterSoldier and Jack J. Woehr and Stefan Woerner and Ryan Woo and Christopher J. Wood and Ryan Wood and Steve Wood and James Wootton and Matt Wright and Lucy Xing and Jintao YU and Bo Yang and Unchun Yang and Jimmy Yao and Daniyar Yeralin and Ryota Yonekura and David Yonge-Mallo and Ryuhei Yoshida and Richard Young and Jessie Yu and Lebin Yu and Yuma-Nakamura and Christopher Zachow and Laura Zdanski and Helena Zhang and Iulia Zidaru and Bastian Zimmermann and Christa Zoufal and aeddins-ibm and alexzhang13 and b63 and bartek-bartlomiej and bcamorrison and brandhsn and chetmurthy and deeplokhande and dekel.meirom and dime10 and dlasecki and ehchen and ewinston and fanizzamarco and fs1132429 and gadial and galeinston and georgezhou20 and georgios-ts and gruu and hhorii and hhyap and hykavitha and itoko and jeppevinkel and jessica-angel7 and jezerjojo14 and jliu45 and johannesgreiner and jscott2 and klinvill and krutik2966 and ma5x and michelle4654 and msuwama and nico-lgrs and nrhawkins and ntgiwsvp and ordmoj and sagar pahwa and pritamsinha2304 and rithikaadiga and ryancocuzzo and saktar-unr and saswati-qiskit and septembrr and sethmerkel and sg495 and shaashwat and smturro2 and sternparky and strickroman and tigerjack and tsura-crisaldo and upsideon and vadebayo49 and welien and willhbang and wmurphy-collabstar and yang.luh and Mantas {\v{C}}epulkovskis},
       title = {Qiskit: An Open-source Framework for Quantum Computing},
       year = {2021},
       doi = {10.5281/zenodo.2573505}
}

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

acknoledgement of usage of gravitational wave open science center data:
"This research has made use of data or software obtained from the Gravitational Wave Open Science Center (gw-openscience.org), a service of LIGO Laboratory, the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA. LIGO Laboratory and Advanced LIGO are funded by the United States National Science Foundation (NSF) as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the construction of Advanced LIGO and construction and operation of the GEO600 detector. Additional support for Advanced LIGO was provided by the Australian Research Council. Virgo is funded, through the European Gravitational Observatory (EGO), by the French Centre National de Recherche Scientifique (CNRS), the Italian Istituto Nazionale di Fisica Nucleare (INFN) and the Dutch Nikhef, with contributions by institutions from Belgium, Germany, Greece, Hungary, Ireland, Japan, Monaco, Poland, Portugal, Spain. The construction and operation of KAGRA are funded by Ministry of Education, Culture, Sports, Science and Technology (MEXT), and Japan Society for the Promotion of Science (JSPS), National Research Foundation (NRF) and Ministry of Science and ICT (MSIT) in Korea, Academia Sinica (AS) and the Ministry of Science and Technology (MoST) in Taiwan." 




