This is the repository for the paper "Cryptanalysis of ChiLow". This repository contains three folders, each corresponding to one of the three results (three chapters) in the paper.

- Differential and Linear Trails
  - diff_trail_search.cpp: The C++ code for searching the differential trails, which requires the use of Gurobi.    
  - linear_trail_search.cpp: The C++ code for searching the linear trails, which requires the use of Gurobi.
  - differential trails.txt: We have listed the good differential trails that we have found.
  - linear trails.txt: We have listed the good linear trails that we have found.

- Differential-Linear Distinguishers
  - calculate_correlation.cpp: The C++ code is used to calculate the correlation of a given differential-linear pair. The use of OpenMP enables the program to be parallelized.
  - Distinguishers D32.txt: We have listed the good distinguishers for D32.
  - Distinguishers D40.txt: We have listed the good distinguishers for D40.

The search for the entire distinguisher is divided into three parts: the differential trail, the middle connection part, and the linear trail. 
The search for the differential trail and linear trail can be accomplished by diff_trail_search.cpp and linear_trail_search.cpp respectively. 
For the search of the intermediate part, it actually involves calculating the correlation of a differential linear pair, which can be done through calculate_correlation.cpp.

- Cube Attacks
  - Statistics_K (calculate ANF).cpp: The C++ code is used to calculate the ANF of Chilow. The most direct function is to count the key variables that multiply with cube variables after one round.
