#!bin/bash

# Experiment 2: 

for var in voice statistics sensors; do #
	for mode in SV KNN RF; do
	
		echo "TEST Supervisado " $var $mode
		python code/exp2_Supervised.py $var $mode
		
		
		echo "TEST UnSupervisado " $var $mode
		python code/exp2_unSupervised.py $var $mode
		
	done
done
