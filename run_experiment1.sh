#!bin/bash

# Experiment 1.1: select best params for each algorithm.
# (This section is commented, since it is very heavy with 
# respect to computation. If you want to run comment the 
# following lines.)

# for var in voice statistics sensors; do #
	# for mode in SV KNN RF; do
	
		# echo "TEST Supervisado " $var $mode
		# python code/exp1.1_Supervised.py $var $mode
		
		# echo "TEST UnSupervisado " $var $mode
		# python code/exp1.1_unSupervised.py $var $mode
		
	# done
# done




# Experiment 1.2: Results with the best params

for var in voice statistics sensors; do #
	for mode in SV KNN RF; do
	
		echo "TEST Supervisado " $var $mode
		python code/exp1.2_Supervised.py $var $mode
		
		
		echo "TEST UnSupervisado " $var $mode
		python code/exp1.2_unSupervised.py $var $mode
		
	done
done
