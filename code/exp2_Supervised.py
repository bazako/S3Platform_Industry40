import glob
import pandas as pd
import os 
import numpy as np
import json
from progressbar import progressbar
import argparse
from utils import calculate_eer, calculate_metrics


from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RepeatedStratifiedKFold

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('datatype', type=str, default='sensors', help='sensors, statistics or voice')
parser.add_argument('model', type=str, default='KNN', help='KNN, RF, SV')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose')

args = parser.parse_args()
print(args)
datatype=args.datatype 
modelName=args.model


pathToFiles='data/14days/'

if modelName == 'SV':
	model = SVC
	if datatype=='sensors':
		dict_params =  {'C': 50, 'kernel': 'rbf','probability':True}
	elif datatype=='statistics':
		dict_params = {'C': 50, 'kernel': 'rbf','probability':True}
	else:
		dict_params = {'C': 10, 'kernel': 'rbf','probability':True}
	
	

# KNN
if modelName == 'KNN':
	model = KNeighborsClassifier
	if datatype=='sensors':
		dict_params = {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 10, 'weights': 'distance'}
	elif datatype=='statistics':
		dict_params = {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 10, 'weights': 'distance'}
	else:
		dict_params = {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 10, 'weights': 'distance'}

# RF
if modelName == 'RF':
	model = RandomForestClassifier
	if datatype=='sensors':
		dict_params =  {'criterion': 'entropy', 'max_depth': 100, 'min_samples_split': 2, 'n_estimators': 100}
	elif datatype=='statistics':
		dict_params = {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
	else:
		dict_params = {'criterion': 'entropy', 'max_depth': 100, 'min_samples_split': 2, 'n_estimators': 100}

scaler = StandardScaler
	

nameModel=modelName #str(model).split('.')[-1].replace("'","").replace('>','')

pathToScores='./scores/scores_exp2_Supervised/'+nameModel+'/'
os.makedirs(pathToScores,exist_ok=True)



datatype=args.datatype
users=glob.glob(pathToFiles+'train/*'+datatype+'.txt')
users.sort()


eer_users=[]
auc_users=[]
f1max_users=[]




file_resume = pathToScores + '/resume_'+datatype+'.txt'
a_file=open(file_resume, "a")

from itertools import combinations
userCombinations = list(combinations(range(0,len(users)), 4))

import random
randomSample = random.sample(range(0, len(userCombinations)-1), 20)

eer_know=[]
auc_know=[]
f1_know=[]

eer_unknow=[]
auc_unknow=[]
f1_unknow=[]
for iter in progressbar(range(0,20)):
	
	user_unknown = userCombinations[randomSample[iter]]
	a_file.write("user unknow"+str(user_unknown[0]) +','+str(user_unknown[1]) + ',' + str(user_unknown[2])+ ',' + str(user_unknown[3]) +"\n")
	
	scores_global=np.empty([0,2])
	labels_global=np.empty([0])
	scores_global_unknow=np.empty([0,2])
	labels_global_unknow=np.empty([0])

	for iuser in range(0,len(users)):	
		
		userfile = users[iuser]
		if args.verbose:
			print(" USER {}-{}: {}".format(iuser, len(users),userfile))
		
		nombreUsuario=userfile.split('/')[-1].split('_')[0]
		numUsuario = int(nombreUsuario.replace('user',''))
		
		if iuser in user_unknown: 
			if args.verbose:
				print("skiping user")
		else:
		
			dataFrameUserTrain=pd.read_csv(userfile,header=None,prefix='X')
			
			if not datatype=='voice':
				X_train_target = dataFrameUserTrain.values[:,1:]
			else:
				X_train_target = dataFrameUserTrain.values[:,2:]
			
			labels_target=np.zeros(X_train_target.shape[0])

			X_train_nontarget=np.empty([0,X_train_target.shape[1]])
			labels_nontarget=np.empty([0])


			for juser in range(0, len(users)): ## ,10):#
				if not iuser == juser:
					userfile_non = users[juser]
					nombreUsuario=userfile_non.split('/')[-1].split('_')[0]
					numUsuario = int(nombreUsuario.replace('user',''))
					
					if not juser in user_unknown:
						# contUser+=1
						userfile_non = users[juser]
						dataFrameUserTrain_non=pd.read_csv(userfile_non,header=None,prefix='X')

						if not datatype=='voice':
							X_train_nontarget = np.concatenate((X_train_nontarget,dataFrameUserTrain_non.values[:,1:]))
						else:
							X_train_nontarget = np.concatenate((X_train_nontarget, dataFrameUserTrain_non.values[:,2:]))


						labels_nontarget=np.concatenate((labels_nontarget,np.ones(dataFrameUserTrain_non.shape[0])))
			
			X_train=np.concatenate((X_train_target,X_train_nontarget))
			labels=np.concatenate((labels_target,labels_nontarget))

			
			
			scaler_sensor=scaler()
			scaler_sensor.fit(X_train)
			X_train = scaler_sensor.transform(X_train)
			
			## Train Model
			clf = model(**dict_params)
			clf.fit(X_train,labels)
					
			#Test
			userNameTrain=userfile.split('/')[-1].split('_')[0]
			scores_model=np.empty([0,2])
			labels_model=np.empty([0])
			
			scores_unknow=np.empty([0,2])
			labels_unknow=np.empty([0])
			
			for kuser in range(0,len(users)): 
				if iuser==kuser:
					label='target'
					label_num=1
				else:
					label='nontarget'
					label_num=0
					
				filetest = users[kuser].replace('train','test')
				nombreUsuario=filetest.split('/')[-1].split('_')[0]
				numUsuario = int(nombreUsuario.replace('user',''))
				
				
				
				
				dataFrameUserTest=pd.read_csv(filetest,header=None,prefix='X')
			
			
				if not datatype=='voice':
					X_test = dataFrameUserTest.values[:,1:]
				else:
					X_test = dataFrameUserTest.values[:,2:]
			
				X_test=scaler_sensor.transform(X_test)
				Z_clf = clf.predict_proba(X_test)
				
				Y_test = [label for i in range(Z_clf.shape[0])]
				
				if not kuser in user_unknown:

					scores_model=np.concatenate((scores_model,Z_clf))
					labels_model=np.concatenate((labels_model,np.ones(Z_clf.shape[0])*label_num))					
					
				else: 
					scores_unknow = np.concatenate((scores_unknow,Z_clf))
					labels_unknow = np.concatenate((labels_unknow,np.ones(Z_clf.shape[0])*label_num))
		
			scores_global=np.concatenate((scores_global,scores_model))
			labels_global=np.concatenate((labels_global,labels_model))
			
			scores_global_unknow=np.concatenate((scores_global_unknow,scores_unknow))
			labels_global_unknow=np.concatenate((labels_global_unknow,labels_unknow))
					
			eer_sc , auc_sc, f1max, eer_threshold = calculate_metrics(scores_model, labels_model, norm='MinMax')
			eer_users.append(eer_sc)
			auc_users.append(auc_sc)
			f1max_users.append(f1max)
			
			# if args.verbose:
				# print("Metris for known Users") 
				# print(eer_sc)
				# print(auc_sc)
				# print(f1max)
			
			# a_file.write(userfile+','+str(eer_sc) +','+str(auc_sc) + ',' + str(f1max) +"\n")
			
			eer_sc , auc_sc, f1max, eer_threshold = calculate_metrics(np.concatenate((scores_model,scores_unknow)), np.concatenate((labels_model,labels_unknow)), norm='MinMax')
			eer_users.append(eer_sc)
			auc_users.append(auc_sc)
			f1max_users.append(f1max)
			
			# if args.verbose:
				# print("Metris for known Users") 
				# print(eer_sc)
				# print(auc_sc)
				# print(f1max)
		

			# a_file.write(userfile+','+str(eer_sc) +','+str(auc_sc) + ',' + str(f1max) +"\n")
		
	eer_sc , auc_sc, f1max, eer_threshold = calculate_metrics(scores_global, labels_global, norm='MinMax')
	if args.verbose:
		print("Global Metrics for known Users")
		print("EER: " + str(eer_sc))
		print("AUC: " + str(auc_sc))
		print("F1: " + str(f1max))

	a_file.write('global,'+str(eer_sc) +','+str(auc_sc) + ',' + str(f1max) +"\n")

	eer_know.append(eer_sc)
	auc_know.append(auc_sc)
	f1_know.append(f1max)
	
	eer_sc , auc_sc, f1max, eer_threshold = calculate_metrics(np.concatenate((scores_global,scores_global_unknow)), np.concatenate((labels_global,labels_global_unknow)), norm='MinMax')
	if args.verbose:
		print("Global Metrics for unknown Users")
		print("EER: " + str(eer_sc))
		print("AUC: " + str(auc_sc))
		print("F1: " + str(f1max))

	a_file.write('global_unknow,'+str(eer_sc) +','+str(auc_sc) + ',' + str(f1max) +"\n")

	eer_unknow.append(eer_sc)
	auc_unknow.append(auc_sc)
	f1_unknow.append(f1max)
	
print("Confidence interval for know user")
print("EER: ( {} , {} )".format( np.mean(eer_know)-1.96*np.std(eer_know), np.mean(eer_know)+1.96*np.std(eer_know)))
# print("AUC: ( {} , {} )".format( np.mean(auc_know)-1.96*np.std(auc_know), np.mean(auc_know)+1.96*np.std(auc_know)))
# print("F1: ( {} , {} )".format( np.mean(f1_know)-1.96*np.std(f1_know), np.mean(f1_know)+1.96*np.std(f1_know)))


print("Confidence interval for unknow user")
print("EER: ( {} , {} )".format( np.mean(eer_unknow)-1.96*np.std(eer_unknow), np.mean(eer_unknow)+1.96*np.std(eer_unknow)))
# print("AUC: ( {} , {} )".format( np.mean(auc_unknow)-1.96*np.std(auc_unknow), np.mean(auc_unknow)+1.96*np.std(auc_unknow)))
# print("F1: ( {} , {} )".format( np.mean(f1_unknow)-1.96*np.std(f1_unknow), np.mean(f1_unknow)+1.96*np.std(f1_unknow)))

