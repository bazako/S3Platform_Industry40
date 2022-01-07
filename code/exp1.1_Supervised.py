#!bin/bash

import glob
import pandas as pd
import os 
import numpy as np
import json
from progressbar import progressbar
from utils import calculate_eer

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

args = parser.parse_args()
print(args)
datatype=args.datatype 
modelName=args.model


pathToFiles='data/14days/'

numRepeats = 1
numSplits = 5

if datatype=='voice':
	numRepeats = 10
	numSplits = 3

if modelName == 'SV':
	parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'C':[1,5,10,20,50]}
	model = SVC()

# KNN
if modelName == 'KNN':
	parameters = {'weights':('uniform', 'distance'), 'n_neighbors':[1,5,10,50,100], 'algorithm':('auto','ball_tree', 'kd_tree'), 'leaf_size':[10,30,50,100]}
	model = KNeighborsClassifier()

# RF
if modelName == 'RF':
	parameters = {'criterion':('gini', 'entropy'), 'n_estimators':[1,5,10,50,100], 'max_depth':[10,30,50,100, None], 'min_samples_split':[2,5,10]}
	model = RandomForestClassifier()



 
scaler = StandardScaler


scoring = {'EER':make_scorer(calculate_eer,greater_is_better=False),'AUC': 'roc_auc', 'F1': 'f1', 'Accuracy': make_scorer(accuracy_score)}


users=glob.glob(pathToFiles+'train/*'+datatype+'.txt')
users.sort()


nameModel=modelName #str(model).split('.')[-1].replace("'","").replace('>','')

pathToScores='./scores/scores_exp1.1_Supervised/'+nameModel + '/'
os.makedirs(pathToScores,exist_ok=True)

	
for iuser in progressbar(range(0,len(users))):

	userfile = users[iuser]
	
	dataFrameUserTrain=pd.read_csv(userfile,header=None,prefix='X')
	userfile=userfile.split('/')[-1].split('_')[0]
	# print(" USER {}-{}: {}".format(iuser, len(users),userfile))
	
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

			# contUser+=1
			userfile_non = users[juser]
			dataFrameUserTrain_non=pd.read_csv(userfile_non,header=None,prefix='X')

			if not datatype=='voice':
				X_train_nontarget = np.concatenate((X_train_nontarget,dataFrameUserTrain_non.values[:,1:]))
			else:
				X_train_nontarget = np.concatenate((X_train_nontarget, dataFrameUserTrain_non.values[:,2:]))


			labels_nontarget=np.concatenate((labels_nontarget,np.ones(dataFrameUserTrain_non.shape[0])))
	
	index = np.random.choice(np.arange(X_train_nontarget.shape[0]), X_train_target.shape[0], replace=False)
	
	if datatype=='sensors' or datatype=='statistics':
		X_train=np.concatenate((X_train_target,X_train_nontarget[index,:]))
		labels=np.concatenate((labels_target,labels_nontarget[index]))
	
	else:
		X_train=np.concatenate((X_train_target,X_train_nontarget))
		labels=np.concatenate((labels_target,labels_nontarget))

	
	skf = RepeatedStratifiedKFold(n_splits=numSplits, n_repeats=numRepeats, random_state=1)
	scaler_sensor=scaler()
	scaler_sensor.fit(X_train)
	X_train = scaler_sensor.transform(X_train)
	
	gs = GridSearchCV(model,parameters,scoring=scoring,n_jobs=-1,refit='EER', cv=skf, verbose=0)
	gs.fit(X_train,labels)
	
	
	pd_results = pd.DataFrame.from_dict(gs.cv_results_)
	pd_results['Model']=userfile
	
	if iuser==0:
		pd_all = pd_results
	else:
		pd_all = pd.concat([pd_all,pd_results],ignore_index=True, sort=False)
	
	
print(pd_all)
pd_all.to_csv(path_or_buf=pathToScores+datatype+".csv", sep=',')


## Resume 
gs.get_params()

pd_all['params'] = pd_all['params'].astype(str)

list_params = pd_all['params'].unique()

list_columns = pd_all.columns


resume=[]
for params in list_params:
	resume_params = [params]
	print(params)
    
	for key in scoring:
		columns_filter = [x for x in list_columns if key in x and 'split' in x]
       
		pd_filter = pd_all[pd_all['params']==params]
		scores = pd_filter[columns_filter].values.flatten()
       
		resume_params.append(np.mean(scores))
		resume_params.append(np.std(scores))
    
    
	resume_params.append(scores.shape[0])
	resume.append(resume_params)


columns_name=['params']
for key in scoring:
    columns_name.append('mu'+key)
    columns_name.append('std'+key)
columns_name.append('numTest')

resume_pd = pd.DataFrame(resume,columns=columns_name)
resume_pd.to_csv(path_or_buf=pathToScores+"resume_"+datatype+".csv", sep=',')
