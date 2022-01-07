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


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('datatype', type=str, default='sensors', help='sensors, statistics or voice')
parser.add_argument('model', type=str, default='KNN', help='KNN, RF, SV')

args = parser.parse_args()
print(args)
datatype=args.datatype 
modelName=args.model

pathToFiles='data/14days/'

if modelName == 'SV':
	model = OCSVM
	if datatype=='sensors':
		dict_params =  {'kernel': 'rbf', 'nu': 0.1}
	elif datatype=='statistics':
		dict_params = {'kernel': 'rbf', 'nu': 0.1}
	else:
		dict_params = {'kernel': 'rbf', 'nu': 0.1}
	
	

# KNN
if modelName == 'KNN':
	model = KNN
	if datatype=='sensors':
		dict_params = {'n_neighbors': 1}
	elif datatype=='statistics':
		dict_params = {'n_neighbors': 1}
	else:
		dict_params = {'n_neighbors': 10}

# RF
if modelName == 'RF':
	model = IForest
	if datatype=='sensors':
		dict_params =  {'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 100}
	elif datatype=='statistics':
		dict_params = {'max_features': 0.2, 'max_samples': 1.0, 'n_estimators': 100}
	else:
		dict_params = {'max_features': 1.0, 'max_samples': 'auto', 'n_estimators': 100}

scaler = StandardScaler
	

nameModel=modelName #str(model).split('.')[-1].replace("'","").replace('>','')

pathToScores='./scores/scores_exp1.2_UnSupervised/'+nameModel+'/'
os.makedirs(pathToScores,exist_ok=True)



datatype=args.datatype
users=glob.glob(pathToFiles+'train/*'+datatype+'.txt')
users.sort()


eer_users=[]
auc_users=[]
f1max_users=[]

scores_global=np.empty([0,2])
labels_global=np.empty([0])
model_global=np.empty([0])
test_global=np.empty([0])

file_resume = pathToScores + '/resume_'+datatype+'.txt'
a_file=open(file_resume, "a")

for iuser in progressbar(range(0,len(users))):	
		
		
	userfile = users[iuser]
	# print(" USER {}-{}: {}".format(iuser, len(users),userfile))
	
	dataFrameUserTrain=pd.read_csv(userfile,header=None,prefix='X')
	
	if not datatype=='voice':
		X_train = dataFrameUserTrain.values[:,1:]
	else:
		X_train = dataFrameUserTrain.values[:,2:]
		
	scaler_sensor=scaler()
	scaler_sensor.fit(X_train)
	X_train = scaler_sensor.transform(X_train)
	
	## Train Model
	clf = model(**dict_params)
	clf.fit(X_train)
			
	#Test
	userNameTrain=userfile.split('/')[-1].split('_')[0]
	scores_model=np.empty([0,2])
	labels_model=np.empty([0])
	for kuser in range(0,len(users)): 
		if iuser==kuser:
			label='target'
			label_num=1
		else:
			label='nontarget'
			label_num=0
			
		filetest = users[kuser].replace('train','test')
		dataFrameUserTest=pd.read_csv(filetest,header=None,prefix='X')
	
	
		if not datatype=='voice':
			X_test = dataFrameUserTest.values[:,1:]
		else:
			X_test = dataFrameUserTest.values[:,2:]
	
		X_test=scaler_sensor.transform(X_test)
		Z_clf = clf.predict_proba(X_test)
		
		Y_test = [label for i in range(Z_clf.shape[0])]

		scores_model=np.concatenate((scores_model,Z_clf))
		labels_model=np.concatenate((labels_model,np.ones(Z_clf.shape[0])*label_num))
		model_global=np.concatenate((model_global,np.ones(Z_clf.shape[0])*iuser))
		test_global=np.concatenate((test_global,np.ones(Z_clf.shape[0])*kuser))
	scores_global=np.concatenate((scores_global,scores_model))
	labels_global=np.concatenate((labels_global,labels_model))
		
			
	eer_sc , auc_sc, f1max, eer_threshold = calculate_metrics(scores_model, labels_model, norm='MinMax')
	eer_users.append(eer_sc)
	auc_users.append(auc_sc)
	f1max_users.append(f1max)
	
	# print(eer_sc)
	# print(auc_sc)
	# print(f1max)
	

	a_file.write(userfile+','+str(eer_sc) +','+str(auc_sc) + ',' + str(f1max) +"\n")
	
eer_sc , auc_sc, f1max, eer_threshold = calculate_metrics(scores_global, labels_global, norm='MinMax')
print("EER: " + str(eer_sc))
print("AUC: " + str(auc_sc))
print("F1: " + str(f1max))

a_file.write('global,'+str(eer_sc) +','+str(auc_sc) + ',' + str(f1max) +"\n")

from sklearn.metrics import roc_curve

fpr_global, tpr_global, threshold_global = roc_curve(labels_global,scores_global[:,0], pos_label=1)
fnr_global = 1 - tpr_global


file_resume = pathToScores + '/FPR_'+datatype+'.csv'
np.savetxt(file_resume, fpr_global, delimiter=",")

file_resume = pathToScores + '/FNR_'+datatype+'.csv'
np.savetxt(file_resume, fnr_global, delimiter=",")

file_resume = pathToScores + '/thr_'+datatype+'.csv'
np.savetxt(file_resume, threshold_global, delimiter=",")


