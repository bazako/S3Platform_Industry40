# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from progressbar import progressbar

def getData(fileSesion):
    sensor=[]
    statistics=[]
    voiceNotes=[]
    CallRecordings=[]    
    fid=open(fileSesion,'r', encoding = "ISO-8859-1") 
    
    lines=fid.readlines()
    for line in lines:
        #line = fid.readline()
        line=line.replace('\n','')
        #print(line)
        if not line:
            break
        
        lineSplit=line.split(',')
        numfields=len(lineSplit)
        second_field=lineSplit[1]
        
        
        if numfields>512:
            if second_field=='vn':
                # Voicenote 
                # voiceNotes.append(map(str.strip, line.split(',')))
                voiceNotes.append(line.split(','))
            else:
                # Call recorded
                CallRecordings.append(line.split(','))
            
        elif numfields>20:
            # App' statistic
            #data=map(str.strip, line.split(','))
            sensor.append(line.split(','))

        else:
            #data=map(str.strip, line.split(','))
            statistics.append(line.split(','))
        
        
        
        
    fid.close()
    
    sensor = np.array(sensor)
    statistics = np.array(statistics)
    voiceNotes = np.array(voiceNotes)
    CallRecordings = np.array(CallRecordings)
    
    dias=['lun.','mar.','mié.','jue.','vie.','sáb.','dom.']
    days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for idia in range(0,7):
        dia=dias[idia]
        day=days[idia]
        if sensor.shape[0]>0:
            sensor[sensor==dia]=str(idia)
            sensor[sensor==day]=str(idia)
        if statistics.shape[0]>0:
            statistics[statistics==dia]=str(idia)
            statistics[statistics==day]=str(idia)
    
    return sensor, statistics, voiceNotes, CallRecordings


infoData = pd.read_csv('./S3Dataset/info/S3dataset.txt',
                       sep=' ', names=['file','user','timestampIni',
                                       'timestampFin','dur','type'])


protocolTrain = pd.read_csv('./S3Dataset/protocols/14days-vs-all.trn',
                       sep=' ', names=['model','files'])

pathToTrain='./data/14days/train'
pathToTest='./data/14days/test'
os.makedirs(pathToTrain,exist_ok=True)
os.makedirs(pathToTest,exist_ok=True)

protocolTrain='./S3Dataset/protocols/14days-vs-all.trn'
fid_pro_train = open(protocolTrain,'r')
lines=fid_pro_train.readlines()
fid_pro_train.close()

filesUsedToTrain=[]
for iline in range(0, len(lines)):
    # Get next line from file 
    line = lines[iline] 
    # print(line)
    
    # if line is empty 
    # end of file is reached 
    if not line: 
        break
    
    line=line.replace('\n','').replace('data/','')
    line_split=line.split(' ')
    model_name=line_split[0]
    model_dur=line_split[1]
    model_tistart=line_split[2]
    model_tiend=line_split[3]
    model_tipe=line_split[4]
    model_files=line_split[5:]
    
    filesUsedToTrain.append(model_files)
filesUsedToTrain = [item for sublist in filesUsedToTrain for item in sublist]    

for ifile in progressbar(range(len(infoData))):
    fileToMove=infoData.file.iloc[ifile]
    dirToFile='./S3Dataset/data/'+fileToMove
    userFile=infoData.user.iloc[ifile]#.replace('user','model')
    
    
    sensors, statistics, voicenotes, callrecordings = getData(dirToFile)
    
    if fileToMove in filesUsedToTrain:
        # Write in the train files
        
        
        if sensors.shape[0]>0:
            file_write_train_sensors = pathToTrain+'/'+userFile+'_sensors.txt'
            fid_write_train_sensors = open(file_write_train_sensors,'a')
            linesToWrite=[','.join(map(str,line)) for line in sensors]
            for line in linesToWrite:
                fid_write_train_sensors.write(line+'\n')
            fid_write_train_sensors.close()
        
        if statistics.shape[0]>0:
            file_write_train_statistics = pathToTrain+'/'+userFile+'_statistics.txt'
            fid_write_train_statistics = open(file_write_train_statistics,'a')
            linesToWrite=[','.join(map(str,line)) for line in statistics]
            for line in linesToWrite:
                fid_write_train_statistics.write(line+'\n')
            fid_write_train_statistics.close()
            
            
        if voicenotes.shape[0]>0:
            file_write_train_voicenotes = pathToTrain+'/'+userFile+'_voicenotes.txt'
            fid_write_train_voicenotes = open(file_write_train_voicenotes,'a')
            linesToWrite=[','.join(map(str,line)) for line in voicenotes]
            for line in linesToWrite:
                fid_write_train_voicenotes.write(line+'\n')
            fid_write_train_voicenotes.close()
            
        if callrecordings.shape[0]>0:
            file_write_train_callrecordings = pathToTrain+'/'+userFile+'_callrecordings.txt'
            fid_write_train_callrecordings = open(file_write_train_callrecordings,'a')
            linesToWrite=[','.join(map(str,line)) for line in callrecordings]
            for line in linesToWrite:
                fid_write_train_callrecordings.write(line+'\n')
            fid_write_train_callrecordings.close()
			
        if voicenotes.shape[0]>0 or callrecordings.shape[0]>0:
            file_write_train_voice = pathToTrain+'/'+userFile+'_voice.txt'
            fid_write_train_voice = open(file_write_train_voice,'a')
		
            if voicenotes.shape[0]>0:
                linesToWrite=[','.join(map(str,line)) for line in voicenotes]
                for line in linesToWrite:
                    fid_write_train_voice.write(line+'\n')
		
            if callrecordings.shape[0]>0:
                linesToWrite=[','.join(map(str,line)) for line in callrecordings]
                for line in linesToWrite:
                    fid_write_train_voice.write(line+'\n')
            fid_write_train_voice.close()
            
        
    else:
        
        userTest=userFile
        if sensors.shape[0]>0:
            file_write_train_sensors = pathToTest+'/'+userTest+'_sensors.txt'
            fid_write_train_sensors = open(file_write_train_sensors,'a')
            linesToWrite=[','.join(map(str,line)) for line in sensors]
            for line in linesToWrite:
                fid_write_train_sensors.write(line+'\n')
#                fid_write_train_sensors.write(line+','+label+'\n')
            fid_write_train_sensors.close()
        
        if statistics.shape[0]>0:
            file_write_train_statistics = pathToTest+'/'+userTest+'_statistics.txt'
            fid_write_train_statistics = open(file_write_train_statistics,'a')
            linesToWrite=[','.join(map(str,line)) for line in statistics]
            for line in linesToWrite:
#                fid_write_train_statistics.write(line+','+label+'\n')
                fid_write_train_statistics.write(line+'\n')
            fid_write_train_statistics.close()
            
            
        if voicenotes.shape[0]>0:
            file_write_train_voicenotes = pathToTest+'/'+userTest+'_voicenotes.txt'
            fid_write_train_voicenotes = open(file_write_train_voicenotes,'a')
            linesToWrite=[','.join(map(str,line)) for line in voicenotes]
            for line in linesToWrite:
                fid_write_train_voicenotes.write(line+'\n')
#                fid_write_train_voicenotes.write(line+','+label+'\n')
            fid_write_train_voicenotes.close()
            
        if callrecordings.shape[0]>0:
            file_write_train_callrecordings = pathToTest+'/'+userTest+'_callrecordings.txt'
            fid_write_train_callrecordings = open(file_write_train_callrecordings,'a')
            linesToWrite=[','.join(map(str,line)) for line in callrecordings]
            for line in linesToWrite:
#                fid_write_train_callrecordings.write(line+','+label+'\n')
                fid_write_train_callrecordings.write(line+'\n')
            fid_write_train_callrecordings.close()
	
        if voicenotes.shape[0]>0 or callrecordings.shape[0]>0:
            file_write_train_voice = pathToTest+'/'+userTest+'_voice.txt'
            fid_write_train_voice = open(file_write_train_voice,'a')
		
            if voicenotes.shape[0]>0:
                linesToWrite=[','.join(map(str,line)) for line in voicenotes]
                for line in linesToWrite:
                    fid_write_train_voice.write(line+'\n')
		
            if callrecordings.shape[0]>0:
                linesToWrite=[','.join(map(str,line)) for line in callrecordings]
                for line in linesToWrite:
                    fid_write_train_voice.write(line+'\n')
            fid_write_train_voice.close()

