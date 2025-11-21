import os 
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
print(f"Current working directory: {os.getcwd()}")
os.chdir(r"train/audio")
print(f"Current working directory: {os.getcwd()}")
folders = [f for f in os.listdir() if os.path.isdir(f)]
print(folders[1])
names=[]
labels=[]
for n in tqdm(folders):
    path = os.path.join(r"C:/Users/sunde/Downloads/tensorflow-speech-recognition-challenge/train/audio",n)
    file_names=os.listdir(path)
    count=0
    for i in file_names:
        if i.endswith('.wav'):
            names.append(n+'/'+i)
            labels.append(n)
print(names[3000])
print(labels[5])
# create dataframe

with open('data1.csv' , 'w') as csvfile:
    csvwriter = csv.writer(csvfile) 

    header = ('file_name','label')
    csvwriter.writerow(header)
    
data = pd.read_csv('data1.csv')
data['file_name']=names
data['label']=labels
print(data.shape)
print(data.head)