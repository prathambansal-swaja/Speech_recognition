import os 
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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

# Build dataframe correctly
data = pd.DataFrame({
    'file_name': names,
    'label': labels
})

print(data.shape)   # should be (64727, 2)
print(data.head())

def plot(column_name , title):
    #get unique_values of each categories
    dic = Counter(data[column_name].values)
    #sort in deccending order by values
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    column_list = []


    #get name of sorted dic
    for name in dic:
        column_list.append(name[0])

    plt.figure(figsize = (16,6))
    ax = sns.countplot(x = column_name ,  data = data , order = column_list, dodge = False)

    h,l = ax.get_legend_handles_labels()
    ax.legend(h ,column_list,bbox_to_anchor=(1.05, 1) ,loc = 'upper left')
    plt.setp(ax.get_xticklabels() , rotation = 90 )
    plt.title('count plot of {}'.format(title))
    plt.show()
    return dic
dic = plot('label' , 'Class')
print(dict(Counter(data['label'])))