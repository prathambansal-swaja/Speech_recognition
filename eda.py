import os 
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import librosa
import IPython.display as ipd
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
    print("data[column_name].values=",data[column_name].values)
    #sort in deccending order by values
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    print(dic)
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
#Find duration of all files
clip_len=[]
for file in tqdm(data['file_name'].values):
    clip_len.append(librosa.get_duration(filename=file))
data['clip_len'] = clip_len
#data.to_csv('data_final.csv')
print(clip_len[-6:])
# not calculate last 6 because last clips are background noise
# length of backgroung noise are higher than 1 sec so they 6 points corrupt histogram
# After you compute data['clip_len']
# Remove background noise explicitly
data_no_noise = data[data['label'] != '_background_noise_']

# (optional) also limit to clips shorter than 1.1 s
data_no_noise = data_no_noise[data_no_noise['clip_len'] < 1.1]

print("Original rows:", len(data))
print("After filtering:", len(data_no_noise))

sns.set_theme()
plt.figure(figsize=(8, 4))
sns.histplot(data=data_no_noise, x='clip_len', bins=30)
plt.title('PDF of Duration of Clip')
plt.xlabel('clip_len')
plt.ylabel('Count')
plt.show()

# Important percentiles
print("\nKey Percentiles:")
for p in [1,3,5,7,9,10]:
    print(f"{p}th percentile: {np.percentile(data_no_noise['clip_len'], p)}")
#As on 10th percentile we got the value of 10th perecentile as 1 
# that means 10% of data has audio duration less than 1 second
def plot_signals(signals):
    fig , ax = plt.subplots(nrows = 6 , ncols = 5 , figsize = (20,15))
    i = 0
    for x in range(6):
        for y in range(5):
            ax[x,y].set_title(list(signals.keys())[i])
            ax[x,y].plot(list(signals.values())[i])
            ax[x,y].get_xaxis().set_visible(False)
            ax[x,y].get_yaxis().set_visible(False)
            i += 1
# store all signal in dic
signals = {}
labels = np.unique(data['label'])

# get all signal array except backgroud noise
for name in labels[1:]:
    file = data[data['label'] == name ][:1]

    signal , rate = librosa.load(file['file_name'].values[0])

    signals[name] = signal
plot_signals(signals)
plt.show()
''' noise signals '''

noise = data[-6:]
noise_signals = {}
title = []
for name in noise['file_name'].values:
    title.append(name.split('/')[1].split('.')[0])

    signal , rate = librosa.load(name)

    noise_signals[name] = signal

''' plot noise signals '''

fig , ax = plt.subplots(nrows = 2 , ncols = 3 , figsize = (22,8))
i = 0
for x in range(2):
    for y in range(3):
        ax[x,y].set_title(list(title)[i])
        ax[x,y].plot(list(noise_signals.values())[i])
        if x == 0 and y == 0 or x == 1 and y == 0:
            ax[x,y].set_ylabel('Amplitude')
        if x == 1:
                ax[x,y].set_xlabel('Time (Seconds)')            
        i += 1

plt.show()
# Observation = Frome above : All noise files are above 1 second
# try with different sample_rate for dimensionality reduction
samples , sample_rate = librosa.load('eight/bbb2eb5b_nohash_0.wav' , sr=16000)
print(sample_rate)

ipd.Audio(samples, rate=sample_rate)
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(211)

ax.set_title('Raw wave of '+ 'eight/bbb2eb5b_nohash_0.wav'+ '   [sample_rate = 16000]')
ax.set_xlabel('Time(second)')
ax.set_ylabel('Amplitude')
ax.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)
plt.show()
# reduce sample_rate to 8000
# it reduce dimensonality 
# we reduce the sample_rate but it is not
#samples = librosa.resample(samples,sample_rate , 8000)
samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=8000)
ipd.Audio(samples , rate = 8000)
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(211)

ax.set_title('Raw wave of '+ 'eight/bbb2eb5b_nohash_0.wav' + '   [sample_rate = 8000]')
ax.set_xlabel('Time(second)')
ax.set_ylabel('Amplitude')
ax.plot(np.linspace(0, len(samples)/sample_rate, len(samples)), samples)
plt.show()
#Conclusion:
# Dataset is little imbalanced.First we try we as it is than we can try with down sampling.
# 10 % data point's duration or length are < 1. For make all length equal we use zero padding.
# for dimensonality reduction we can use 8000 hz.
# for argumentation we use noise data and other argumentation techniques.
