from common import load_metadata, add_clip_lengths, BASE_DIR ,pad_or_truncate
from tqdm import tqdm
import librosa
import os
import pandas as pd
import numpy as np
import csv
# Load metadata and clip durations
data = load_metadata()
data = add_clip_lengths(data)

# Load raw waveforms
raw_data = []
for rel_path in tqdm(data["file_name"]):
    full_path = os.path.join(BASE_DIR, rel_path)   # Construct full path
    signal, rate = librosa.load(full_path, sr=16000)
    raw_data.append(signal)

data["raw_data"] = raw_data
print(data.head())
''' padding - make all file same length(16000) '''
# if len of audio is < 16000 we add zeros.
# if len of audio is > 16000 we truncate.

#pad_seq = pad_or_truncate(data, target_len=16000, column="raw_data")
data, pad_seq = pad_or_truncate(data, target_len=16000, column="raw_data", save_csv=True, csv_name="data_final.csv")

print("Padded shape:", pad_seq.shape)
print(data.head())