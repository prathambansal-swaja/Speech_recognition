# common.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import librosa
import IPython.display as ipd
import soundfile as sf
import csv
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------
# Global configuration
# -------------------------------------------------------------------

# Base directory where the train/audio folder lives
BASE_DIR = r"C:/Users/sunde/Downloads/tensorflow-speech-recognition-challenge/train/audio"

# -------------------------------------------------------------------
# Data loading and basic metadata
# -------------------------------------------------------------------

def load_metadata():
    """
    Scan all .wav files under BASE_DIR and return a DataFrame with:
      - file_name (relative path like 'bed/xxx.wav')
      - label     (folder name)
    """
    folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    print("Example folder:", folders[1])

    names = []
    labels = []

    for n in tqdm(folders, desc="Scanning folders"):
        path = os.path.join(BASE_DIR, n)
        file_names = os.listdir(path)
        for f in file_names:
            if f.endswith(".wav"):
                names.append(f"{n}/{f}")
                labels.append(n)

    data = pd.DataFrame({
        "file_name": names,
        "label": labels
    })

    print("Metadata shape:", data.shape)
    print(data.head())
    return data


# -------------------------------------------------------------------
# Plotting utilities
# -------------------------------------------------------------------

def plot_label_counts(data, column_name="label", title="Class"):
    """
    Count-plot of a categorical column (e.g., label).
    Returns the sorted dictionary of counts.
    """
    dic = Counter(data[column_name].values)
    print("data[column_name].values =", data[column_name].values)
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    print(dic)

    column_list = [name for name, _ in dic]

    plt.figure(figsize=(16, 6))
    ax = sns.countplot(x=column_name, data=data, order=column_list, dodge=False)
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.title(f"count plot of {title}")
    plt.show()

    return dic


# -------------------------------------------------------------------
# Duration / clip length utilities
# -------------------------------------------------------------------

def add_clip_lengths(data):
    """
    Compute clip_len (seconds) for each file and add as a column.
    Uses BASE_DIR + file_name to locate each file.
    """
    clip_len = []
    for rel_path in tqdm(data["file_name"].values, desc="Computing durations"):
        full_path = os.path.join(BASE_DIR, rel_path)
        clip_len.append(librosa.get_duration(filename=full_path))

    data["clip_len"] = clip_len
    print("Example last 6 clip lengths:", clip_len[-6:])
    return data


def filter_background_and_short(data, max_len=1.1):
    """
    Remove background_noise and (optionally) keep only clips shorter than max_len.
    Returns filtered DataFrame.
    """
    data_no_noise = data[data["label"] != "_background_noise_"]
    if max_len is not None:
        data_no_noise = data_no_noise[data_no_noise["clip_len"] < max_len]

    print("Original rows:", len(data))
    print("After filtering:", len(data_no_noise))
    return data_no_noise


def plot_clip_len_hist(data_no_noise, bins=30):
    """
    Plot histogram of clip_len for filtered data.
    """
    sns.set_theme()
    plt.figure(figsize=(8, 4))
    sns.histplot(data=data_no_noise, x="clip_len", bins=bins)
    plt.title("PDF of Duration of Clip")
    plt.xlabel("clip_len")
    plt.ylabel("Count")
    plt.show()


def print_key_percentiles(data_no_noise, percentiles=None):
    """
    Print given percentiles of clip_len.
    """
    if percentiles is None:
        percentiles = [1, 3, 5, 7, 9, 10]

    print("\nKey Percentiles:")
    for p in percentiles:
        val = np.percentile(data_no_noise["clip_len"], p)
        print(f"{p}th percentile: {val}")


# -------------------------------------------------------------------
# Signal plotting utilities
# -------------------------------------------------------------------

def plot_signals(signals):
    """
    Plot up to 30 signals in a 6x5 grid.
    Expects signals: dict[label] = 1D numpy array
    """
    fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(20, 15))
    i = 0
    for x in range(6):
        for y in range(5):
            if i >= len(signals):
                ax[x, y].axis("off")
                continue
            ax[x, y].set_title(list(signals.keys())[i])
            ax[x, y].plot(list(signals.values())[i])
            ax[x, y].get_xaxis().set_visible(False)
            ax[x, y].get_yaxis().set_visible(False)
            i += 1
    plt.tight_layout()


def collect_example_signals(data):
    """
    For each label (except background noise), load one example signal.
    Returns dict[label] = signal_array
    """
    signals = {}
    labels = np.unique(data["label"])

    # skip _background_noise_ if present
    for name in labels:
        if name == "_background_noise_":
            continue
        file_row = data[data["label"] == name][:1]
        rel_path = file_row["file_name"].values[0]
        full_path = os.path.join(BASE_DIR, rel_path)
        signal, rate = librosa.load(full_path, sr=None)
        signals[name] = signal

    return signals


def collect_noise_signals(data):
    """
    Collect all noise signals (label == '_background_noise_').
    Returns:
      - noise_signals: dict[title] = signal array
      - titles: list of titles
    """
    noise_rows = data[data["label"] == "_background_noise_"]
    noise_signals = {}
    titles = []

    for rel_path in noise_rows["file_name"].values:
        full_path = os.path.join(BASE_DIR, rel_path)
        title = rel_path.split("/")[1].split(".")[0]
        signal, rate = librosa.load(full_path, sr=None)
        noise_signals[rel_path] = signal
        titles.append(title)

    return noise_signals, titles


def plot_noise_signals(noise_signals, titles):
    """
    Plot noise signals in a 2x3 grid.
    """
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(22, 8))
    i = 0
    for x in range(2):
        for y in range(3):
            if i >= len(noise_signals):
                ax[x, y].axis("off")
                continue
            ax[x, y].set_title(titles[i])
            ax[x, y].plot(list(noise_signals.values())[i])
            if (x == 0 and y == 0) or (x == 1 and y == 0):
                ax[x, y].set_ylabel("Amplitude")
            if x == 1:
                ax[x, y].set_xlabel("Time (Samples)")
            i += 1
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Resampling demo utility
# -------------------------------------------------------------------

def demo_resample(example_rel_path="eight/bbb2eb5b_nohash_0.wav",
                  orig_sr=16000,
                  target_sr=8000,
                  out_filename="output_8khz.wav"):
    """
    Load one example file, plot original waveform, resample to target_sr,
    save to WAV, and plot resampled waveform.
    """
    full_path = os.path.join(BASE_DIR, example_rel_path)
    samples, sample_rate = librosa.load(full_path, sr=orig_sr)
    print("Original sample_rate:", sample_rate)

    # Original waveform
    ipd.display(ipd.Audio(samples, rate=sample_rate))
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(211)
    ax.set_title(f"Raw wave of {example_rel_path}   [sample_rate = {sample_rate}]")
    ax.set_xlabel("Time(second)")
    ax.set_ylabel("Amplitude")
    ax.plot(np.linspace(0, len(samples) / sample_rate, len(samples)), samples)
    plt.show()

    # Resample
    samples_resampled = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
    sf.write(out_filename, samples_resampled, target_sr, subtype="PCM_16")
    print(f"Successfully saved resampled audio to {out_filename}")

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(211)
    ax.set_title(f"Raw wave of {example_rel_path}   [sample_rate = {target_sr}]")
    ax.set_xlabel("Time(second)")
    ax.set_ylabel("Amplitude")
    ax.plot(np.linspace(0, len(samples_resampled) / target_sr, len(samples_resampled)), samples_resampled)
    plt.show()

    return out_filename

def pad_or_truncate(data, target_len=16000, column="raw_data", save_csv=False, csv_name="data_final.csv"):
    """
    Pad or truncate audio signals so they all have the same fixed length.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Must contain a column with 1-D numpy arrays (raw audio)
    target_len : int
        Desired fixed length in samples (e.g., 16000 for 1 second at 16kHz)
    column : str
        Name of the column containing raw audio arrays

    Returns
    -------
    numpy.ndarray
        Array of shape (num_samples, target_len)
    """
    padded = []
    for signal in tqdm(data[column], desc="Padding/Truncating"):
        sig = np.array(signal)

        if len(sig) < target_len:
            # Pad zeros at end
            sig = np.pad(sig, (0, target_len - len(sig)), mode='constant')
        else:
            # Truncate
            sig = sig[:target_len]

        padded.append(sig)
        
    # add padded sequences as a new column
    data["pad_seq"] = padded

    # optionally save
    if save_csv:
        data.to_csv(csv_name, index=False)
        print(f"Saved CSV: {csv_name}")

    return data, np.array(padded)

    #return np.array(padded)
