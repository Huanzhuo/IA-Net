import torch
import random
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn import preprocessing
import soundfile as sf
import math
import sys
import os
import glob


def file_to_array(file_name, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=2.0):
    dims = n_mels * frames
    y, sr = librosa.load(file_name, sr=16000, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
    if vector_array_size < 1:
        return np.empty((0, dims))

    vector_array = np.zeros((vector_array_size, dims), dtype=np.float32)
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array


def mel(data, n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0):
    dims = n_mels * frames
    mel_spectrogram = librosa.feature.melspectrogram(data, sr=16000, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                                                     power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
    if vector_array_size < 1:
        return np.empty((0, dims))

    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T
    return vector_array


class AutoDataset(Dataset):
    def __init__(self, source_path, files_list):
        self.path = source_path
        self.files_list = files_list

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        path = self.files_list[idx]
        data = file_to_array(path)
        return torch.tensor(data)


if __name__ == '__main__':
    path = '../../MIMII/valve/id_00/normal/'
    dataset = AutoDataset(path)
    for i in range(len(dataset)):
        dataset[i]



