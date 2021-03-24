import torch
import random
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn import preprocessing
import soundfile as sf
import math


class SeparationDataset(Dataset):
    def __init__(self, sources_path, normals_ids, abnormals_ids, p=0.3, sr=16000, l=4, n=1600, status="train"):
        self.p = p
        self.sr = sr
        self.samples = int(l * sr)
        self.scaler = preprocessing.MaxAbsScaler()
        self.normal_paths = [i + '/normal/' for i in sources_path]
        self.abnormal_paths = [i + '/abnormal/' for i in sources_path]
        self.normals_ids = normals_ids
        self.abnormals_ids = abnormals_ids
        self.n = n
        self.status = status

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        n_spk = len(self.normals_ids)
        targets = np.zeros([n_spk, self.samples], dtype=np.float32)
        mix = np.zeros(self.samples, dtype=np.float32)
        labels = np.ones(n_spk, dtype=np.float32)
        for i in range(n_spk):
            p = random.random()
            n_normal = len(self.normals_ids[i])
            n_abnormal = len(self.abnormals_ids[i])
            if p > self.p:
                path = self.abnormal_paths[i] + self.abnormals_ids[i][random.randint(0, n_abnormal - 1)]
                labels[i] = 0
            else:
                path = self.normal_paths[i] + self.normals_ids[i][random.randint(0, n_normal - 1)]
            data = self.load_audio(path)
            mix += data
            data = self.scaler.fit_transform(data.reshape(-1, 1)).T
            targets[i] = data[0, :]

        mix = self.scaler.fit_transform(mix.reshape(-1, 1)).T
        if self.status == "train":
            return torch.tensor(mix), torch.tensor(targets)
        else:
            return torch.tensor(mix), torch.tensor(targets), torch.tensor(labels)

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        target_length = audio.shape[0]
        start_pos = np.random.randint(0, max(target_length - self.samples + 1, 1))
        end_pos = start_pos + self.samples
        return audio[start_pos:end_pos]
