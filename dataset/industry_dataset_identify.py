import torch
import random
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn import preprocessing
import soundfile as sf
import math


class BaseDataset(Dataset):
    def __init__(self, sources_path, normals_ids, p=0.5, sr=16000, l=4, n=1600):
        self.p = p
        self.sr = sr
        self.samples = int(l * sr)
        self.scaler = preprocessing.MaxAbsScaler()
        self.normal_paths = [i + '{}/normal/' for i in sources_path]
        self.normals_ids = normals_ids
        self.n = n
        self.ids = ["id_00", "id_02", "id_04", "id_06"]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        raise NotImplementedError

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        target_length = audio.shape[0]
        start_pos = np.random.randint(0, max(target_length - self.samples + 1, 1))
        end_pos = start_pos + self.samples
        return audio[start_pos:end_pos]


class IndustryDatasetIdentify(BaseDataset):
    '''
    s is always same and s' maybe different
    '''
    def __init__(self, sources_path, normals_ids, p=0.5, sr=16000, l=4, n=3200):
        super().__init__(sources_path, normals_ids, p, sr, l, n)

    def __getitem__(self, idx):
        n_sources = len(self.normal_paths)
        status = np.ones(n_sources, dtype=np.float32)
        mix = np.zeros(self.samples, dtype=np.float32)
        mix_ = np.zeros(self.samples, dtype=np.float32)
        for i in range(n_sources):
            id_ = random.randint(1, 3)
            n_normal = len(self.normals_ids[i][0])
            path = self.normal_paths[i] + self.normals_ids[i][0][random.randint(0, n_normal - 1)]
            path = path.format(self.ids[0])
            p = random.random()
            if p > self.p:
                n_normal_ = len(self.normals_ids[i][id_])
                path_ = self.normal_paths[i] + self.normals_ids[i][id_][random.randint(0, n_normal_ - 1)]
                path_ = path_.format(self.ids[id_])
                status[i] = 0
            else:
                path_ = (self.normal_paths[i] + self.normals_ids[i][0][random.randint(0, n_normal - 1)])
                path_ = path_.format(self.ids[0])
            data = self.load_audio(path)
            data_ = self.load_audio(path_)
            mix += data
            mix_ += data_
        mix = self.scaler.fit_transform(mix.reshape(-1, 1)).T
        mix_ = self.scaler.fit_transform(mix_.reshape(-1, 1)).T

        return torch.tensor(mix), torch.tensor(mix_), torch.tensor(status)


class IndustryDatasetIdentifyDiff(BaseDataset):
    '''
    s and s' maybe different
    '''
    def __init__(self, sources_path, normals_ids, p=0.5, sr=16000, l=4, n=3200):
        super().__init__(sources_path, normals_ids, p, sr, l, n)

    def __getitem__(self, idx):
        n_sources = len(self.normal_paths)
        status = np.ones(n_sources, dtype=np.float32)
        mix = np.zeros(self.samples, dtype=np.float32)
        mix_ = np.zeros(self.samples, dtype=np.float32)
        for i in range(n_sources):
            id = random.randint(0, 3)
            id_ = random.randint(0, 3)
            n_normal = len(self.normals_ids[i][id])
            n_normal_ = len(self.normals_ids[i][id_])
            path = self.normal_paths[i] + self.normals_ids[i][id][random.randint(0, n_normal - 1)]
            path = path.format(self.ids[id])
            if id != id_:
                path_ = self.normal_paths[i] + self.normals_ids[i][id_][random.randint(0, n_normal_ - 1)]
                path_ = path_.format(self.ids[id_])
                status[i] = 0
            else:
                path_ = (self.normal_paths[i] + self.normals_ids[i][id][random.randint(0, n_normal - 1)])
                path_ = path_.format(self.ids[id])
            data = self.load_audio(path)
            data_ = self.load_audio(path_)
            mix += data
            mix_ += data_
        mix = self.scaler.fit_transform(mix.reshape(-1, 1)).T
        mix_ = self.scaler.fit_transform(mix_.reshape(-1, 1)).T

        return torch.tensor(mix), torch.tensor(mix_), torch.tensor(status)


if __name__ == '__main__':
    print(random.randint(0,3))
