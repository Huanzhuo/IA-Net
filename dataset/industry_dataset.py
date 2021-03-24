import torch
import random
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn import preprocessing
import soundfile as sf
import math


class IndustryDataset(Dataset):
    def __init__(self, source_1_path, source_2_path, s1_normal, s1_abnormal, s2_normal, s2_abnormal, p=0.5, sr=16000, l=2):
        self.p = p
        self.sr = sr
        self.samples = int(l * sr)
        self.scaler = preprocessing.MaxAbsScaler()
        self.s1_normal_path = source_1_path + '/normal/'
        self.s2_normal_path = source_2_path + '/normal/'
        self.s1_abnormal_path = source_1_path + '/abnormal/'
        self.s2_abnormal_path = source_2_path + '/abnormal/'

        self.s1_normal = s1_normal
        self.s1_abnormal = s1_abnormal
        self.s2_normal = s2_normal
        self.s2_abnormal = s2_abnormal

    def __len__(self):
        n = len(self.s1_normal) + len(self.s2_normal)
        return n

    def __getitem__(self, idx):
        if idx < len(self.s1_normal):
            # pick normal data
            s1_path = self.s1_normal_path + self.s1_normal[idx]
            s2_path = self.s2_normal_path + self.s2_normal[random.randint(0, len(self.s2_normal)-1)]
        else:
            idx = idx - len(self.s1_normal)
            s1_path = self.s1_normal_path + self.s1_normal[random.randint(0, len(self.s1_normal)-1)]
            s2_path = self.s2_normal_path + self.s2_normal[idx]

        s1 = self.load_audio(s1_path)
        s2 = self.load_audio(s2_path)

        mix_0 = s1 + s2
        s1_status = np.ones(1, dtype=np.float32)
        s2_status = np.ones(1, dtype=np.float32)
        if random.random() > self.p:
            if random.random() > 0.5:
                s1_path_ = self.s1_abnormal_path + self.s1_abnormal[random.randint(0, len(self.s1_abnormal)-1)]
                s2_path_ = self.s2_normal_path + self.s2_normal[random.randint(0, len(self.s2_normal)-1)]
                s1_status[0] = 0
            else:
                s1_path_ = self.s1_normal_path + self.s1_normal[random.randint(0, len(self.s1_normal)-1)]
                s2_path_ = self.s2_abnormal_path + self.s2_abnormal[random.randint(0, len(self.s2_abnormal)-1)]
                s2_status[0] = 0
        else:
            s1_path_ = self.s1_normal_path + self.s1_normal[random.randint(0, len(self.s1_normal)-1)]
            s2_path_ = self.s2_normal_path + self.s2_normal[random.randint(0, len(self.s2_normal)-1)]

        s1_ = self.load_audio(s1_path_)
        s2_ = self.load_audio(s2_path_)
        mix_1 = s1_ + s2_
        mix_0 = self.scaler.fit_transform(mix_0.reshape(-1, 1)).T
        mix_1 = self.scaler.fit_transform(mix_1.reshape(-1, 1)).T

        return torch.tensor(mix_0), torch.tensor(mix_1), torch.tensor(s1_status), torch.tensor(s2_status)

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        target_length = audio.shape[0]
        start_pos = np.random.randint(0, max(target_length - self.samples + 1, 1))
        end_pos = start_pos + self.samples
        return audio[start_pos:end_pos]


class IndusDatasetBoth(Dataset):
    def __init__(self, source_1_path, source_2_path, s1_normal, s1_abnormal, s2_normal, s2_abnormal, p=0.7, sr=16000,
                 l=4):
        self.p = p
        self.sr = sr
        self.samples = int(l * sr)
        self.scaler = preprocessing.MaxAbsScaler()

        self.s1_normal_path = source_1_path + '/normal/'
        self.s2_normal_path = source_2_path + '/normal/'
        self.s1_abnormal_path = source_1_path + '/abnormal/'
        self.s2_abnormal_path = source_2_path + '/abnormal/'

        self.s1_normal = s1_normal
        self.s1_abnormal = s1_abnormal
        self.s2_normal = s2_normal
        self.s2_abnormal = s2_abnormal

    def __len__(self):
        n = len(self.s1_normal) + len(self.s2_normal)
        return n

    def __getitem__(self, idx):
        if idx < len(self.s1_normal):
            # pick normal data
            s1_path = self.s1_normal_path + self.s1_normal[idx]
            s2_path = self.s2_normal_path + self.s2_normal[random.randint(0, len(self.s2_normal) - 1)]
        else:
            idx = idx - len(self.s1_normal)
            s1_path = self.s1_normal_path + self.s1_normal[random.randint(0, len(self.s1_normal) - 1)]
            s2_path = self.s2_normal_path + self.s2_normal[idx]

        s1 = self.load_audio(s1_path)
        s2 = self.load_audio(s2_path)

        mix_0 = s1 + s2
        s1_status = np.ones(1, dtype=np.float32)
        s2_status = np.ones(1, dtype=np.float32)
        p1, p2 = random.random(), random.random()
        if p1 > self.p > p2:
            s1_path_ = self.s1_abnormal_path + self.s1_abnormal[random.randint(0, len(self.s1_abnormal)-1)]
            s2_path_ = self.s2_normal_path + self.s2_normal[random.randint(0, len(self.s2_normal)-1)]
            s1_status[0] = 0
        elif p1 < self.p < p2:
            s1_path_ = self.s1_normal_path + self.s1_normal[random.randint(0, len(self.s1_normal) - 1)]
            s2_path_ = self.s2_abnormal_path + self.s2_abnormal[random.randint(0, len(self.s2_abnormal)-1)]
            s2_status[0] = 0
        elif p1 > self.p and p2 > self.p:
            s1_path_ = self.s1_abnormal_path + self.s1_abnormal[random.randint(0, len(self.s1_abnormal) - 1)]
            s2_path_ = self.s2_abnormal_path + self.s2_abnormal[random.randint(0, len(self.s2_abnormal) - 1)]
            s1_status[0] = 0
            s2_status[0] = 0
        else:
            s1_path_ = self.s1_normal_path + self.s1_normal[random.randint(0, len(self.s1_normal)-1)]
            s2_path_ = self.s2_normal_path + self.s2_normal[random.randint(0, len(self.s2_normal)-1)]

        s1_ = self.load_audio(s1_path_)
        s2_ = self.load_audio(s2_path_)
        mix_1 = s1_ + s2_
        mix_0 = self.scaler.fit_transform(mix_0.reshape(-1, 1)).T
        mix_1 = self.scaler.fit_transform(mix_1.reshape(-1, 1)).T
        # sf.write('t_0.wav', mix_0[0, :], samplerate=16000)
        # sf.write('t_1.wav', mix_1[0, :], samplerate=16000)

        return torch.tensor(mix_0), torch.tensor(mix_1), torch.tensor(s1_status), torch.tensor(s2_status)

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        target_length = audio.shape[0]
        start_pos = np.random.randint(0, max(target_length - self.samples + 1, 1))
        end_pos = start_pos + self.samples
        return audio[start_pos:end_pos]


class ValDataset(Dataset):
    def __init__(self, pump_path, slider_path, pump_normal, pump_abnormal, slider_normal, slider_abnormal, p=0.5, sr=16000, l=4):
        self.p = p
        self.sr = sr
        self.samples = int(l * sr)
        self.pump_normal_path = pump_path + '/normal/'
        self.pump_abnormal_path = pump_path + '/abnormal/'
        self.slider_normal_path = slider_path + '/normal/'
        self.slider_abnormal_path = slider_path + '/abnormal/'
        self.pump_normal = pump_normal
        self.pump_abnormal = pump_abnormal
        self.slider_normal = slider_normal
        self.slider_abnormal = slider_abnormal
        self.scaler = preprocessing.MaxAbsScaler()

    def __len__(self):
        n = len(self.pump_abnormal) + len(self.slider_abnormal)
        return n

    def __getitem__(self, idx):
        pump_status = np.ones(1, dtype=np.float32)
        slider_status = np.ones(1, dtype=np.float32)
        s1_path = self.pump_normal_path + self.pump_normal[random.randint(0, len(self.pump_normal) - 1)]
        s2_path = self.slider_normal_path + self.slider_normal[random.randint(0, len(self.slider_normal) - 1)]
        s1 = self.load_audio(s1_path)
        s2 = self.load_audio(s2_path)
        mix_0 = s1 + s2
        if idx < len(self.pump_abnormal):
            pump_status[0] = 0
            # pick pump abnormal + slider normal
            s1_path_ = self.pump_abnormal_path + self.pump_abnormal[idx]
            s2_path_ = self.slider_normal_path + self.slider_normal[random.randint(0, len(self.slider_normal)-1)]
        else:
            slider_status[0] = 0
            idx = idx - len(self.pump_abnormal)
            s1_path_ = self.pump_normal_path + self.pump_normal[random.randint(0, len(self.pump_normal)-1)]
            s2_path_ = self.slider_abnormal_path + self.slider_abnormal[idx]

        s1_ = self.load_audio(s1_path_)
        s2_ = self.load_audio(s2_path_)
        mix_1 = s1_ + s2_

        mix_0 = self.scaler.fit_transform(mix_0.reshape(-1, 1)).T
        mix_1 = self.scaler.fit_transform(mix_1.reshape(-1, 1)).T
        return torch.tensor(mix_0), torch.tensor(mix_1), torch.tensor(pump_status), torch.tensor(slider_status)

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        target_length = audio.shape[0]
        start_pos = np.random.randint(0, max(target_length - self.samples + 1, 1))
        end_pos = start_pos + self.samples
        return audio[start_pos:end_pos]


class IndusDatasetMulti(Dataset):
    def __init__(self, sources_path, normals_ids, abnormals_ids, p=0.5, sr=16000, l=4, n=3200, state="train"):
        if state == "train":
            self.p = pow(p, 1/len(sources_path))
        else:
            self.p = p
        self.sr = sr
        self.samples = int(l * sr)
        self.scaler = preprocessing.MaxAbsScaler()
        self.normal_paths = [i + '/normal/' for i in sources_path]
        self.abnormal_paths = [i + '/abnormal/' for i in sources_path]
        self.normals_ids = normals_ids
        self.abnormals_ids = abnormals_ids
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        n_sources = len(self.normal_paths)
        status = np.ones(n_sources, dtype=np.float32)
        mix = np.zeros(self.samples, dtype=np.float32)
        mix_ = np.zeros(self.samples, dtype=np.float32)
        for i in range(n_sources):
            n_normal = len(self.normals_ids[i])
            n_abnormal = len(self.abnormals_ids[i])
            path = self.normal_paths[i] + self.normals_ids[i][random.randint(0, n_normal-1)]
            data = self.load_audio(path)
            p = random.random()
            if p > self.p:
                path_ = self.abnormal_paths[i] + self.abnormals_ids[i][random.randint(0, n_abnormal-1)]
                status[i] = 0
            else:
                path_ = self.normal_paths[i] + self.normals_ids[i][random.randint(0, n_normal-1)]
            data_ = self.load_audio(path_)
            mix += data
            mix_ += data_
        mix = self.scaler.fit_transform(mix.reshape(-1, 1)).T
        mix_ = self.scaler.fit_transform(mix_.reshape(-1, 1)).T

        sf.write('./mix.wav', mix_[0, :], samplerate=16000)

        return torch.tensor(mix), torch.tensor(mix_), torch.tensor(status)

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        target_length = audio.shape[0]
        start_pos = np.random.randint(0, max(target_length - self.samples + 1, 1))
        end_pos = start_pos + self.samples
        return audio[start_pos:end_pos]


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    import os
    from tqdm import tqdm

    def get_files(path):
        wav_files = []
        for _, _, files in os.walk(path):
            for f in files:
                if f.split('.')[-1] == 'wav':
                    wav_files.append(f)
        return wav_files

    pump_normal_path = '../../MIMII/pump/id_00/normal'
    pump_abnormal_path = '../../MIMII/pump/id_00/abnormal'
    valve_normal_path = '../../MIMII/valve/id_00/normal'
    valve_abnormal_path = '../../MIMII/valve/id_00/abnormal'
    pump_normal_files = get_files(pump_normal_path)
    pump_abnormal_files = get_files(pump_abnormal_path)
    valve_normal_files = get_files(valve_normal_path)
    valve_abnormal_files = get_files(valve_abnormal_path)

    train_pump_normal, val_pump_normal = train_test_split(pump_normal_files, random_state=42, test_size=0.2)
    train_valve_normal, val_valve_normal = train_test_split(valve_normal_files, random_state=42, test_size=0.2)
    train_pump_abnormal, val_pump_abnormal = train_test_split(pump_abnormal_files, random_state=42, test_size=0.8)
    train_valve_abnormal, val_valve_abnormal = train_test_split(valve_abnormal_files, random_state=42, test_size=0.8)

    pump_path = '../../MIMII/pump/id_00'
    valve_path = '../../MIMII/valve/id_00'

    train_dataset = IndusDatasetBoth(pump_path, valve_path, train_pump_normal, train_pump_abnormal, train_valve_normal,
                                     train_valve_abnormal)
    val_dataset = IndusDatasetBoth(pump_path, valve_path, val_pump_normal, val_pump_abnormal, val_valve_normal,
                                   val_valve_abnormal)
    for i in tqdm(range(len(train_dataset))):
        d = train_dataset[i]

    for i in tqdm(range(len(val_dataset))):
        d = val_dataset[i]




