import torch
import random
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn import preprocessing
import soundfile as sf


class ValDataset(Dataset):
    def __init__(self, num_val, s1_path, s2_path, s1_normal, s1_abnormal, s2_normal, s2_abnormal, p=0.3, sr=16000, l=4):
        self.p = p
        self.sr = sr
        self.num_val = num_val
        self.samples = int(l * sr)
        self.s1_normal_path = s1_path + '/normal/'
        self.s1_abnormal_path = s1_path + '/abnormal/'
        self.s2_normal_path = s2_path + '/normal/'
        self.s2_abnormal_path = s2_path + '/abnormal/'
        self.scaler = preprocessing.MaxAbsScaler()
        self.s1_normal = s1_normal
        self.s1_abnormal = s1_abnormal
        self.s2_normal = s2_normal
        self.s2_abnormal = s2_abnormal

    def __len__(self):
        return self.num_val

    def __getitem__(self, idx):
        s1_path = self.s1_normal_path + self.s1_normal[random.randint(0, len(self.s1_normal)-1)]
        s2_path = self.s2_normal_path + self.s2_normal[random.randint(0, len(self.s2_normal)-1)]
        s1 = self.load_audio(s1_path)
        s2 = self.load_audio(s2_path)
        mix_0 = s1 + s2
        s1_status = np.ones(1, dtype=np.float32)
        s2_status = np.ones(1, dtype=np.float32)
        p1, p2 = random.random(), random.random()

        if p2 < self.p < p1:
            # s1 is abnormal
            s1_path_ = self.s1_abnormal_path + self.s1_abnormal[random.randint(0, len(self.s1_abnormal)-1)]
            s2_path_ = self.s2_normal_path + self.s2_normal[random.randint(0, len(self.s2_normal)-1)]
            s1_status[0] = 0
        elif p1 < self.p < p2:
            s1_path_ = self.s1_normal_path + self.s1_normal[random.randint(0, len(self.s1_normal)-1)]
            s2_path_ = self.s2_abnormal_path + self.s2_abnormal[random.randint(0, len(self.s2_abnormal)-1)]
            s2_status[0] = 0
        elif p1 > self.p and p2 > self.p:
            s1_path_ = self.s1_abnormal_path + self.s1_abnormal[random.randint(0, len(self.s1_abnormal)-1)]
            s2_path_ = self.s2_abnormal_path + self.s2_abnormal[random.randint(0, len(self.s2_abnormal)-1)]
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
        return torch.tensor(mix_0), torch.tensor(mix_1), torch.tensor(s1_status), torch.tensor(s2_status)

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        target_length = audio.shape[0]
        start_pos = np.random.randint(0, max(target_length - self.samples + 1, 1))
        end_pos = start_pos + self.samples
        return audio[start_pos:end_pos]
