import torch
import librosa
from sklearn import preprocessing
import numpy as np
import soundfile as sf

# IA-Net-Lite
# MODEL_PATH = "./results/mobile_net/all_ids/id00/256/2021-02-27-02/model_best.pt"
# IA-Net-Lite with Y-Net
MODEL_PATH = "./results/pump_slider_fan_valve/model_best.pt"

source_dict = {0: "pump",
               1: "slider",
               2: "fan",
               3: "valve"}


def load_audio(path):
    audio, _ = librosa.load(path, sr=16000, mono=True)
    target_length = audio.shape[0]
    start_pos = np.random.randint(0, max(target_length - 64000 + 1, 1))
    end_pos = start_pos + 64000
    return audio[start_pos:end_pos]


def collect_data(source_id):
    path = "./demo_files/{}/{}/{}.wav"
    # path_r = "./demo_files/{}/{}/{}.wav"
    path_r = "./demo_files/s{}.wav"
    print("Processing source {}...".format(source_dict[source_id]))
    i = input("Normal (N) or Abnormal (A): ")
    if i == 'N':
        name = "normals"
    elif i == 'A':
        name = "abnormals"
    else:
        raise ValueError("The input should be 'N' or 'A'")
    i = input("input id (from 0 to 5): ")
    if 0 <= int(i) <= 5:
        id = int(i)
    else:
        raise ValueError("The id should be between 0 to 5")
    path = path.format(name, source_dict[source_id], id)
    path_r = path_r.format(source_id)
    # i = random.randint(0, 5)
    # path_r = path_r.format("normals", source_dict[source_id], random.randint(0, 5))
    data = load_audio(path)
    # data_ = load_audio(path.format("normals", source_dict[source_id], id))
    data_r = load_audio(path_r)
    return data, data_r, name


def demo():
    print("Loading model...")
    model = torch.load(MODEL_PATH, map_location="cpu")['model']
    scaler = preprocessing.MaxAbsScaler()
    mix = np.zeros(64000, dtype=np.float32)
    mix_r = np.zeros(64000, dtype=np.float32)

    names = []
    for i in range(4):
        data, data_r, name = collect_data(i)
        mix += data
        mix_r += data_r
        names.append(name[:-1])
    mix = scaler.fit_transform(mix.reshape(-1, 1)).T
    mix_r = scaler.fit_transform(mix_r.reshape(-1, 1)).T
    print("The mixture is stored as mix.wav")
    sf.write('./mix.wav', mix[0, :], samplerate=16000)
    mix = torch.tensor(mix).unsqueeze(0)
    mix_r = torch.tensor(mix_r).unsqueeze(0)

    print("Processing...")
    features, features_ = model(mix_r, mix)
    distances = []
    for j in range(4):
        d = (features_[j] - features[j]).pow(2).sum(1)
        distances.append(d.item())
    # features_r, features = model([mix, mix_r])
    print("----------------------")
    print("The anomalys score for pump, slider, fan and valve are:")
    print(distances)
    print("The sources are:...")
    print(names)

    # distances_ = []
    # features, features_ = model([mix_r, mix_])
    # for j in range(4):
    #     d = (features_[j] - features[j]).pow(2).sum(1)
    #     distances_.append(d.item())
    # print(distances_)
    # [0.0006447461200878024, 0.33421263098716736, 0.11688090860843658, 0.010976912453770638]


if __name__ == '__main__':
    demo()
