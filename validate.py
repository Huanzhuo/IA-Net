import torch
from utils import *
from losses.loss import ContrastiveLoss
from torch.utils.data import DataLoader
from dataset.val_dataset import ValDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_files(path):
    wav_files = []
    for _, _, files in os.walk(path):
        for f in files:
            if f.split('.')[-1] == 'wav':
                wav_files.append(f)
    return wav_files


def validate_vis(model_path):
    # prepare dataset
    start_pos = 0
    step = 4

    s1_normal_path = '../MIMII/pump/id_00/normal'
    s1_abnormal_path = '../MIMII/pump/id_00/abnormal'
    s2_normal_path = '../MIMII/slider/id_00/normal'
    s2_abnormal_path = '../MIMII/slider/id_00/abnormal'
    s1_normal_files = get_files(s1_normal_path)
    s1_abnormal_files = get_files(s1_abnormal_path)
    s2_normal_files = get_files(s2_normal_path)
    s2_abnormal_files = get_files(s2_abnormal_path)

    train_s1_normal, val_s1_normal = train_test_split(s1_normal_files, random_state=42, test_size=0.2)
    train_s2_normal, val_s2_normal = train_test_split(s2_normal_files, random_state=42, test_size=0.2)
    train_s1_abnormal, val_s1_abnormal = train_test_split(s1_abnormal_files, random_state=42, test_size=0.95)
    train_s2_abnormal, val_s2_abnormal = train_test_split(s2_abnormal_files, random_state=42, test_size=0.95)

    s1_path = '../MIMII/pump/id_00'
    s2_path = '../MIMII/slider/id_00'
    dataset = ValDataset(2000, s1_path, s2_path, val_s1_normal, val_s1_abnormal, val_s2_normal, val_s2_abnormal, p=0.7)
    abnormal_loader = DataLoader(dataset, batch_size=step, shuffle=False)

    # prepare model
    model = torch.load(model_path)['model']

    # prepare criterion
    criterion = ContrastiveLoss()
    model.eval()
    labels = np.ones([2, len(dataset)], dtype=np.float32)
    distances = np.zeros([2, len(dataset)], dtype=np.float32)
    save_features = np.zeros([2, len(dataset), 256])
    save_features_ = np.zeros([2, len(dataset), 256], dtype=np.float32)

    model.eval()

    for i, (mix_0, mix_1, labels_0, labels_1) in enumerate(tqdm(abnormal_loader)):
        mix_0 = mix_0.cuda()
        mix_1 = mix_1.cuda()
        labels_0 = labels_0.cuda()
        labels_1 = labels_1.cuda()
        features, features_ = model(mix_0, mix_1)
        s1_d = (features[0] - features_[0]).pow(2).sum(1)
        s2_d = (features[1] - features_[1]).pow(2).sum(1)
        # loss, s1_distance, s2_distance = criterion(s1_features, s2_features, labels_0, labels_1)

        labels[0, start_pos: start_pos + step] = labels_0.cpu().numpy()[:, 0]
        labels[1, start_pos: start_pos + step] = labels_1.cpu().numpy()[:, 0]
        distances[0, start_pos: start_pos + step] = s1_d.detach().cpu().numpy()
        distances[1, start_pos: start_pos + step] = s2_d.detach().cpu().numpy()
        save_features[0, start_pos: start_pos + step, :] = features[0].detach().cpu().numpy()
        save_features[1, start_pos: start_pos + step, :] = features[1].detach().cpu().numpy()
        save_features_[0, start_pos: start_pos + step, :] = features_[0].detach().cpu().numpy()
        save_features_[1, start_pos: start_pos + step, :] = features_[1].detach().cpu().numpy()
        start_pos += step

    np.save("labels.npy", labels)
    np.save("distances.npy", distances)

    np.save("features.npy", save_features)
    np.save("features_.npy", save_features_)


def plot():
    labels = np.load('./s1_labels.npy')
    distances = np.load('./s1_distances.npy')
    import seaborn as sns
    features = np.load('./s1_features.npy')
    l2_distances = np.load('./s1_l2_distance.npy')
    # plot_embeddings(features, labels, xlim=None, ylim=None)
    print('test')


if __name__ == '__main__':
    # path = './result/valve_fan/margin_3_0.80_testset/model_best.pt'
    path = './result_identify/pump_slider/margin3_for_all_ids/model_best.pt'
    validate_vis(path)
    # plot()
