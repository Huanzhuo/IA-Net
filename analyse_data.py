import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.cluster import KMeans


def z_score_analyse(labels, distances):
    labels = -labels + 1
    # scaler = preprocessing.StandardScaler()
    # mean = np.mean(distances)
    # std = np.std(distances)
    # p = (distances - min(distances)) / (max(distances) - min(distances))
    # thres, fpr, tpr = roc_curve(labels, p)
    # auc_socre = auc(fpr, tpr)
    d_abnormal = distances[:43]
    d_normal = distances[43:]
    mean_normal = np.mean(d_normal)

    labels_ = distances > 0.3

    normal = labels_[43:]
    n = np.sum(normal)
    n_ab = np.sum(labels_[:43])

def cal_roc_curve(labels, distances):
    


s1_labels_path = './eval_result/pump00_slider00_margin3/s1_labels.npy'
s2_labels_path = './eval_result/pump00_slider00_margin3/s2_labels.npy'
s1_d_path = './eval_result/pump00_slider00_margin3/s1_distances.npy'
s2_d_path = './eval_result/pump00_slider00_margin3/s2_distances.npy'

s1_labels = np.load(s1_labels_path)
s2_labels = np.load(s2_labels_path)
s1_d = np.load(s1_d_path)
s2_d = np.load(s2_d_path)

z_score_analyse(s1_labels, s1_d)


print('test')