import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

classes = ['pump_abnormal', 'pump_normal', 'slider_abnormal', 'slider_normal']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(4):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    plt.show()


def get_list(pump_path, slider_path):
    pump_normal = pump_path + '/normal'
    pump_abnormal = pump_path + '/abnormal'
    slider_normal = slider_path + '/normal'
    slider_abnormal = slider_path + '/abnormal'
    p_n_list = []
    p_ab_list = []
    s_n_list = []
    s_ab_list = []
    for _, _, files in os.walk(pump_normal):
        for f in files:
            if f.split('.')[-1] == 'wav':
                p_n_list.append(f)

    for _, _, files in os.walk(pump_abnormal):
        for f in files:
            if f.split('.')[-1] == 'wav':
                p_ab_list.append(f)

    for _, _, files in os.walk(slider_normal):
        for f in files:
            if f.split('.')[-1] == 'wav':
                s_n_list.append(f)

    for _, _, files in os.walk(slider_abnormal):
        for f in files:
            if f.split('.')[-1] == 'wav':
                s_ab_list.append(f)
    return p_n_list, p_ab_list, s_n_list, s_ab_list