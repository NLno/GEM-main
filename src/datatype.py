from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

def random_sample(array, size: int, replace=True):
    """随机抽样: 每个样本等概率抽样
    :param array: 待采样数组
    :param size: 采样个数
    :param replace: 是否放回，True为有放回的抽样，False为无放回的抽样
    """
    return np.random.choice(array, size=size, replace=replace)


def cluster_sample(array):
    """聚类抽样: 也称整群抽样，先对样本聚出多个类，然后随机的抽类，抽中哪个类，这一类的所有样本点都会被抽出来，不会对单个点进行抽样
    :param array: 样本点
    """
    label = DBSCAN(eps=30, min_samples=3).fit(array).labels_  # 使用DBSCAN做聚类，这个可以换
    select_cluster = random_sample(np.unique(label), 1)  # 随机选择一个类
    return array[label == select_cluster]


def systematic_sample(array, step):
    """系统抽样: 以固定的节奏从总体中抽样，隔step个抽1个，再隔step个抽一个，循环下去
    :param array: 样本点
    :param step: 步长
    """
    select_index = list(range(0, len(array), 3))
    return array[select_index]


def stratify_sample(array, label, size: int):
    """分层抽样: 先按照容量，给每个样本一些指标，然后样本内等概率抽样
    :param array: 样本数据
    :param label: 样本类别
    :param size: 采样个数
    """
    stratified_sample, _ = train_test_split(array, train_size=size, stratify=label)
    return stratified_sample


def main():
    # 构造数据
    array_data = np.arange(0, 100)  # 待取样数据
    array_label = np.random.random_integers(0, 5, size=100)  # 类别
    # 开始采样
    cluster_result = cluster_sample(np.random.random_integers(0, 100, size=(100, 5)))


if __name__ == '__main__':
    main()
