import numpy as np
import pandas
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import pre_process_lidar
import matplotlib.pyplot as plt
import plot_scenes
import seaborn as sns
from sklearn import preprocessing
import pandas as pd

def read_3D_lidar():
    data_path = "../data/lidar/s008/lidar_train_raymobtime.npz"
    data_lidar_process_all_train, data_position_rx_train, data_position_tx_train = pre_process_lidar.read_data(data_path)


    data_path = "../data/lidar/s008/lidar_validation_raymobtime.npz"
    data_lidar_process_all_val, data_position_rx_val, data_position_tx_val = pre_process_lidar.read_data (data_path)

    data_lidar_3D_train = np.concatenate((data_lidar_process_all_train, data_lidar_process_all_val), axis=0)


    data_path = "../data/lidar/s009/lidar_test_raymobtime.npz"
    data_lidar_process_all_test, data_position_rx_test, data_position_tx_test = pre_process_lidar.read_data (data_path)

    return data_lidar_3D_train, data_lidar_process_all_test, data_position_rx_train, data_position_tx_train

def read_2D_lidar():
    data_path = "../data/lidar/s008/lidar_train_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = pre_process_lidar.read_data(data_path)
    pre_process_data_lidar_2D_vector_train, pre_process_data_lidar_2D_train = pre_process_lidar.pre_process_data_lidar_2D(
        data_lidar_process_all, data_position_rx, data_position_tx)

    data_path = "../data/lidar/s008/lidar_validation_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = pre_process_lidar.read_data(data_path)
    pre_process_data_lidar_2D_vector_val, pre_process_data_lidar_2D_val = pre_process_lidar.pre_process_data_lidar_2D(
        data_lidar_process_all, data_position_rx, data_position_tx)

    data_lidar_2D_train = np.concatenate((pre_process_data_lidar_2D_train, pre_process_data_lidar_2D_val), axis=0)
    data_lidar_2D_vector_train = np.concatenate((pre_process_data_lidar_2D_vector_train, pre_process_data_lidar_2D_vector_val), axis=0)

    data_path = "../data/lidar/s009/lidar_test_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = pre_process_lidar.read_data(data_path)
    pre_process_data_lidar_2D_vector_test, pre_process_data_lidar_2D_test = pre_process_lidar.pre_process_data_lidar_2D(
        data_lidar_process_all, data_position_rx, data_position_tx)

    data_lidar_2D_test = pre_process_data_lidar_2D_test

    return data_lidar_2D_train, data_lidar_2D_test, data_lidar_2D_vector_train, pre_process_data_lidar_2D_vector_test

def pca(data_train, data_test, nro_components, binarization_type):
    plot = False

    components = nro_components
    pca = PCA(n_components=components)
    principal_component_lidar = pca.fit(data_train)
    pca_lidar_comp = principal_component_lidar.components_
    var_explained = pca.explained_variance_ratio_
    var_explained_cum = np.cumsum(var_explained)
    print("Variance explained by each component: ", var_explained)
    print("Cumulative variance explained: ", var_explained_cum)

    if plot:
        fig = plt.figure (figsize=(8, 5))
        plt.plot(var_explained, 'ro-', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        plt.show()

    PCA_components_train = pca.transform(data_train)
    PCA_components_test = pca.transform(data_test)

    if binarization_type == 'thermometer_1':
        components_binarized_train, components_binarized_test = thermometer_1(PCA_components_train, PCA_components_test, components)
    elif binarization_type == 'thermometer_2':
        components_binarized_train, components_binarized_test = thermometer_2(PCA_components_train, PCA_components_test)
    elif binarization_type == 'simple_threshold':
        components_binarized_train, components_binarized_test = binarize_data_by_threshold(PCA_components_train, PCA_components_test, 0.5)

    return components_binarized_train, components_binarized_test

def thermometer_1(data_train, data_test, components_of_pca):

    abs_value_data = np.round(np.abs(data_train))
    data_int = abs_value_data.astype(int)
    max_value = np.max(data_int)

    abs_value_data_test = np.round (np.abs (data_test))
    data_int_test = abs_value_data_test.astype (int)
    max_value_test = np.max (data_int_test)

    if max_value < max_value_test:
        size_of_termometer = max_value_test
    else:
        size_of_termometer = max_value

    binarized_data = np.zeros((len(data_int), (size_of_termometer*components_of_pca)), dtype=np.int8)
    for i in range(len(data_int)):
        termomether = np.zeros((components_of_pca, size_of_termometer), dtype=np.int8)
        sample = data_int[i]
        for j in range(components_of_pca):
            termomether[j, 0:sample[j]] = 1
        binarized_data[i] = termomether.reshape(1, size_of_termometer*components_of_pca)

    binarized_data_test = np.zeros((len(data_int_test), (size_of_termometer*components_of_pca)), dtype=np.int8)
    for i in range(len(data_int_test)):
        termomether = np.zeros((components_of_pca, size_of_termometer), dtype=np.int8)
        sample = data_int_test[i]
        for j in range(components_of_pca):
            termomether[j, 0:sample[j]] = 1
        binarized_data_test[i] = termomether.reshape(1, size_of_termometer*components_of_pca)

    return binarized_data, binarized_data_test

def thermometer_2(data_train, data_test):

    abs_value_data_train = np.round(np.abs(data_train))
    sum_of_components_train = np.sum(abs_value_data_train, axis=1).astype(int)
    max_value_train = np.max (sum_of_components_train)

    abs_value_data_test = np.round (np.abs (data_test))
    sum_of_components_test = np.sum (abs_value_data_test, axis=1).astype (int)
    max_value_test = np.max (sum_of_components_test)


    if max_value_train < max_value_test:
        size_of_thermometer = max_value_test
    else:
        size_of_thermometer = max_value_train

    thermometer_train = np.zeros((len(data_train), size_of_thermometer), dtype=np.int8)
    thermometer_test = np.zeros((len (data_test), size_of_thermometer), dtype=np.int8)

    for i in range(len(data_train)):
        sample_train = sum_of_components_train[i]
        thermometer_train[i, 0:sample_train] = 1

    for i in range(len(data_test)):
        sample_test = sum_of_components_test[i]
        thermometer_test[i, 0:sample_test] = 1


    return thermometer_train, thermometer_test

def binarize_data_by_threshold(data_train, data_test, threshold):

    abs_value_data_train = np.abs(data_train)
    abs_value_data_test = np.abs(data_test)

    #normalizer = preprocessing.Normalizer().fit(abs_value_data_train)
    #data_normalized_train_1 = normalizer.transform(abs_value_data_train)
    #data_normalized_test_1 = normalizer.transform(abs_value_data_test)

    min_max_scaler = preprocessing.MinMaxScaler()
    train_data = min_max_scaler.fit_transform(data_train)
    test_data = min_max_scaler.fit_transform(data_test)

    #data_normalized_train = preprocessing.normalize(abs_value_data_train, norm='l1')
    #data_normalized_test = preprocessing.normalize(abs_value_data_test, norm='l1')

    train_data[train_data < threshold] = 0
    train_data[train_data >= threshold] = 1

    test_data[test_data < threshold] = 0
    test_data[test_data >= threshold] = 1

    return train_data.astype(np.int8) , test_data.astype(np.int8)


def plot_3_variables():

    x = [20, 30, 40, 50]
    y = [21.5, 21.4, 21.08, 20.89]
    y2 = [11.4, 11.4, 11.9, 12.5]
    z = [75.5, 79.9, 82.3, 83.9]
    z2 = [79.9, 79.9, 82.3, 83.9]

    components = ("20", "30", "40", "50")
    scores = {
        'Termomether 1': (21.5, 21.4, 21.08, 20.89),
        'Termomether 2': (11.4, 11.4, 11.9, 12.5),
        #'Flipper Length': (189.95, 195.82, 217.19),
    }

    x = np.arange(len(components))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots (layout='constrained')

    for attribute, measurement in scores.items ():
        offset = width * multiplier
        rects = ax.bar (x + offset, measurement, width, label=attribute)
        ax.bar_label (rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel ('Accuracy [%]')
    ax.set_xlabel ('Components of PCA')
    ax.set_title ('Performance of WiSARD using PCA \n Input LiDAR 3D with elimination of variance', size=14)
    ax.set_xticks (x + width, components)
    ax.legend (loc='best', ncols=3)
    ax.set_ylim (0, 25)

    plt.show()














    '''
    pca = PCA(n_components=components)
    principal_component_lidar_2D_train = pca.fit(data_lidar_2D_train_without_var)
    #principal_component_lidar_3D = pca.fit(data_lidar_3D_train_without_var)

    pca_test = PCA(n_components=components)
    principal_component_lidar_2D_test = pca_test.fit(data_lidar_2D_test_without_var)

    pca_lidar_2D_comp = principal_component_lidar_2D_train.components_
    var_explained = pca.explained_variance_ratio_
    var_explained_cum = np.cumsum(var_explained)

    fig = plt.figure(figsize=(8,5))
    plt.plot(var_explained, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.show()

    PCA_pca = pca.transform(data_lidar_2D_train_without_var)
    PCA_1_train = pca.transform(data_lidar_2D_train_without_var)[:,0]
    PCA_2_train = pca.transform(data_lidar_2D_train_without_var)[:,1]
    PCA_3_train = pca.transform(data_lidar_2D_train_without_var)[:,2]
    sum_of_all_PCA_train = (PCA_1_train+PCA_2_train+PCA_3_train).astype(int)

    PCA_1_test = pca_test.transform(data_lidar_2D_test_without_var)[:,0]
    PCA_2_test = pca_test.transform(data_lidar_2D_test_without_var)[:,1]
    PCA_3_test = pca_test.transform(data_lidar_2D_test_without_var)[:,2]
    sum_of_all_PCA_test = (PCA_1_test+PCA_2_test+PCA_3_test).astype(int)

    #Thermomether implementation
    termomether_size_train = np.max(sum_of_all_PCA_train)
    termomether_size_test = np.max (sum_of_all_PCA_test)

    if termomether_size_train < termomether_size_test:
        termomether_size = termomether_size_test
    else:
        termomether_size = termomether_size_train

    feature_binarized_train = np.zeros((len(sum_of_all_PCA_train), termomether_size), dtype=np.int8)

    for i in range(len(sum_of_all_PCA_train)):
        sample_binarized = np.zeros(termomether_size, dtype=np.int8)
        sample_binarized[0: sum_of_all_PCA_train[i]] = 1
        if sum_of_all_PCA_train[i] < 0:
            sample_binarized = np.flip(sample_binarized)
        feature_binarized_train[i] = sample_binarized


    feature_binarized_test = np.zeros((len(sum_of_all_PCA_test), termomether_size), dtype=np.int8)

    for i in range(len(sum_of_all_PCA_test)):
        sample_binarized = np.zeros(termomether_size, dtype=np.int8)
        sample_binarized[0: sum_of_all_PCA_test[i]] = 1
        if sum_of_all_PCA_test[i] < 0:
            sample_binarized = np.flip(sample_binarized)
        feature_binarized_test[i] = sample_binarized

    return feature_binarized_train, feature_binarized_test
    '''


#pca(nro_components=20)
#plot_3_variables()








#pca(10)