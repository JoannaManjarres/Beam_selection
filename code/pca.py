import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import pre_process_lidar
import matplotlib.pyplot as plt
import plot_scenes
import seaborn as sns

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

def pca():

    #data_lidar_2D_train_without_var, data_lidar_2D_test_without_var = pre_process_lidar.data_lidar_2D_binary_without_variance()
    #data_lidar_3D_train_without_var, data_lidar_3D_test_without_var = pre_process_lidar.data_lidar_3D_binary_without_variance()
    '''Read 3D lidar data'''
    '''transform to 2D matriz, preserve the original height information'''
    data_lidar_in_vector_train, data_lidar_in_vector_test, data_lidar_2D_matrix_train, data_lidar_2D_matrix_test = pre_process_lidar.process_data_lidar_into_2D_matrix ()

    # preprocess data
    # Data without variance
    th = 0
    selector = VarianceThreshold (threshold=th)
    vt_train = selector.fit (data_lidar_in_vector_train)
    data_lidar_2D_train_without_var = data_lidar_in_vector_train [:, vt_train.variances_ > th]

    vt_test = selector.fit (data_lidar_in_vector_test)
    data_lidar_2D_test_without_var = data_lidar_in_vector_test [:, vt_test.variances_ > th]

    correlation = np.corrcoef(data_lidar_2D_train_without_var.T)

    #heat map
    #sns.heatmap(correlation)
    #plt.show()

    components = 3

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

    '''Thermomether implementation'''
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

#pca()







#pca()