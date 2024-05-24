import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import csv
import cv2
from scipy import ndimage
from sklearn.feature_selection import VarianceThreshold
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data, color

def read_data(data_path):
    label_cache_file = np.load(data_path)
    data_lidar = label_cache_file['input']

    data_lidar_process = np.where(data_lidar == -1, 0, data_lidar)
    data_lidar_process_all = np.where(data_lidar_process == -2, 1, data_lidar_process)

    data_position_rx = np.where(data_lidar == -2, 1, 0)
    data_position_tx = np.where(data_lidar == -1, 1, 0)

    return data_lidar_process_all, data_position_rx, data_position_tx

def process_data_rx_like_cube_of_s008_and_s009():
    data_path = "../data/lidar/s008/lidar_train_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = read_data(data_path)
    all_data_train = pre_process_data_rx_like_cube(data_lidar_process_all,
                                                   data_position_rx,
                                                   data_position_tx,
                                                   plot=True,
                                                   sample_for_plot=0,
                                                   data_name='s008')

    data_path = "../data/lidar/s008/lidar_validation_raymobtime.npz"
    data_lidar_process_all_val, data_position_rx_val, data_position_tx_val = read_data(data_path)
    all_data_val = pre_process_data_rx_like_cube(data_lidar_process_all_val,
                                                  data_position_rx_val,
                                                  data_position_tx_val,
                                                  plot=False,
                                                  sample_for_plot=0,
                                                  data_name='s008')

    data_lidar_train = np.concatenate((all_data_train, all_data_val), axis=0)
    saveDataPath = "../data/lidar/s008/"
    np.savez(saveDataPath + 'all_data_lidar_+_rx_like_cube_train' + '.npz', data_lidar=data_lidar_train)

    np.savez(saveDataPath + 'lidar_val' + '.npz', data_lidar=all_data_val)
    np.savez(saveDataPath + 'lidar_train' + '.npz', data_lidar=all_data_train)


    data_path_test = "../data/lidar/s009/lidar_test_raymobtime.npz"
    data_lidar_process_all_test, data_position_rx_test, data_position_tx_test = read_data(data_path_test)
    data_lidar_test = pre_process_data_rx_like_cube(data_lidar_process_all_test,
                                                    data_position_rx_test,
                                                    data_position_tx_test,
                                                    plot=True,
                                                    sample_for_plot=0,
                                                    data_name='s009')
    saveDataPath = "../data/lidar/s009/"
    np.savez(saveDataPath + 'all_data_lidar_+_rx_like_cube_test' + '.npz', data_lidar=data_lidar_test)



def process_data_lidar_to_2D():
    save = False
    path = '../data/lidar/s008/lidar_train_raymobtime.npz'
    data_lidar_process_all_train, data_position_rx_train, data_position_tx_train = read_data(path)
    data_lidar_2D_train_vector, data_lidar_2D_train_matrix = pre_process_data_lidar_2D(data_lidar_process_all_train, data_position_rx_train, data_position_tx_train)

    path = '../data/lidar/s008/lidar_validation_raymobtime.npz'
    data_lidar_process_all_val, data_position_rx_val, data_position_tx_val = read_data(path)
    data_lidar_2D_val_vector,  data_lidar_2D_val_matrix = pre_process_data_lidar_2D(data_lidar_process_all_val, data_position_rx_val, data_position_tx_val)

    data_train_lidar_2D_matrix = np.concatenate((data_lidar_2D_train_matrix, data_lidar_2D_val_matrix), axis=0)

    path = '../data/lidar/s009/lidar_test_raymobtime.npz'
    data_lidar_process_all_test, data_position_rx_test, data_position_tx_test = read_data(path)
    data_lidar_2D_test_vector, data_lidar_2D_test_matrix = pre_process_data_lidar_2D(data_lidar_process_all_test, data_position_rx_test, data_position_tx_test)

    if save:
        saveDataPath = "../data/lidar/pre_process_data_2D/"
        np.savez(saveDataPath + 'all_data_lidar_2D_train' + '.npz', data_lidar=data_train_lidar_2D_matrix)
        np.savez(saveDataPath + 'all_data_lidar_2D_test' + '.npz', data_lidar=data_lidar_2D_test_matrix)

    return data_train_lidar_2D_matrix, data_lidar_2D_test_matrix
def pre_process_data_lidar_2D(data_lidar_process_all, data_position_rx, data_position_tx):

    plot = False
    sample_for_plot = 3

    samples = data_lidar_process_all.shape[0]
    data_lidar_2D = np.zeros([samples, 20, 200], dtype=np.int8)
    for sample in range(len(data_lidar_process_all)):
        test_matriz = np.zeros([20, 200], dtype=np.int8)
        for var in range(0, 20):
            for y in range(0,10):
                test_matriz[var, :] += data_lidar_process_all[sample, var, :, y]
            test_matriz[var, :] = np.where(test_matriz [var, :] > 0, 1, 0)
        data_lidar_2D[sample] = test_matriz

    dimension_of_data = data_lidar_process_all.shape[1]*data_lidar_process_all.shape[2]
    data_lidar_2D_as_vector = np.zeros ([samples, dimension_of_data], dtype=np.int8)

    #Reshape the data to be used in the model
    for i in range(samples):
        data_lidar_2D_as_vector[i] = data_lidar_2D [i, :, :].reshape(1, dimension_of_data)
        #b = data_lidar_process_all [i, :, :, :].reshape (1, dimension_of_coordenadas)
        #all_data [i] = np.concatenate ((position_of_rx_cube_as_vector, b), axis=1)

    if plot:
        plot_3D_scene (data_lidar_process_all, data_position_rx, data_position_tx, sample_for_plot=sample_for_plot)
        plot_2D_scene(data_lidar_2D, sample_for_plot=sample_for_plot)

    '''
    test_matriz = np.zeros ([20, 200], dtype=np.int8)

    test_matriz_0 = np.zeros ([20, 200], dtype=np.int8)
    test_matriz_1 = np.zeros ([20, 200], dtype=np.int8)
    test_matriz_2 = np.zeros ([20, 200], dtype=np.int8)
    test_matriz_3 = np.zeros ([20, 200], dtype=np.int8)
    test_matriz_4 = np.zeros ([20, 200], dtype=np.int8)
    test_matriz_5 = np.zeros ([20, 200], dtype=np.int8)
    test_matriz_6 = np.zeros ([20, 200], dtype=np.int8)
    test_matriz_7 = np.zeros ([20, 200], dtype=np.int8)
    test_matriz_8 = np.zeros ([20, 200], dtype=np.int8)
    test_matriz_9 = np.zeros ([20, 200], dtype=np.int8)
    for sample in range(len(data_lidar_process_all)):
        for var in range(0, 20):
            test_matriz_0[var, :] = data_lidar_process_all[sample, var, :, 0]
            test_matriz_1[var, :] = data_lidar_process_all[sample, var, :, 1]
            test_matriz_2[var, :] = data_lidar_process_all[sample, var, :, 2]
            test_matriz_3[var, :] = data_lidar_process_all[sample, var, :, 3]
            test_matriz_4[var, :] = data_lidar_process_all[sample, var, :, 4]
            test_matriz_5[var, :] = data_lidar_process_all[sample, var, :, 5]
            test_matriz_6[var, :] = data_lidar_process_all[sample, var, :, 6]
            test_matriz_7[var, :] = data_lidar_process_all[sample, var, :, 7]
            test_matriz_8[var, :] = data_lidar_process_all[sample, var, :, 8]
            test_matriz_9[var, :] = data_lidar_process_all[sample, var, :, 9]
            test_matriz[var, :] = test_matriz_0[var, :] + test_matriz_1[var, :] + test_matriz_2[var, :] + test_matriz_3[var, :] + test_matriz_4[var, :] + test_matriz_5[var, :] + test_matriz_6[var, :] + test_matriz_7[var, :] + test_matriz_8[var, :] + test_matriz_9[var, :]

            test_matriz[var, :] = np.where(test_matriz[var, :] > 0, 1, 0)
        data_lidar_2D[sample] = test_matriz

    plot_2D_scene(data_lidar_2D, sample_for_plot=3)
    '''

    return data_lidar_2D_as_vector, data_lidar_2D
def plot_2D_scene(data_lidar_2D, sample_for_plot):

    plt.imshow (data_lidar_2D[sample_for_plot], cmap='Blues', origin='lower', extent=[0, 200, 0, 20])
    plt.title ("Scene " + str(sample_for_plot))
    plt.tight_layout(h_pad=0.5)
def read_pre_processed_data_lidar_2D():
    path = '../data/lidar/pre_process_data_2D/all_data_lidar_2D_train.npz'
    data_cache_file = np.load(path)
    key = list(data_cache_file.keys())
    data_lidar_2D_train = data_cache_file[key[0]]

    path = '../data/lidar/pre_process_data_2D/all_data_lidar_2D_test.npz'
    data_cache_file = np.load(path)
    key = list(data_cache_file.keys())
    data_lidar_2D_test = data_cache_file[key[0]]

    return data_lidar_2D_train, data_lidar_2D_test


def process_data_rx_2D_like_thermomether():
    path = '../data/lidar/s008/lidar_train_raymobtime.npz'
    data_lidar_process_all, data_position_rx, data_position_tx = read_data (path)
    position_of_rx_2D_as_thermomether_train_s008 = pre_process_data_rx_2D_like_thermometer(data_position_rx)

    path = '../data/lidar/s008/lidar_validation_raymobtime.npz'
    data_lidar_process_all_val, data_position_rx_val, data_position_tx_val = read_data (path)
    position_of_rx_2D_as_thermomether_val_s008 = pre_process_data_rx_2D_like_thermometer(data_position_rx_val)

    position_of_rx_2D_as_thermomether_train = np.concatenate((position_of_rx_2D_as_thermomether_train_s008, position_of_rx_2D_as_thermomether_val_s008), axis=0)

    path = '../data/lidar/s009/lidar_test_raymobtime.npz'
    data_lidar_process_all_test, data_position_rx_test, data_position_tx_test = read_data (path)
    position_of_rx_2D_as_thermomether_test = pre_process_data_rx_2D_like_thermometer(data_position_rx_test)

    return position_of_rx_2D_as_thermomether_train, position_of_rx_2D_as_thermomether_test
def pre_process_data_rx_2D_like_thermometer(data_position_rx):#, data_name, plot=True, sample_for_plot=0):
    #path = '../data/lidar/s008/lidar_train_raymobtime.npz'
    #data_lidar_process_all, data_position_rx, data_position_tx = read_data (path)

    x_dimension = len(data_position_rx [0, :, 0, 0])
    y_dimension = len(data_position_rx [0, 0, :, 0])
    z_dimension = len(data_position_rx [0, 0, 0, :])
    dimension_of_coordenadas = x_dimension * y_dimension
    number_of_samples = data_position_rx.shape[0]

    data = data_position_rx.copy()
    position_of_rx_as_cube = data * 0

    position_of_rx_as_thermomether = np.zeros([number_of_samples, x_dimension, y_dimension], dtype=np.int8)

    for i in range (number_of_samples):
        #number_of_samples = 0
        x_rx, y_rx, z_rx = 0, 0, 0
        pos_rx_in_each_sample = data_position_rx[i, :, :, :]
        x_rx, y_rx, z_rx = np.unravel_index (pos_rx_in_each_sample.argmax (), pos_rx_in_each_sample.shape)
        #position_of_rx_as_cube[i, 0:x_rx, 0:y_rx, 0:z_rx] = 1
        position_of_rx_as_thermomether[i, 0:x_rx, 0:y_rx] = 1

    position_of_rx_as_thermomether_vector = np.zeros([number_of_samples, dimension_of_coordenadas], dtype=np.int8)
    for i in range(number_of_samples):
        position_of_rx_as_thermomether_vector[i] = position_of_rx_as_thermomether[i, :, :].reshape(1, dimension_of_coordenadas)

    #plot_2D_scene(position_of_rx_as_thermomether, sample_for_plot=3)
    #a=0

    #all_data = np.zeros ([number_of_samples, dimension_of_coordenadas * 2], dtype=np.int8)

    #for i in range (number_of_samples):
    #    position_of_rx_cube_as_vector = position_of_rx_as_cube [i, :, :, :].reshape (1, dimension_of_coordenadas)
    #    b = data_lidar_process_all [i, :, :, :].reshape (1, dimension_of_coordenadas)
    #    all_data [i] = np.concatenate ((position_of_rx_cube_as_vector, b), axis=1)

    return position_of_rx_as_thermomether_vector


def process_all_data_2D_with_rx_like_thermometer():
    path = '../data/lidar/s008/lidar_train_raymobtime.npz'
    data_lidar_process_all, data_position_rx, data_position_tx = read_data (path)
    data_lidar_2D_with_rx_as_vector_train_s008 = pre_process_all_data_2D_with_rx_like_thermometer(data_lidar_process_all, data_position_rx)

    path = '../data/lidar/s008/lidar_validation_raymobtime.npz'
    data_lidar_process_all_val, data_position_rx_val, data_position_tx_val = read_data (path)
    data_lidar_2D_with_rx_as_vector_val_s008 = pre_process_all_data_2D_with_rx_like_thermometer(data_lidar_process_all_val, data_position_rx_val)

    data_lidar_2D_with_rx_train = np.concatenate((data_lidar_2D_with_rx_as_vector_train_s008, data_lidar_2D_with_rx_as_vector_val_s008), axis=0)

    path = '../data/lidar/s009/lidar_test_raymobtime.npz'
    data_lidar_process_all_test, data_position_rx_test, data_position_tx_test = read_data (path)
    data_lidar_2D_with_rx_test = pre_process_all_data_2D_with_rx_like_thermometer(data_lidar_process_all_test, data_position_rx_test)

    return data_lidar_2D_with_rx_train, data_lidar_2D_with_rx_test
def pre_process_all_data_2D_with_rx_like_thermometer(data_lidar_process_all, data_position_rx):#, data_position_tx, data_name, plot=True, sample_for_plot=0):

    # Pre process all scenarios in 2D
    samples = data_lidar_process_all.shape[0]
    data_lidar_2D = np.zeros([samples, 20, 200], dtype=np.int8)
    for sample in range(len(data_lidar_process_all)):
        test_matriz = np.zeros([20, 200], dtype=np.int8)
        for var in range(0, 20):
            for y in range(0, 10):
                test_matriz[var, :] += data_lidar_process_all[sample, var, :, y]
            test_matriz[var, :] = np.where(test_matriz[var, :] > 0, 1, 0)
        data_lidar_2D[sample] = test_matriz


    # pre-process of Rx as 2D-Thermometer
    x_dimension = len(data_position_rx[0, :, 0, 0])
    y_dimension = len(data_position_rx[0, 0, :, 0])
    dimension_of_coordenadas = x_dimension * y_dimension
    number_of_samples = data_position_rx.shape[0]

    #data = data_position_rx.copy()
    #position_of_rx_as_cube = data * 0
    dimension_of_data = data_lidar_process_all.shape[1] * data_lidar_process_all.shape[2]
    position_of_rx_as_thermomether = np.zeros([number_of_samples, x_dimension, y_dimension], dtype=np.int8)

    for i in range (number_of_samples):
        # number_of_samples = 0
        x_rx, y_rx, z_rx = 0, 0, 0
        pos_rx_in_each_sample = data_position_rx [i, :, :, :]
        x_rx, y_rx, z_rx = np.unravel_index (pos_rx_in_each_sample.argmax (), pos_rx_in_each_sample.shape)
        # position_of_rx_as_cube[i, 0:x_rx, 0:y_rx, 0:z_rx] = 1
        position_of_rx_as_thermomether[i, 0:x_rx, 0:y_rx] = 1

    position_of_rx_as_thermomether_vector = np.zeros([number_of_samples, dimension_of_coordenadas], dtype=np.int8)
    data_lidar_2D_with_rx_as_vector = np.zeros([samples, dimension_of_data*2], dtype=np.int8)
    for i in range(number_of_samples):
        rx_as_thermomether_vector = position_of_rx_as_thermomether[i, :, :].reshape(1, dimension_of_coordenadas)
        all_scerios_2D_as_vector = data_lidar_2D[i, :, :].reshape(1, dimension_of_data)
        data_lidar_2D_with_rx_as_vector[i] = np.concatenate((all_scerios_2D_as_vector, rx_as_thermomether_vector), axis=1)

    return data_lidar_2D_with_rx_as_vector

def plot_3D_scene(data_lidar_process_all, data_position_rx, data_position_tx, sample_for_plot):
    sample_for_plot = sample_for_plot
    #rx_as_cube = position_of_rx_as_cube [sample_for_plot, :, :, :]
    rx = data_position_rx [sample_for_plot, :, :, :]
    tx = data_position_tx [sample_for_plot, :, :, :]
    scenario_complet = data_lidar_process_all [sample_for_plot, :, :, :]
    fig = plt.figure()

    plt.rcParams.update ({'font.size': 14})
    #title = 'Scene ' + str (sample_for_plot) + ' of dataset ' + data_name
    #plt.title (title)

    #ax = fig.add_subplot (1, 2, 1, projection='3d')
    ax = plt.axes(projection='3d')

    #ax.voxels (rx_as_cube, alpha=0.12, edgecolor=None, shade=True, color='red')  # Voxel visualization
    #ax.voxels (scenario_complet, alpha=0.12, edgecolor=None, shade=True, color='red')  # Voxel visualization
    #ax.set_title ('Rx')
    #ax.set_xlabel ('x', labelpad=10)
    #ax.set_ylabel ('y', labelpad=10)
    #ax.set_zlabel ('z', labelpad=10)
    #plt.tight_layout ()

    objects = scenario_complet
    objects = np.array (objects, dtype=bool)
    rx = np.array (rx, dtype=bool)
    tx = np.array (tx, dtype=bool)

    voxelarray = objects | rx | tx

    # set the colors of each object
    colors = np.empty(voxelarray.shape, dtype=object)

    color_object = '#cccccc90'
    color_rx = 'red'
    color_tx = 'blue'

    colors[objects] = color_object
    colors[rx] = color_rx
    colors[tx] = color_tx

    # and plot everything
    # ax = plt.figure().add_subplot(projection='3d')

    # Set axes label
    ax.set_xlabel ('x', labelpad=10)
    ax.set_ylabel ('y', labelpad=10)
    ax.set_zlabel ('z', labelpad=10)

    # set predefine rotation
    # ax.view_init(elev=49, azim=115)

    #ax = fig.add_subplot (1, 2, 2, projection='3d')
    # ax.voxels(voxelarray, alpha=0.5, edgecolor=None, shade=True, antialiased=False,
    #         color='#cccccc90')  # Voxel visualization
    ax.voxels (voxelarray, facecolors=colors, edgecolor=None, antialiased=False)
    ax.set_title ('Full scenario scene ' + str(sample_for_plot))
    ax.set_xlabel ('x', labelpad=10)
    ax.set_ylabel ('y', labelpad=10)
    ax.set_zlabel ('z', labelpad=10)
    plt.tight_layout ()

    c1 = mpatches.Patch (color=color_object, label='Objects')
    c2 = mpatches.Patch (color=color_rx, label='Rx')
    c3 = mpatches.Patch (color=color_tx, label='Tx')

    ax.legend (handles=[c1, c2, c3], loc='center left', bbox_to_anchor=(-0.1, 0.9))


def pre_process_data_rx_like_cube(data_lidar_process_all, data_position_rx, data_position_tx, data_name, plot=True, sample_for_plot=0):

    x_dimension = len(data_position_rx[0, :, 0, 0])
    y_dimension = len(data_position_rx[0, 0, :, 0])
    z_dimension = len(data_position_rx[0, 0, 0, :])
    dimension_of_coordenadas = x_dimension * y_dimension * z_dimension
    number_of_samples = data_position_rx.shape[0]

    data = data_position_rx.copy()
    position_of_rx_as_cube = data * 0

    for i in range(number_of_samples):
        pos_rx_in_each_sample = data_position_rx[i, :, :, :]
        x_rx, y_rx, z_rx = np.unravel_index(pos_rx_in_each_sample.argmax(), pos_rx_in_each_sample.shape)
        position_of_rx_as_cube[i, 0:x_rx, 0:y_rx, 0:z_rx] = 1

    all_data = np.zeros([number_of_samples, dimension_of_coordenadas * 2], dtype=np.int8)

    for i in range(number_of_samples):
        position_of_rx_cube_as_vector = position_of_rx_as_cube[i, :, :, :].reshape(1, dimension_of_coordenadas)
        b = data_lidar_process_all[i, :, :, :].reshape(1, dimension_of_coordenadas)
        all_data[i] = np.concatenate((position_of_rx_cube_as_vector, b), axis=1)


    if plot:
        print("plotando o cenario...")
        # ------- PLOT RX E CENA COMPLETA
        sample_for_plot = sample_for_plot
        rx_as_cube = position_of_rx_as_cube[sample_for_plot, :, :, :]
        rx = data_position_rx[sample_for_plot, :, :, :]
        tx = data_position_tx[sample_for_plot,:,:,:]
        scenario_complet = data_lidar_process_all[sample_for_plot, :, :, :]
        fig = plt.figure()

        plt.rcParams.update ({'font.size': 14})
        title = 'Scene ' + str(sample_for_plot) + ' of dataset ' + data_name
        plt.title(title)

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.voxels(rx_as_cube, alpha=0.12, edgecolor=None, shade=True, color='red')  # Voxel visualization
        ax.set_title('Rx')
        ax.set_xlabel('x', labelpad=10)
        ax.set_ylabel('y', labelpad=10)
        ax.set_zlabel('z', labelpad=10)
        plt.tight_layout()

        objects = scenario_complet
        objects = np.array(objects, dtype=bool)
        rx = np.array(rx, dtype=bool)
        tx = np.array(tx, dtype=bool)

        voxelarray = objects | rx | tx

        # set the colors of each object
        colors = np.empty(voxelarray.shape, dtype=object)

        color_object = '#cccccc90'
        color_rx = 'red'
        color_tx = 'blue'

        colors[objects] = color_object
        colors[rx] = color_rx
        colors[tx] = color_tx

        # and plot everything
        #ax = plt.figure().add_subplot(projection='3d')

        # Set axes label
        ax.set_xlabel('x', labelpad=10)
        ax.set_ylabel('y', labelpad=10)
        ax.set_zlabel('z', labelpad=10)

        # set predefine rotation
        # ax.view_init(elev=49, azim=115)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        #ax.voxels(voxelarray, alpha=0.5, edgecolor=None, shade=True, antialiased=False,
         #         color='#cccccc90')  # Voxel visualization
        ax.voxels(voxelarray, facecolors=colors, edgecolor=None, antialiased=False)
        ax.set_title('Full scenario')
        ax.set_xlabel('x', labelpad=10)
        ax.set_ylabel('y', labelpad=10)
        ax.set_zlabel('z', labelpad=10)
        plt.tight_layout()

        c1 = mpatches.Patch(color=color_object, label='Objects')
        c2 = mpatches.Patch(color=color_rx, label='Rx')
        c3 = mpatches.Patch(color=color_tx, label='Tx')

        ax.legend(handles=[c1, c2, c3], loc='center left', bbox_to_anchor=(-0.1, 0.9))

        #plot('plotei')

    return all_data
def read_pre_processed_data_rx_like_cube():
    data_path = "../data/lidar/s008/all_data_lidar_+_rx_like_cube_train.npz"
    label_cache_file = np.load(data_path)
    key = list(label_cache_file.keys())
    data_lidar_train = label_cache_file[key[0]]

    data_path = "../data/lidar/s009/all_data_lidar_+_rx_like_cube_test.npz"
    label_cache_file = np.load(data_path)
    key = list(label_cache_file.keys())
    data_lidar_test = label_cache_file[key[0]]

    return data_lidar_train, data_lidar_test
def read_lidar_data_of_s008():
    data_path = "../data/lidar/s008/lidar_train_raymobtime.npz"
    label_cache_file = np.load (data_path)
    key = list (label_cache_file.keys ())
    data_lidar_train = label_cache_file [key [0]]

    data_path = "../data/lidar/s008/lidar_validation_raymobtime.npz"
    label_cache_file = np.load (data_path)
    key = list (label_cache_file.keys ())
    data_lidar_val = label_cache_file [key [0]]

    return data_lidar_train, data_lidar_val

def read_lidar_data_of_s009():
    data_path = "../data/lidar/s009/lidar_test_raymobtime.npz"
    label_cache_file = np.load (data_path)
    key = list(label_cache_file.keys())
    data_lidar_test = label_cache_file[key[0]]

    return data_lidar_test

def process_lidar_2D_dilation(ite):
    path = '../data/lidar/s008/lidar_train_raymobtime.npz'
    data_lidar_process_all, data_position_rx, data_position_tx = read_data (path)
    _, data_lidar_2D = pre_process_data_lidar_2D(data_lidar_process_all, data_position_rx, data_position_tx)
    data_lidar_2D_dilated = pre_process_lidar_2D_dilation(data_lidar_2D, ite)

    path = '../data/lidar/s008/lidar_validation_raymobtime.npz'
    data_lidar_process_all_val, data_position_rx_val, data_position_tx_val = read_data (path)
    _, data_lidar_2D_val = pre_process_data_lidar_2D(data_lidar_process_all_val, data_position_rx_val, data_position_tx_val)
    data_lidar_2D_dilated_val = pre_process_lidar_2D_dilation(data_lidar_2D_val, ite)

    data_lidar_2D_dilated_train = np.concatenate((data_lidar_2D_dilated, data_lidar_2D_dilated_val), axis=0)

    path = '../data/lidar/s009/lidar_test_raymobtime.npz'
    data_lidar_process_all_test, data_position_rx_test, data_position_tx_test = read_data (path)
    _, data_lidar_2D_test = pre_process_data_lidar_2D(data_lidar_process_all_test, data_position_rx_test, data_position_tx_test)
    data_lidar_2D_dilated_test = pre_process_lidar_2D_dilation(data_lidar_2D_test, ite)

    return data_lidar_2D_dilated_train, data_lidar_2D_dilated_test
def pre_process_lidar_2D_dilation(data_lidar_2D, ite):#data_lidar_process_all, data_position_rx, data_position_tx, plot=True, sample_for_plot=0):
    #path = '../data/lidar/s008/lidar_train_raymobtime.npz'
    #data_lidar_process_all, data_position_rx, data_position_tx = read_data (path)
    #_, data_lidar_2D = pre_process_data_lidar_2D(data_lidar_process_all, data_position_rx, data_position_tx)
    #plot_2D_scene(data_lidar_2D, sample_for_plot=0)

    all_dilated_obj = data_lidar_2D.copy () * 0
    number_of_samples = data_lidar_2D.shape[0]
    for i in range (number_of_samples):
        pos_obj_in_each_sample = data_lidar_2D[i, :, :]
        dilated_obj_per_sample = ndimage.binary_dilation(pos_obj_in_each_sample, iterations=ite).astype(pos_obj_in_each_sample.dtype)
        all_dilated_obj[i, :, :] = dilated_obj_per_sample

    dimensao = data_lidar_2D.shape[1] * data_lidar_2D.shape[2]
    all_data = np.zeros([number_of_samples, dimensao], dtype=np.int8)
    for i in range(number_of_samples):
        all_dilated_obj_as_vector = all_dilated_obj[i, :, :].reshape(1, dimensao)
        all_data[i] = all_dilated_obj_as_vector

    plot_2D_scene(all_dilated_obj, sample_for_plot=0)


    return all_data

def pre_process_lidar_3D_rx_therm_2D():
    data_path = "../data/lidar/s008/lidar_train_raymobtime.npz"
    data_lidar_3D_process_all, data_position_rx, data_position_tx = read_data(data_path)

    data_path = "../data/lidar/s008/lidar_validation_raymobtime.npz"
    data_lidar_3D_process_all_val, data_position_rx_val, data_position_tx_val = read_data(data_path)

    data_lidar_3D_train = np.concatenate((data_lidar_3D_process_all, data_lidar_3D_process_all_val), axis=0)
    #plot_3D_scene(data_lidar_3D_train, data_position_rx, data_position_tx, sample_for_plot=0)

    data_path = "../data/lidar/s009/lidar_test_raymobtime.npz"
    data_lidar_3D_test, data_position_rx_test, data_position_tx_test = read_data(data_path)

    dimension_of_data = data_lidar_3D_train.shape[1] * data_lidar_3D_train.shape[2] * data_lidar_3D_train.shape[3]
    samples_train = data_lidar_3D_train.shape[0]
    data_lidar_3D_as_vector_train = np.zeros([samples_train, dimension_of_data], dtype=np.int8)

    samples_test = data_lidar_3D_test.shape[0]
    data_lidar_3D_as_vector_test = np.zeros([samples_test, dimension_of_data], dtype=np.int8)

    # Reshape the data to be used in the model
    for i in range(samples_train):
        data_lidar_3D_as_vector_train[i] = data_lidar_3D_train[i, :, :].reshape(1, dimension_of_data)

    for i in range(samples_test):
        data_lidar_3D_as_vector_test[i] = data_lidar_3D_test[i, :, :].reshape(1, dimension_of_data)

    position_of_rx_2D_as_thermomether_train, position_of_rx_2D_as_thermomether_test = process_data_rx_2D_like_thermomether()

    data_lidar_3D_rx_therm_2D_train = np.concatenate((data_lidar_3D_as_vector_train, position_of_rx_2D_as_thermomether_train), axis=1)
    data_lidar_3D_rx_therm_2D_test = np.concatenate((data_lidar_3D_as_vector_test, position_of_rx_2D_as_thermomether_test), axis=1)

    return data_lidar_3D_rx_therm_2D_train, data_lidar_3D_rx_therm_2D_test


#-------------------------------------------
def data_lidar_2D_binary_without_variance():
    data_path = "../data/lidar/s008/lidar_train_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = read_data(data_path)
    pre_process_data_lidar_2D_vector_train, pre_process_data_lidar_2D_train = pre_process_data_lidar_2D(
        data_lidar_process_all, data_position_rx, data_position_tx)

    data_path = "../data/lidar/s008/lidar_validation_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = read_data(data_path)
    pre_process_data_lidar_2D_vector_val, pre_process_data_lidar_2D_val = pre_process_data_lidar_2D(
        data_lidar_process_all, data_position_rx, data_position_tx)

    data_lidar_2D_train = np.concatenate((pre_process_data_lidar_2D_train, pre_process_data_lidar_2D_val), axis=0)
    data_lidar_2D_vector_train = np.concatenate((pre_process_data_lidar_2D_vector_train,
                                                        pre_process_data_lidar_2D_vector_val), axis=0)

    data_path = "../data/lidar/s009/lidar_test_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = read_data(data_path)
    pre_process_data_lidar_2D_vector_test, pre_process_data_lidar_2D_test = pre_process_data_lidar_2D(
        data_lidar_process_all, data_position_rx, data_position_tx)

    data_lidar_2D_vector_test = pre_process_data_lidar_2D_vector_test

    th = 0
    selector = VarianceThreshold (threshold=th)
    vt_train = selector.fit(data_lidar_2D_vector_train)
    data_lidar_2D_train_without_var = data_lidar_2D_vector_train[:, vt_train.variances_ > th]

    vt_test = selector.fit(data_lidar_2D_vector_test)
    data_lidar_2D_test_without_var = data_lidar_2D_vector_test[:, vt_test.variances_ > th]

    return data_lidar_2D_train_without_var, data_lidar_2D_test_without_var



def data_lidar_3D():
    data_path = "../data/lidar/s008/lidar_train_raymobtime.npz"
    data_lidar_all_train, data_position_rx, data_position_tx = read_data (data_path)

    data_path = "../data/lidar/s008/lidar_validation_raymobtime.npz"
    data_lidar_all_val, data_position_rx_val, data_position_tx_val = read_data (data_path)

    data_path = "../data/lidar/s009/lidar_test_raymobtime.npz"
    data_lidar_all_test, data_position_rx_test, data_position_tx_test = read_data (data_path)

    dimension_of_data = data_lidar_all_train.shape [1] * data_lidar_all_train.shape [2] * data_lidar_all_train.shape [3]

    # Reshape the data to be used in the model
    samples_train = data_lidar_all_train.shape [0]
    data_lidar_3D_train_as_vector = (np.zeros ([samples_train, dimension_of_data], dtype=np.int8))
    for i in range(samples_train):
        data_lidar_3D_train_as_vector[i] = data_lidar_all_train [i, :, :, :].reshape(1, dimension_of_data)

    samples_val = data_lidar_all_val.shape [0]
    data_lidar_3D_val_as_vector = (np.zeros([samples_val, dimension_of_data], dtype=np.int8))
    for i in range(samples_val):
        data_lidar_3D_val_as_vector[i] = data_lidar_all_val[i, :, :, :].reshape(1, dimension_of_data)

    data_train = np.concatenate((data_lidar_3D_train_as_vector, data_lidar_3D_val_as_vector), axis=0)
    data_test = data_lidar_all_test


    return data_train, data_test

def data_lidar_3D_binary_without_variance():
    data_path = "../data/lidar/s008/lidar_train_raymobtime.npz"
    data_lidar_all_train, data_position_rx, data_position_tx = read_data(data_path)

    data_path = "../data/lidar/s008/lidar_validation_raymobtime.npz"
    data_lidar_all_val, data_position_rx_val, data_position_tx_val = read_data (data_path)

    data_path = "../data/lidar/s009/lidar_test_raymobtime.npz"
    data_lidar_all_test, data_position_rx_test, data_position_tx_test = read_data (data_path)

    dimension_of_data = data_lidar_all_train.shape[1] * data_lidar_all_train.shape[2]* data_lidar_all_train.shape[3]

    # Reshape the data to be used in the model
    samples_train = data_lidar_all_train.shape [0]
    data_lidar_3D_train_as_vector = (np.zeros ([samples_train, dimension_of_data], dtype=np.int8))
    for i in range(samples_train):
        data_lidar_3D_train_as_vector[i] = data_lidar_all_train[i, :, :, :].reshape (1, dimension_of_data)

    samples_val = data_lidar_all_val.shape[0]
    data_lidar_3D_val_as_vector = (np.zeros ([samples_val, dimension_of_data], dtype=np.int8))
    for i in range(samples_val):
        data_lidar_3D_val_as_vector[i] = data_lidar_all_val[i, :, :, :].reshape (1, dimension_of_data)

    samples_test = data_lidar_all_test.shape[0]
    data_lidar_3D_test_as_vector = (np.zeros ([samples_test, dimension_of_data], dtype=np.int8))
    for i in range(samples_test):
        data_lidar_3D_test_as_vector[i] = data_lidar_all_test[i, :, :, :].reshape (1, dimension_of_data)

    data_lidar_3D_vector_train = np.concatenate((data_lidar_3D_train_as_vector,
                                                        data_lidar_3D_val_as_vector), axis=0)

    th = 0.1
    selector = VarianceThreshold(threshold=th)
    print("******** PRE PROCESSING DATA ********")
    print('Eliminando variâncias menores que ', th)
    print('\tMax variancias')
    print('\tTrain\t\tTest')
    print('\t',round(np.max(np.var(data_lidar_3D_vector_train, axis=0)),2),'\t',np.max(np.var(data_lidar_3D_test_as_vector, axis=0)))

    variance_threshold = selector.fit(data_lidar_3D_vector_train)
    #vt_test = selector.fit(data_lidar_3D_test_as_vector)
    data_lidar_3D_train_without_var = data_lidar_3D_vector_train[:, variance_threshold.variances_ > th]
    data_lidar_3D_test_without_var = data_lidar_3D_test_as_vector[:, variance_threshold.variances_ > th]

    print('\tNew size of Dataset')
    print('\tTrain',data_lidar_3D_train_without_var.shape,'\tTest',data_lidar_3D_test_without_var.shape)

    return data_lidar_3D_train_without_var, data_lidar_3D_test_without_var



def process_data_lidar_into_2D_matrix():
    data_path = "../data/lidar/s008/lidar_train_raymobtime.npz"
    data_lidar_process_all, data_position_rx, data_position_tx = read_data (data_path)
    data_lidar_2D_as_vector_train, data_lidar_2D_matrix_train = pre_process_data_lidar_into_2D(data_lidar_process_all, data_position_rx, data_position_tx)

    data_path = "../data/lidar/s008/lidar_validation_raymobtime.npz"
    data_lidar_process_all_val, data_position_rx_val, data_position_tx_val = read_data (data_path)
    data_lidar_2D_as_vector_val, data_lidar_2D_matrix_val = pre_process_data_lidar_into_2D(data_lidar_process_all_val, data_position_rx_val, data_position_tx_val)

    data_lidar_2D_vector_train = np.concatenate((data_lidar_2D_as_vector_train, data_lidar_2D_as_vector_val), axis=0)
    data_lidar_2D_matrix_train = np.concatenate((data_lidar_2D_matrix_train, data_lidar_2D_matrix_val), axis=0)

    data_path = "../data/lidar/s009/lidar_test_raymobtime.npz"
    data_lidar_process_all_test, data_position_rx_test, data_position_tx_test = read_data (data_path)
    data_lidar_2D_vector_test, data_lidar_2D_matrix_test = pre_process_data_lidar_into_2D(data_lidar_process_all_test, data_position_rx_test, data_position_tx_test)

    return data_lidar_2D_vector_train, data_lidar_2D_vector_test, data_lidar_2D_matrix_train, data_lidar_2D_matrix_test

def pre_process_data_lidar_into_2D(data_lidar_process_all, data_position_rx, data_position_tx, plot=True, sample_for_plot=0):
    ''' criated a 2D matrix from the 3D matrix of lidar data preserving the high information '''

    plot = False
    sample_for_plot = 3

    samples = data_lidar_process_all.shape [0]
    data_lidar_2D = np.zeros([samples, 20, 200], dtype=np.int8)

    for sample in range(len(data_lidar_process_all)):
        test_matriz = np.zeros([20, 200], dtype=np.int8)
        for var in range(0, 20):
            for y in range(0, 10):
                test_matriz[var, :] += data_lidar_process_all [sample, var, :, y]
            #test_matriz [var, :] = np.where (test_matriz [var, :] > 0, 1, 0)
        data_lidar_2D[sample] = test_matriz

    dimension_of_data = data_lidar_process_all.shape [1] * data_lidar_process_all.shape [2]
    data_lidar_2D_as_vector = np.zeros([samples, dimension_of_data], dtype=np.int8)

    # Reshape the data to be used in the model
    for i in range(samples):
        data_lidar_2D_as_vector[i] = data_lidar_2D[i, :, :].reshape (1, dimension_of_data)
        # b = data_lidar_process_all [i, :, :, :].reshape (1, dimension_of_coordenadas)
        # all_data [i] = np.concatenate ((position_of_rx_cube_as_vector, b), axis=1)

    if plot:
        #plot_3D_scene (data_lidar_process_all, data_position_rx, data_position_tx, sample_for_plot=sample_for_plot)
        plot_2D_scene (data_lidar_2D, sample_for_plot=sample_for_plot)
        print('entro no plot')

    return data_lidar_2D_as_vector, data_lidar_2D







#data_lidar_3D_binary_without_variance()