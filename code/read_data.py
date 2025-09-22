import pandas as pd
import numpy as np
import pre_process_lidar
import pre_process_coord



def read_beams_raymobtime(num_antennas_rx, path_of_data):

    data_path = path_of_data
    beams = np.load(data_path)['output_classification']

    best_beam_index = []
    for sample in range(beams.shape[0]):
        best_beam_index.append(np.argmax(beams[sample, :]))

    beam_index_rx = np.array(best_beam_index)

    tx_index = np.zeros((beams.shape[0]), dtype=int)
    rx_index = np.zeros((beams.shape[0]), dtype=int)

    for sample in range(len(beam_index_rx)):
        index_tx = best_beam_index[sample] // int(num_antennas_rx)
        index_rx = best_beam_index[sample] % int(num_antennas_rx)
        tx_index[sample] = index_tx
        rx_index[sample] = index_rx

    return tx_index, rx_index, best_beam_index

def read_data_s009(scale_to_coord):
    filename = '../data/coord/CoordVehiclesRxPerScene_s009.csv'
    data = pd.read_csv (filename)
    valid_data = data [data ['Val'] == 'V']

    path = '../data/beams_output/beam_output_baseline_raymobtime_s009/beams_output_test.npz'
    tx_index, rx_index, best_beam_index = read_beams_raymobtime (num_antennas_rx=8, path_of_data=path)

    data_lidar_2D_with_rx_termomether_s008, data_lidar_2D_with_rx_termomether_s009 = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer ()
    th_variance = 0.1
    data_lidar_2D_without_variance_s008, data_lidar_2D_without_variance_s009 = pre_process_lidar.data_lidar_2D_binary_without_variance (
        data_lidar_2D_with_rx_termomether_s008, data_lidar_2D_with_rx_termomether_s009, th_variance)

    encondig_coord = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s009(escala=scale_to_coord)

    lidar_coord = np.concatenate((data_lidar_2D_without_variance_s009, encondig_coord), axis=1)
    valid_data.insert (10, 'index_beams', best_beam_index)
    valid_data.insert (11, 'tx_index', tx_index)
    valid_data.insert (12, 'rx_index', rx_index)
    valid_data.insert (13, 'lidar', data_lidar_2D_without_variance_s009.tolist())
    valid_data.insert (14, 'enconding_coord', encondig_coord.tolist())
    valid_data.insert (15, 'lidar_coord', lidar_coord.tolist())

    data_LOS = valid_data[valid_data['LOS'] == 'LOS=1']
    data_NLOS = valid_data[valid_data['LOS'] == 'LOS=0']

    return data_LOS, data_NLOS, valid_data

def read_data_s008_train_and_validation(scale_to_coord=8):
    filename = '../data/coord/CoordVehiclesRxPerScene_s008.csv'
    limit_ep_train = 1564
    data = pd.read_csv (filename)
    valid_data = data [data ['Val'] == 'V']
    # train_data = valid_data [valid_data ['EpisodeID'] <= limit_ep_train]

    path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_train.npz'
    tx_index_train, rx_index_train, best_beam_index_train = read_beams_raymobtime (num_antennas_rx=8, path_of_data=path)

    path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_validation.npz'
    tx_index_val, rx_index_val, best_beam_index_val = read_beams_raymobtime (num_antennas_rx=8, path_of_data=path)





def read_data_s008(scale_to_coord):#=8):
    filename = '../data/coord/CoordVehiclesRxPerScene_s008.csv'
    limit_ep_train = 1564
    data = pd.read_csv (filename)
    valid_data = data[data['Val'] == 'V']
    #train_data = valid_data [valid_data ['EpisodeID'] <= limit_ep_train]

    path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_train.npz'
    tx_index_train, rx_index_train, best_beam_index_train = read_beams_raymobtime(num_antennas_rx=8, path_of_data=path)

    path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_validation.npz'
    tx_index_val, rx_index_val, best_beam_index_val = read_beams_raymobtime(num_antennas_rx=8, path_of_data=path)

    data_lidar_2D_with_rx_termomether_s008, data_lidar_2D_with_rx_termomether_s009 = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer ()
    th_variance = 0.1
    data_lidar_2D_without_variance_s008, data_lidar_2D_without_variance_s009 = pre_process_lidar.data_lidar_2D_binary_without_variance (
        data_lidar_2D_with_rx_termomether_s008, data_lidar_2D_with_rx_termomether_s009, th_variance)

    encondign_coord_train, encondign_coord_validation, encondig_coord = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s008(escala=scale_to_coord)

    lidar_coord = np.concatenate((data_lidar_2D_without_variance_s008, encondig_coord), axis=1)

    tx_index = np.concatenate ((tx_index_train, tx_index_val), axis=0)
    rx_index = np.concatenate ((rx_index_train, rx_index_val), axis=0)
    best_beam_index = np.concatenate ((best_beam_index_train, best_beam_index_val), axis=0)

    valid_data.insert (10, 'index_beams', best_beam_index)
    valid_data.insert (11, 'tx_index', tx_index)
    valid_data.insert (12, 'rx_index', rx_index)
    valid_data.insert (13, 'lidar', data_lidar_2D_without_variance_s008.tolist())
    valid_data.insert (14, 'enconding_coord', encondig_coord.tolist())
    valid_data.insert (15, 'lidar_coord', lidar_coord.tolist())

    data_LOS = valid_data [valid_data ['LOS'] == 'LOS=1']
    NLOS = valid_data [valid_data ['LOS'] == 'LOS=0']

    return data_LOS, NLOS, valid_data