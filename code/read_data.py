import pandas as pd
import numpy as np




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

def read_data_s009():
    filename = '../data/coord/CoordVehiclesRxPerScene_s009.csv'
    data = pd.read_csv (filename)
    valid_data = data [data ['Val'] == 'V']

    path = '../data/beams_output/beam_output_baseline_raymobtime_s009/beams_output_test.npz'
    tx_index, rx_index, best_beam_index = read_beams_raymobtime (num_antennas_rx=8, path_of_data=path)

    valid_data.insert (6, 'index_beams', best_beam_index)
    valid_data.insert (7, 'tx_index', tx_index)
    valid_data.insert (8, 'rx_index', rx_index)

    data_LOS = valid_data[valid_data['LOS'] == 'LOS=1']
    data_NLOS = valid_data[valid_data['LOS'] == 'LOS=0']

    return data_LOS, data_NLOS, valid_data

def read_data_s008():
    filename = '../data/coord/CoordVehiclesRxPerScene_s008.csv'
    limit_ep_train = 1564
    data = pd.read_csv (filename)
    valid_data = data[data['Val'] == 'V']
    #train_data = valid_data [valid_data ['EpisodeID'] <= limit_ep_train]

    path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_train.npz'
    tx_index_train, rx_index_train, best_beam_index_train = read_beams_raymobtime (num_antennas_rx=8, path_of_data=path)

    path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_validation.npz'
    tx_index_val, rx_index_val, best_beam_index_val = read_beams_raymobtime (num_antennas_rx=8, path_of_data=path)

    tx_index = np.concatenate ((tx_index_train, tx_index_val), axis=0)
    rx_index = np.concatenate ((rx_index_train, rx_index_val), axis=0)
    best_beam_index = np.concatenate ((best_beam_index_train, best_beam_index_val), axis=0)

    valid_data.insert (6, 'index_beams', best_beam_index)
    valid_data.insert (7, 'tx_index', tx_index)
    valid_data.insert (8, 'rx_index', rx_index)

    data_LOS = valid_data [valid_data ['LOS'] == 'LOS=1']
    NLOS = valid_data [valid_data ['LOS'] == 'LOS=0']

    return data_LOS, NLOS, valid_data