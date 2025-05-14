import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def getBeamOutput(output_file):
    thresholdBelowMax = 6
    #print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]

    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y = beamsLogScale(y, thresholdBelowMax)

    return y,num_classes
def beamsLogScale(y, thresholdBelowMax):
    y_shape = y.shape  # shape is (#,256)

    for i in range (0, y_shape [0]):
        thisOutputs = y [i, :]
        logOut = 20 * np.log10 (thisOutputs + 1e-30)
        minValue = np.amax (logOut) - thresholdBelowMax
        zeroedValueIndices = logOut < minValue
        thisOutputs [zeroedValueIndices] = 0
        thisOutputs = thisOutputs / sum (thisOutputs)
        y [i, :] = thisOutputs
    return y
def open_npz(path):
    #data = np.load(path)[key]
    #return data
    cache = np.load(path, allow_pickle=True)
    keys = list(cache.keys())
    data = cache[keys[0]]

    return data

def get_index_beams():
    data_folder = '../../data/'
    train_data_folder = 'beams_output/beam_output_baseline_raymobtime_s008/'
    file_train = 'beams_output_train.npz'
    file_val = 'beams_output_validation.npz'

    output_train_file = data_folder + train_data_folder + file_train
    output_validation_file = data_folder + train_data_folder + file_val
    y_train, num_classes = getBeamOutput (output_train_file)
    y_validation, _ = getBeamOutput (output_validation_file)

    test_data_folder = 'beams_output/beam_output_baseline_raymobtime_s009/'
    file_test = 'beams_output_test.npz'
    output_test_file = data_folder + test_data_folder + file_test
    y_test, _ = getBeamOutput (output_test_file)

    return y_train, y_validation, y_test, num_classes
def read_all_data():

    filename = '../../data/coord/CoordVehiclesRxPerScene_s008.csv'
    all_csv_data = pd.read_csv (filename)
    valid_data = all_csv_data[all_csv_data['Val'] == 'V']
    limit_ep_train = 1564

    train_data = valid_data[valid_data['EpisodeID'] <= limit_ep_train]

    coord_for_train = np.zeros((len(train_data), 2))
    coord_for_train[:, 0] = train_data['x']
    coord_for_train[:, 1] = train_data['y']
    coord_train = normalize(coord_for_train, axis=1, norm='l1')

    validation_data = valid_data[valid_data['EpisodeID'] > limit_ep_train]
    coord_for_validation = np.zeros((len(validation_data), 2))
    coord_for_validation[:, 0] = validation_data['x']
    coord_for_validation[:, 1] = validation_data['y']
    coord_validation = normalize(coord_for_validation, axis=1, norm='l1')

    y_train, y_validation, y_test, num_classes = get_index_beams ()

    data_folder = '../../data/'
    lidar_train = open_npz(data_folder + 'lidar/s008/lidar_train_raymobtime.npz') / 2
    lidar_validation = open_npz (data_folder + 'lidar/s008/lidar_validation_raymobtime.npz') / 2
    lidar_test = open_npz (data_folder + 'lidar/s009/lidar_test_raymobtime.npz') / 2

    lidar_train_reshaped = lidar_train.reshape(9234, -1)
    lidar_validation_reshaped = lidar_validation.reshape(1960, -1)
    lidar_test_reshaped = lidar_test.reshape(9638, -1)

    data_for_train = pd.DataFrame({"Episode": train_data['EpisodeID'],
                                   "coord": coord_train.tolist(),
                                   "x": coord_train[:, 0],
                                   "y": coord_train[:, 1],
                                   "LOS": train_data['LOS'],
                                   "index_beams": y_train.tolist(),
                                   "lidar": lidar_train_reshaped.tolist()})

    data_for_validation = pd.DataFrame({"Episode": validation_data['EpisodeID'],
                                        "coord": coord_validation.tolist(),
                                        "x": coord_validation[:, 0],
                                        "y": coord_validation[:, 1],
                                        "LOS": validation_data['LOS'],})
    data_for_validation["lidar"] = lidar_validation_reshaped.tolist()
    data_for_validation["index_beams"] = y_validation.tolist()

    filename = '../../data/coord/CoordVehiclesRxPerScene_s009.csv'
    all_csv_data = pd.read_csv(filename)
    valid_data = all_csv_data[all_csv_data['Val'] == 'V']

    coord_for_test = np.zeros((len(valid_data), 2))
    coord_for_test[:, 0] = valid_data['x']
    coord_for_test[:, 1] = valid_data['y']
    coord_test = normalize(coord_for_test, axis=1, norm='l1')

    data_for_test = pd.DataFrame({"Episode": valid_data['EpisodeID'],
                                    "coord": coord_test.tolist(),
                                    "x": coord_test[:, 0],
                                    "y": coord_test[:, 1],
                                    "LOS": valid_data['LOS'],})
    data_for_test["lidar"] = lidar_test_reshaped.tolist()
    data_for_test["index_beams"] = y_test.tolist()

    return data_for_train, data_for_validation, data_for_test, num_classes
