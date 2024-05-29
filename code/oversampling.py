from imblearn.over_sampling import SMOTE
import pre_process_lidar
import analyse_s009, analyse_s008
import numpy as np
from imblearn.over_sampling import ADASYN, RandomOverSampler
import matplotlib.pyplot as plt


def oversampling(input_type):
    #input_type = 'lidar_2D'
    #input_type = 'lidar_3D'

    # ------- Get Beams
    index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline()
    index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline()

    label_train = np.concatenate((index_beams_train, index_beam_validation), axis=0)
    label_test = index_beams_test

    if input_type == 'lidar_2D':
        #data_lidar_in_vector_train, data_lidar_in_vector_test, data_lidar_2D_matrix_train, data_lidar_2D_matrix_test = pre_process_lidar.process_data_lidar_into_2D_matrix()
        data_lidar_2D_train, data_lidar_2D_test = pre_process_lidar.read_pre_processed_data_lidar_2D ()
        data_train  = data_lidar_2D_train
        data_test = data_lidar_2D_test

    elif input_type == 'lidar_3D':
        data_train, data_test = pre_process_lidar.data_lidar_3D()



    X = data_train

    y = np.zeros(len(label_train))
    for i in range(len(label_train)):
        y[i] = label_train[i].astype(int)

    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    x_train = X_resampled
    y_train = y_resampled.astype(int).astype(str)

    x_test = data_test
    y_test = label_test

    plt.figure()
    plt.hist(y_resampled,  alpha=0.5, color='r', label='resampled beams')
    plt.hist (y, alpha=0.5, color='b', label='real beams')
    plt.xlabel('Beams')
    plt.title('Comparision of orignal data with resampled data \n usign RandomOverSampler')
    plt.legend()
    plt.show()
    #smote = SMOTE()
    #ada = ADASYN ()
    #X_resampled, y_resampled = ada.fit_resample(X, y)


    #X_resampled, y_resampled = smote.fit_resample(X, y)

    return x_train, y_train, x_test, y_test

#oversampling()