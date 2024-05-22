from imblearn.over_sampling import SMOTE
import pre_process_lidar
import analyse_s009, analyse_s008
import numpy as np


def oversampling():
    # ------- Get Beams
    index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline()
    index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline()

    label_train = np.concatenate((index_beams_train, index_beam_validation), axis=0)
    label_test = index_beams_test

    data_lidar_in_vector_train, data_lidar_in_vector_test, data_lidar_2D_matrix_train, data_lidar_2D_matrix_test = pre_process_lidar.process_data_lidar_into_2D_matrix()
    X = data_lidar_in_vector_train
    y = label_train

    smote = SMOTE()

    X_resampled, y_resampled = smote.fit_resample(X, y)

    a=0

oversampling()