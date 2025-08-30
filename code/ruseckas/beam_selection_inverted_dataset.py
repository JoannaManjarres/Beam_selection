import sys
import pandas as pd
import numpy as np

sys.path.append ("../")

# Agora é possível importar o arquivo como um módulo
import tools as tls
import online_learning_ruseckas as olb
import prepare_for_online_learning as prepare

def beam_selection_with_inverted_dataset(input_type='lidar_coord'):
    data_for_train, data_for_validation, data_s009, num_classes = prepare.read_all_data ()
    test_s008 = pd.concat([data_for_train, data_for_validation], axis=0)

    size_for_train = int(len(data_s009)*0.8)
    train_s009 = data_s009 [:size_for_train]
    val_s009 = data_s009 [size_for_train:]

    label_test = np.array(test_s008['index_beams'].tolist())
    label_train = np.array(train_s009['index_beams'].tolist())
    label_validation = np.array(val_s009['index_beams'].tolist())

    BATCH_SIZE = 32


    if input_type == 'coord':
        input_train = np.array(train_s009['coord'].tolist())
        input_validation = np.array(val_s009['coord'].tolist())
        input_test = np.array(test_s008['coord'].tolist())

        train_generator = prepare.DataGenerator_coord (input_train,
                                                       label_train,
                                                       BATCH_SIZE, shuffle=True)
        val_generator = prepare.DataGenerator_coord (input_validation,
                                                     label_validation,
                                                     BATCH_SIZE, shuffle=True)
        input_train_shape = input_train.shape[1:]

    if input_type == 'lidar':
        lidar_train = np.array(train_s009['lidar'].tolist())
        lidar_val = np.array(val_s009['lidar'].tolist())
        lidar_test = np.array(test_s008['lidar'].tolist())

        input_train = lidar_train.reshape (lidar_train.shape[0], 20, 200, 10)
        input_validation = lidar_val.reshape (lidar_val.shape[0], 20, 200, 10)
        input_test = lidar_test.reshape (lidar_test.shape[0], 20, 200, 10)

        train_generator = prepare.DataGenerator_lidar (input_train, label_train, BATCH_SIZE, shuffle=True)
        val_generator = prepare.DataGenerator_lidar (input_validation, label_validation, BATCH_SIZE, shuffle=True)
        input_train_shape = input_train.shape[1:]

    if input_type == 'lidar_coord':
        coord_train = np.array (train_s009 ['coord'].tolist ())
        coord_val = np.array (val_s009 ['coord'].tolist ())
        coord_test = np.array (test_s008 ['coord'].tolist ())

        lidar_train = np.array (train_s009 ['lidar'].tolist ())
        lidar_val = np.array (val_s009 ['lidar'].tolist ())
        lidar_test = np.array (test_s008 ['lidar'].tolist ())

        lidar_train = lidar_train.reshape (lidar_train.shape [0], 20, 200, 10)
        input_train = [lidar_train.reshape (lidar_train.shape [0], 20, 200, 10), coord_train]

        lidar_val = lidar_val.reshape (lidar_val.shape [0], 20, 200, 10)
        input_val = [lidar_val.reshape (lidar_val.shape [0], 20, 200, 10), coord_val]

        lidar_test = lidar_test.reshape (lidar_test.shape [0], 20, 200, 10)
        input_test = [lidar_test.reshape (lidar_test.shape [0], 20, 200, 10), coord_test]

        X_lidar_train = input_train [0]
        X_coord_train = input_train [1]
        train_generator = prepare.DataGenerator_both (X_lidar_train,
                                                      X_coord_train,
                                                      label_train, BATCH_SIZE, shuffle=True)
        X_lidar_val = input_val [0]
        X_coord_val = input_val [1]
        val_generator = prepare.DataGenerator_both (X_lidar_val,
                                                    X_coord_val,
                                                    label_validation, BATCH_SIZE)

        input_train_shape = [X_lidar_train.shape [1:], X_coord_train.shape [1:]]

    df_results_top_k, all_index_predict_order = olb.beam_selection_ruseckas(type_of_input = input_type,
                                                                            train_generator=train_generator,
                                                                            val_generator=val_generator,
                                                                            input_test=input_test,
                                                                            label_test=label_test,
                                                                            num_classes=num_classes,
                                                                            input_train_shape=input_train_shape,
                                                                            episode=0, train=True)

    path_result = ('../../results/inverter_dataset/score/Ruseckas/' + input_type + '/ALL/')
    all_index_predict_order.to_csv (path_result + 'index_predict_'+input_type+'.csv', index=False)
    df_results_top_k.to_csv (path_result + 'accuracy_'+input_type+'.csv', index=False)

    a=0


beam_selection_with_inverted_dataset()