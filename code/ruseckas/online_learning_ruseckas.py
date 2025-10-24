# Imports

import argparse

import numpy as np
import pandas as pd
import prepare_for_online_learning as prepare
import sys
sys.path.append ("../")

# Agora é possível importar o arquivo como um módulo
import tools as tls

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'





def sliding_prepare_label_for_trainning(label_for_train):
    new_form_for_label = np.array(label_for_train)
    size_of_label = new_form_for_label.shape
    size_of_train = int(size_of_label[0] * 0.8)
    label_train = new_form_for_label[:size_of_train]
    label_validation = new_form_for_label[size_of_train:]
    return label_train, label_validation
def sliding_prepare_for_trainning(input_for_train , type_input):

    if type_input == 'lidar_coord':
        lidar_data = np.array(input_for_train[0])
        coord_data = np.array(input_for_train[1])
        size_of_input = lidar_data.shape
        size_of_train = int(size_of_input[0] * 0.8)

        train_coord = coord_data[:size_of_train]
        train_lidar = lidar_data[:size_of_train]
        train = [train_lidar.reshape (train_lidar.shape [0], 20, 200, 10), train_coord]

        val_coord = coord_data[size_of_train:]
        val_lidar = lidar_data[size_of_train:]
        val = [val_lidar.reshape(val_lidar.shape[0], 20, 200, 10), val_coord]
    elif type_input == 'coord':
        a = np.array(input_for_train)
        size_of_input = a.shape
        size_of_train = int(size_of_input [0] * 0.8)
        train = a[:size_of_train]
        val = a [size_of_train:]
    elif type_input == 'lidar':
        a = np.array(input_for_train)
        size_of_input = a.shape
        size_of_train = int(size_of_input[0] * 0.8)
        train_data = a[:size_of_train]
        val_data = a[size_of_train:]
        train = train_data.reshape (train_data.shape [0], 20, 200, 10)
        val = val_data.reshape (val_data.shape [0], 20, 200, 10)

    return train, val

def beam_selection_ruseckas(type_of_input,
                            train_generator, val_generator,
                            input_test, label_test,
                            num_classes,
                            input_train_shape,
                            episode, train):
    if train:
        trainning_process_time, samples_for_train = prepare.train_model (type_of_input,
                                                                         train_generator,
                                                                         val_generator,
                                                                         num_classes,
                                                                         input_train_shape)

        df_results_top_k, all_index_predict_order = prepare.test_model (type_input=type_of_input,
                                                                        X_data=input_test,
                                                                        index_true=label_test,
                                                                        episode=episode)
        df_results_top_k['trainning_process_time'] = trainning_process_time
        df_results_top_k['samples_for_train'] = np.shape(train_generator.indexes)[0]

        return df_results_top_k, all_index_predict_order
    else:
        df_results_top_k, all_index_predict_order = prepare.test_model (type_input=type_of_input,
                                                                    X_data=input_test,
                                                                    index_true=label_test,
                                                                        episode=episode)
        df_results_top_k ['trainning_process_time'] = 0
        df_results_top_k ['samples_for_train'] = 0


        return df_results_top_k, all_index_predict_order




def fixed_window_top_k(label_input_type, start_epi_test=0, stop_epi_test=2000):
    data_for_train, data_for_validation, data_for_test, num_classes = prepare.read_all_data()
    all_dataset_s008 = pd.concat([data_for_train, data_for_validation], axis=0)

    BATCH_SIZE = 32
    start_index_s008 = 0

    train = True

    input_for_train, label_for_train = tls.extract_training_data_from_s008_sliding_window(all_dataset_s008,
                                                                                          start_index_s008,
                                                                                          label_input_type)
    input_train, input_validation = sliding_prepare_for_trainning(input_for_train, label_input_type)
    label_train, label_validation = sliding_prepare_label_for_trainning(label_for_train)

    if label_input_type == 'lidar_coord':
        X_lidar_train = input_train[0]
        X_coord_train = input_train[1]
        train_generator = prepare.DataGenerator_both(X_lidar_train,
                                                     X_coord_train,
                                                     label_train, BATCH_SIZE, shuffle=True)
        X_lidar_val = input_validation[0]
        X_coord_val = input_validation[1]
        val_generator = prepare.DataGenerator_both(X_lidar_val,
                                                   X_coord_val,
                                                   label_validation, BATCH_SIZE)

        input_train_shape = [X_lidar_train.shape[1:], X_coord_train.shape[1:]]
    elif label_input_type == 'lidar':
        train_generator = prepare.DataGenerator_lidar(input_train, label_train, BATCH_SIZE, shuffle=True)
        val_generator = prepare.DataGenerator_lidar(input_validation, label_validation, BATCH_SIZE, shuffle=True)
        input_train_shape = input_train.shape[1:]
    elif label_input_type == 'coord':
        train_generator = prepare.DataGenerator_coord(input_train,
                                                      label_train,
                                                      BATCH_SIZE, shuffle=True)
        val_generator = prepare.DataGenerator_coord(input_validation,
                                                    label_validation,
                                                    BATCH_SIZE, shuffle=True)
        input_train_shape = input_train.shape[1:]


    #episode_for_test = np.arange(0, episodes_for_test, 1)
    df_all_results_top_k = pd.DataFrame ()
    df_all_index_predict = pd.DataFrame()
    for i in range(start_epi_test, stop_epi_test, 1):
        if i in data_for_test['Episode'].tolist ():
            input_test, label_test = tls.extract_test_data_from_s009_sliding_window(i,
                                                                                    label_input_type,
                                                                                    data_for_test)
            label_test = np.array(label_test)
            if label_input_type == 'coord':
                input_test = np.array(input_test)
            if label_input_type == 'lidar':
                input_test = np.array(input_test)
                input_test = input_test.reshape (input_test.shape [0], 20, 200, 10)
            if label_input_type == 'lidar_coord':
                input_test_lidar = np.array(input_test[1])
                input_test_coord = np.array(input_test[0])
                input_test_lidar = input_test_lidar.reshape(input_test_lidar.shape[0], 20, 200, 10)
                input_test = [input_test_lidar, input_test_coord]

            print(i, end=' ', flush=True)
            df_results_top_k, all_index_predict_order = beam_selection_ruseckas(label_input_type,
                                    train_generator, val_generator,
                                    input_test, label_test,
                                    num_classes,
                                    input_train_shape,
                                    i, train)
            df_all_results_top_k = pd.concat ([df_all_results_top_k, df_results_top_k], ignore_index=True)
            df_all_index_predict = pd.concat ([df_all_index_predict, all_index_predict_order], ignore_index=True)
            path_result = ('../../results/score/ruseckas/online/top_k/') + label_input_type + '/fixed_window/'
            df_all_results_top_k.to_csv (path_result + 'all_results_fixed_window_top_k.csv', index=False)
            df_all_index_predict.to_csv (path_result + 'all_index_predict_fixed_window_top_k.csv', index=False)

def incremental_window_top_k(label_input_type):
    data_for_train, data_for_validation, s009_data, num_classes = prepare.read_all_data ()
    all_dataset_s008 = pd.concat ([data_for_train, data_for_validation], axis=0)
    episode_for_test = np.arange (0, 2000, 20)
    BATCH_SIZE = 32
    train = True

    df_all_results_top_k = pd.DataFrame ()
    df_all_index_predict = pd.DataFrame ()
    for i in range (len (episode_for_test)):
        # for i in tqdm(range(len(episode_for_test))):
        #i=101
        if i in s009_data ['Episode'].tolist ():
            if episode_for_test[i] == 0:
                start_index_s008 = 0
                input_for_train, label_for_train = tls.extract_training_data_from_s008_sliding_window (all_dataset_s008,
                                                                                                       start_index_s008,
                                                                                                       label_input_type)

                input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (episode_for_test[i],
                                                                                                 label_input_type,
                                                                                                 s009_data)
                input_train, input_validation = sliding_prepare_for_trainning (input_for_train, label_input_type)
                label_train, label_validation = sliding_prepare_label_for_trainning (label_for_train)
                label_test = np.array (label_for_test)

                if label_input_type == 'lidar_coord':
                    X_lidar_train = input_train[0]
                    X_coord_train = input_train[1]
                    train_generator = prepare.DataGenerator_both (X_lidar_train,
                                                                  X_coord_train,
                                                                  label_train, BATCH_SIZE, shuffle=True)
                    X_lidar_val = input_validation[0]
                    X_coord_val = input_validation[1]
                    val_generator = prepare.DataGenerator_both (X_lidar_val,
                                                                X_coord_val,
                                                                label_validation, BATCH_SIZE)
                    input_test_lidar = np.array (input_for_test [1])
                    input_test_coord = np.array (input_for_test [0])
                    input_test_lidar = input_test_lidar.reshape (input_test_lidar.shape [0], 20, 200, 10)
                    input_test = [input_test_lidar, input_test_coord]

                    input_train_shape = [X_lidar_train.shape [1:], X_coord_train.shape [1:]]
                elif label_input_type == 'lidar':
                    train_generator = prepare.DataGenerator_lidar (input_train, label_train, BATCH_SIZE, shuffle=True)
                    val_generator = prepare.DataGenerator_lidar (input_validation, label_validation, BATCH_SIZE,
                                                                 shuffle=True)
                    input_train_shape = input_train.shape [1:]
                    input_test = np.array (input_for_test)
                    input_test = input_test.reshape (input_test.shape [0], 20, 200, 10)
                elif label_input_type == 'coord':
                    train_generator = prepare.DataGenerator_coord (input_train,
                                                                   label_train,
                                                                   BATCH_SIZE, shuffle=True)
                    val_generator = prepare.DataGenerator_coord (input_validation,
                                                                 label_validation,
                                                                 BATCH_SIZE, shuffle=True)
                    input_train_shape = input_train.shape [1:]
                    input_test = np.array (input_for_test)
            else:
                start_index_s008 = 0
                start_index_s009 = 0
                end_index_s009 = episode_for_test[i]
                input_for_train_s008, label_for_train_s008 = tls.extract_training_data_from_s008_sliding_window (all_dataset_s008,
                                                                                                                 start_index_s008,
                                                                                                                 label_input_type)
                input_for_train_s009, label_for_train_s009 = tls.extract_training_data_from_s009_sliding_window (s009_data=s009_data,
                                                                                                                 start_index=start_index_s009,
                                                                                                                 end_index=end_index_s009,
                                                                                                                 input_type=label_input_type)

                input_train_S008, input_validation_s008 = sliding_prepare_for_trainning (input_for_train_s008,
                                                                                         label_input_type)
                input_train_s009, input_validation_s009 = sliding_prepare_for_trainning (input_for_train_s009,
                                                                                         label_input_type)
                label_train_s008, label_val_s008 = sliding_prepare_label_for_trainning (label_for_train_s008)
                label_train_s009, label_val_s009 = sliding_prepare_label_for_trainning (label_for_train_s009)

                label_train = np.concatenate ((label_train_s008, label_train_s009), axis=0)
                label_validation = np.concatenate ((label_val_s008, label_val_s009), axis=0)

                if label_input_type == 'lidar_coord':
                    input_train_lidar = np.concatenate ((input_train_S008 [0], input_train_s009 [0]), axis=0)
                    input_validation_lidar = np.concatenate ((input_validation_s008 [0], input_validation_s009 [0]),
                                                             axis=0)

                    input_train_coord = np.concatenate ((input_train_S008 [1], input_train_s009 [1]), axis=0)
                    input_validation_coord = np.concatenate ((input_validation_s008 [1], input_validation_s009 [1]),
                                                             axis=0)
                    input_train = [input_train_lidar, input_train_coord]
                    input_validation = [input_validation_lidar, input_validation_coord]
                else:
                    input_train = np.concatenate ((input_train_S008, input_train_s009), axis=0)
                    input_validation = np.concatenate ((input_validation_s008, input_validation_s009), axis=0)



                input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (episode_for_test[i],
                                                                                                 label_input_type,
                                                                                                 s009_data)
                label_test = np.array (label_for_test)

                if label_input_type == 'coord':

                    train_generator = prepare.DataGenerator_coord (input_train,
                                                                   label_train,
                                                                   BATCH_SIZE, shuffle=True)
                    val_generator = prepare.DataGenerator_coord (input_validation,
                                                                 label_validation,
                                                                 BATCH_SIZE, shuffle=True)
                    input_train_shape = input_train.shape [1:]
                    input_test = np.array (input_for_test)
                elif label_input_type == 'lidar_coord':
                    X_lidar_train = input_train [0]
                    X_coord_train = input_train [1]
                    train_generator = prepare.DataGenerator_both (X_lidar_train,
                                                                  X_coord_train,
                                                                  label_train, BATCH_SIZE, shuffle=True)
                    X_lidar_val = input_validation [0]
                    X_coord_val = input_validation [1]
                    val_generator = prepare.DataGenerator_both (X_lidar_val,
                                                                X_coord_val,
                                                                label_validation, BATCH_SIZE)
                    input_test_lidar = np.array (input_for_test [1])
                    input_test_coord = np.array (input_for_test [0])
                    input_test_lidar = input_test_lidar.reshape (input_test_lidar.shape [0], 20, 200, 10)
                    input_test = [input_test_lidar, input_test_coord]

                    input_train_shape = [X_lidar_train.shape [1:], X_coord_train.shape [1:]]
                elif label_input_type == 'lidar':
                    train_generator = prepare.DataGenerator_lidar (input_train, label_train, BATCH_SIZE,
                                                                   shuffle=True)
                    val_generator = prepare.DataGenerator_lidar (input_validation, label_validation, BATCH_SIZE,
                                                                 shuffle=True)
                    input_train_shape = input_train.shape [1:]
                    input_test = np.array (input_for_test)
                    input_test = input_test.reshape (input_test.shape [0], 20, 200, 10)
        print ('Ep test: ',episode_for_test[i], end=' ', flush=True)
        df_results_top_k, all_index_predict_order = beam_selection_ruseckas (label_input_type,
                                                                             train_generator, val_generator,
                                                                             input_test, label_test,
                                                                             num_classes,
                                                                             input_train_shape,
                                                                             episode_for_test[i], train)
        df_all_results_top_k = pd.concat ([df_all_results_top_k, df_results_top_k], ignore_index=True)
        df_all_index_predict = pd.concat ([df_all_index_predict, all_index_predict_order], ignore_index=True)
        path_result = ('../../results/score/ruseckas/online/top_k/' + label_input_type + '/incremental_window/')
        ##print (path_result)
        df_all_results_top_k.to_csv (path_result + 'all_results_incremental_window_top_k.csv', index=False)
        df_all_index_predict.to_csv (path_result + 'all_index_predict_incremental_window_top_k.csv', index=False)


def sliding_window_top_k(label_input_type,
                         episodes_for_test=10,#2000,
                         window_size=1000):
    data_for_train, data_for_validation, s009_data, num_classes = prepare.read_all_data ()
    all_dataset_s008 = pd.concat ([data_for_train, data_for_validation], axis=0)

    episode_for_test = np.arange (0, episodes_for_test, 1)
    see_trainning_progress = 0
    start_index_s009 = 0
    nro_episodes_s008 = 2086
    train = True
    BATCH_SIZE = 32

    df_all_results_top_k = pd.DataFrame ()
    df_all_index_predict = pd.DataFrame()
    for i in range (len (episode_for_test)):
        # for i in tqdm(range(len(episode_for_test))):
        #i=101
        if i in s009_data ['Episode'].tolist ():
            if i == 0:
                start_index_s008 = nro_episodes_s008 - window_size
                input_for_train, label_for_train = tls.extract_training_data_from_s008_sliding_window (all_dataset_s008,
                                                                                                       start_index_s008,
                                                                                                       label_input_type)

                input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (i,
                                                                                                 label_input_type,
                                                                                                 s009_data)
                input_train, input_validation = sliding_prepare_for_trainning (input_for_train, label_input_type)
                label_train, label_validation = sliding_prepare_label_for_trainning (label_for_train)
                label_test = np.array (label_for_test)

                if label_input_type == 'lidar_coord':
                    X_lidar_train = input_train[0]
                    X_coord_train = input_train[1]
                    train_generator = prepare.DataGenerator_both (X_lidar_train,
                                                                  X_coord_train,
                                                                  label_train, BATCH_SIZE, shuffle=True)
                    X_lidar_val = input_validation[0]
                    X_coord_val = input_validation[1]
                    val_generator = prepare.DataGenerator_both (X_lidar_val,
                                                                X_coord_val,
                                                                label_validation, BATCH_SIZE)
                    input_test_lidar = np.array (input_for_test [1])
                    input_test_coord = np.array (input_for_test [0])
                    input_test_lidar = input_test_lidar.reshape (input_test_lidar.shape [0], 20, 200, 10)
                    input_test = [input_test_lidar, input_test_coord]

                    input_train_shape = [X_lidar_train.shape [1:], X_coord_train.shape [1:]]
                elif label_input_type == 'lidar':
                    train_generator = prepare.DataGenerator_lidar (input_train, label_train, BATCH_SIZE, shuffle=True)
                    val_generator = prepare.DataGenerator_lidar (input_validation, label_validation, BATCH_SIZE,
                                                                 shuffle=True)
                    input_train_shape = input_train.shape [1:]
                    input_test = np.array (input_for_test)
                    input_test = input_test.reshape (input_test.shape [0], 20, 200, 10)
                elif label_input_type == 'coord':
                    train_generator = prepare.DataGenerator_coord (input_train,
                                                                   label_train,
                                                                   BATCH_SIZE, shuffle=True)
                    val_generator = prepare.DataGenerator_coord (input_validation,
                                                                 label_validation,
                                                                 BATCH_SIZE, shuffle=True)
                    input_train_shape = input_train.shape [1:]
                    input_test = np.array (input_for_test)

            else:
                start_index_s008 = (nro_episodes_s008 - window_size) + i
                #start_index_s008 = 1565
                if start_index_s008 < nro_episodes_s008:
                    start_index_s009 = 0
                    end_index_s009 = window_size - (nro_episodes_s008 - start_index_s008)


                    input_for_train_s008, label_for_train_s008 = tls.extract_training_data_from_s008_sliding_window (
                        all_dataset_s008,
                        start_index_s008,
                        label_input_type)
                    input_for_train_s009, label_for_train_s009 = tls.extract_training_data_from_s009_sliding_window (
                        s009_data=s009_data,
                        start_index=start_index_s009,
                        end_index=end_index_s009,
                        input_type=label_input_type)

                    input_train_S008, input_validation_s008 = sliding_prepare_for_trainning (input_for_train_s008,
                                                                                             label_input_type)
                    input_train_s009, input_validation_s009 = sliding_prepare_for_trainning(input_for_train_s009,
                                                                                            label_input_type)
                    if label_input_type == 'lidar_coord':
                        input_train_lidar = np.concatenate ((input_train_S008[0], input_train_s009[0]), axis=0)
                        input_validation_lidar = np.concatenate ((input_validation_s008[0], input_validation_s009[0]), axis=0)

                        input_train_coord = np.concatenate ((input_train_S008[1], input_train_s009[1]), axis=0)
                        input_validation_coord = np.concatenate ((input_validation_s008[1], input_validation_s009[1]), axis=0)
                        input_train = [input_train_lidar, input_train_coord]
                        input_validation = [input_validation_lidar, input_validation_coord]
                    else:
                        input_train = np.concatenate ((input_train_S008, input_train_s009), axis=0)
                        input_validation = np.concatenate ((input_validation_s008, input_validation_s009), axis=0)

                    label_train_s008, label_val_s008 = sliding_prepare_label_for_trainning (label_for_train_s008)
                    label_train_s009, label_val_s009 = sliding_prepare_label_for_trainning (label_for_train_s009)



                    label_train = np.concatenate ((label_train_s008, label_train_s009), axis=0)
                    label_validation = np.concatenate ((label_val_s008, label_val_s009), axis=0)

                    input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (i,
                                                                                                     label_input_type,
                                                                                                     s009_data)
                    label_test = np.array(label_for_test)

                    if label_input_type == 'coord':

                        train_generator = prepare.DataGenerator_coord (input_train,
                                                                       label_train,
                                                                       BATCH_SIZE, shuffle=True)
                        val_generator = prepare.DataGenerator_coord (input_validation,
                                                                     label_validation,
                                                                     BATCH_SIZE, shuffle=True)
                        input_train_shape = input_train.shape [1:]
                        input_test = np.array (input_for_test)
                    elif label_input_type == 'lidar_coord':
                        X_lidar_train = input_train [0]
                        X_coord_train = input_train [1]
                        train_generator = prepare.DataGenerator_both (X_lidar_train,
                                                                      X_coord_train,
                                                                      label_train, BATCH_SIZE, shuffle=True)
                        X_lidar_val = input_validation [0]
                        X_coord_val = input_validation [1]
                        val_generator = prepare.DataGenerator_both (X_lidar_val,
                                                                    X_coord_val,
                                                                    label_validation, BATCH_SIZE)
                        input_test_lidar = np.array (input_for_test [1])
                        input_test_coord = np.array (input_for_test [0])
                        input_test_lidar = input_test_lidar.reshape (input_test_lidar.shape [0], 20, 200, 10)
                        input_test = [input_test_lidar, input_test_coord]

                        input_train_shape = [X_lidar_train.shape [1:], X_coord_train.shape [1:]]
                    elif label_input_type == 'lidar':
                        train_generator = prepare.DataGenerator_lidar (input_train, label_train, BATCH_SIZE,
                                                                       shuffle=True)
                        val_generator = prepare.DataGenerator_lidar (input_validation, label_validation, BATCH_SIZE,
                                                                     shuffle=True)
                        input_train_shape = input_train.shape [1:]
                        input_test = np.array (input_for_test)
                        input_test = input_test.reshape (input_test.shape [0], 20, 200, 10)

                else:
                    end_index_s009 = start_index_s009 + window_size
                    input_for_train_s009, label_for_train_s009 = tls.extract_training_data_from_s009_sliding_window (
                        s009_data=s009_data,
                        start_index=start_index_s009,
                        end_index=end_index_s009,
                        input_type=label_input_type)

                    label_train, label_validation = sliding_prepare_label_for_trainning (label_for_train_s009)
                    input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (i,
                                                                                                     label_input_type,
                                                                                                     s009_data)

                    label_test = np.array (label_for_test)
                    input_train, input_validation = sliding_prepare_for_trainning (input_for_train_s009, label_input_type)

                    if label_input_type == 'coord':

                        train_generator = prepare.DataGenerator_coord (input_train,
                                                                       label_train,
                                                                       BATCH_SIZE, shuffle=True)
                        val_generator = prepare.DataGenerator_coord (input_validation,
                                                                     label_validation,
                                                                     BATCH_SIZE, shuffle=True)
                        input_train_shape = input_train.shape [1:]
                        input_test = np.array (input_for_test)
                    elif label_input_type == 'lidar_coord':
                        X_lidar_train = input_train [0]
                        X_coord_train = input_train [1]
                        train_generator = prepare.DataGenerator_both (X_lidar_train,
                                                                      X_coord_train,
                                                                      label_train, BATCH_SIZE, shuffle=True)
                        X_lidar_val = input_validation [0]
                        X_coord_val = input_validation [1]
                        val_generator = prepare.DataGenerator_both (X_lidar_val,
                                                                    X_coord_val,
                                                                    label_validation, BATCH_SIZE)
                        input_test_lidar = np.array (input_for_test [1])
                        input_test_coord = np.array (input_for_test [0])
                        input_test_lidar = input_test_lidar.reshape (input_test_lidar.shape [0], 20, 200, 10)
                        input_test = [input_test_lidar, input_test_coord]

                        input_train_shape = [X_lidar_train.shape [1:], X_coord_train.shape [1:]]
                    elif label_input_type == 'lidar':
                        train_generator = prepare.DataGenerator_lidar (input_train, label_train, BATCH_SIZE,
                                                                       shuffle=True)
                        val_generator = prepare.DataGenerator_lidar (input_validation, label_validation, BATCH_SIZE,
                                                                     shuffle=True)
                        input_train_shape = input_train.shape [1:]
                        input_test = np.array (input_for_test)
                        input_test = input_test.reshape (input_test.shape [0], 20, 200, 10)

                    start_index_s009 += 1

        print (i, end=' ', flush=True)
        df_results_top_k, all_index_predict_order = beam_selection_ruseckas (label_input_type,
                                                                             train_generator, val_generator,
                                                                             input_test, label_test,
                                                                             num_classes,
                                                                             input_train_shape,
                                                                             i, train)
        df_all_results_top_k = pd.concat ([df_all_results_top_k, df_results_top_k], ignore_index=True)
        df_all_index_predict = pd.concat ([df_all_index_predict, all_index_predict_order], ignore_index=True)
        path_result = ('../../results/score/ruseckas/online/top_k/' + label_input_type + '/sliding_window/')
        ##print (path_result)
        df_all_results_top_k.to_csv (path_result + 'all_results_sliding_window_top_k.csv', index=False)
        df_all_index_predict.to_csv (path_result + 'all_index_predict_sliding_window_top_k.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_type', type=str, default='coord')
    episodes_for_test = 2000
    args = parser.parse_args()

    print('---------------------')
    print('|     Fixed Window      |')
    print('|   input = '+args.input_type)
    fixed_window_top_k(args.input_type, start_epi_test=0, stop_epi_test=episodes_for_test)

def test_LOS_NLOS(connection_type,label_input_type):

    train_data, val_data, test_data, num_classes = prepare.read_all_data()
    reduce_samples_for_train = True

    #connection_type = 'NLOS'
    #label_input_type = 'coord'#'lidar' 'lidar_coord'
    BATCH_SIZE = 32
    train = True

    if connection_type == 'LOS':

        if reduce_samples_for_train:
            data_for_train = train_data[train_data['LOS'] == 'LOS=1'].sample(n=4285, random_state=1)
            data_for_validation = val_data[val_data['LOS'] == 'LOS=1'].sample(n=427, random_state=1)
        else:
            data_for_train = train_data[train_data['LOS'] == 'LOS=1']
            data_for_validation = val_data[val_data['LOS'] == 'LOS=1']

        data_for_test = test_data[test_data['LOS'] == 'LOS=1']


        a=0


    elif connection_type == 'NLOS':

        if reduce_samples_for_train:
            data_for_train = train_data[train_data['LOS'] == 'LOS=0'].sample(n=4285, random_state=1)
            data_for_validation = val_data[val_data['LOS'] == 'LOS=0'].sample(n=427, random_state=1)
        else:
            data_for_train = train_data[train_data['LOS'] == 'LOS=0']
            data_for_validation = val_data[val_data['LOS'] == 'LOS=0']

        data_for_test = test_data[test_data['LOS'] == 'LOS=0']

    elif connection_type == 'ALL':

        if reduce_samples_for_train:
            data_for_train = train_data.sample (n=4285, random_state=1)
            data_for_validation = val_data.sample (n=427, random_state=1)
        else:
            data_for_train = train_data
            data_for_validation = val_data

        data_for_test = test_data

    #data_for_test = test_data
    label_for_train = data_for_train ['index_beams'].tolist ()
    label_train = np.array(label_for_train)
    label_for_validation = data_for_validation ['index_beams'].tolist ()
    label_validation = np.array(label_for_validation)
    label_for_test = data_for_test ['index_beams'].tolist ()
    label_test = np.array(label_for_test)

    if label_input_type == 'coord':
        input_train = data_for_train['coord'].tolist()
        input_train = np.array(input_train)
        input_validation = data_for_validation ['coord'].tolist ()
        input_validation = np.array (input_validation)

        train_generator = prepare.DataGenerator_coord (input_train,
                                                       label_train,
                                                       BATCH_SIZE, shuffle=True)
        val_generator = prepare.DataGenerator_coord (input_validation,
                                                     label_validation,
                                                     BATCH_SIZE, shuffle=True)

        input_train_shape = input_train.shape [1:]
        input_test = data_for_test ['coord'].tolist ()
        input_test = np.array (input_test)

    elif label_input_type == 'lidar':
        input_train = data_for_train['lidar'].tolist()
        input_train = np.array(input_train)
        input_train = input_train.reshape(input_train.shape [0], 20, 200, 10)

        input_validation = data_for_validation['lidar'].tolist()
        input_validation = np.array(input_validation)
        input_validation = input_validation.reshape(input_validation.shape [0], 20, 200, 10)

        train_generator = prepare.DataGenerator_lidar(input_train,
                                                      label_train,
                                                      BATCH_SIZE, shuffle=True)
        val_generator = prepare.DataGenerator_lidar(input_validation,
                                                    label_validation,
                                                    BATCH_SIZE, shuffle=True)
        input_train_shape = input_train.shape[1:]

        input_test = data_for_test['lidar'].tolist()
        input_test = np.array(input_test)
        input_test = input_test.reshape(input_test.shape[0], 20, 200, 10)

    elif label_input_type =='lidar_coord':
        input_train_lidar = data_for_train ['lidar'].tolist ()
        input_train_lidar = np.array (input_train_lidar)
        input_train_lidar = input_train_lidar.reshape (input_train_lidar.shape [0], 20, 200, 10)

        input_validation_lidar = data_for_validation['lidar'].tolist()
        input_validation_lidar = np.array(input_validation_lidar)
        input_validation_lidar = input_validation_lidar.reshape(input_validation_lidar.shape [0], 20, 200, 10)

        input_test_lidar = data_for_test['lidar'].tolist()
        input_test_lidar = np.array(input_test_lidar)
        input_test_lidar = input_test_lidar.reshape(input_test_lidar.shape[0], 20, 200, 10)

        input_train_coord = data_for_train['coord'].tolist()
        input_train_coord = np.array(input_train_coord)
        input_validation_coord = data_for_validation['coord'].tolist()
        input_validation_coord = np.array(input_validation_coord)

        input_test_coord = data_for_test['coord'].tolist()
        input_test_coord = np.array(input_test_coord)

        train_generator = prepare.DataGenerator_both (input_train_lidar,
                                                      input_train_coord,
                                                      label_train, BATCH_SIZE, shuffle=True)

        val_generator = prepare.DataGenerator_both (input_validation_lidar,
                                                    input_validation_coord,
                                                    label_validation, BATCH_SIZE)

        input_train_shape = [input_train_lidar.shape[1:], input_train_coord.shape[1:]]

        input_test = [input_test_lidar, input_test_coord]

    print(' input: ', label_input_type)
    print(' sample for trainning: ', np.shape(train_generator.indexes)[0])
    print(' sample for validation: ', np.shape(val_generator.indexes)[0])
    if label_input_type == 'lidar_coord':
        print(' sample for test: ', np.shape(input_test[0])[0])
    else:
        print(' sample for test: ', np.shape(input_test)[0])


    df_results_top_k, all_index_predict_order = beam_selection_ruseckas (label_input_type,
                                                                         train_generator, val_generator,
                                                                         input_test, label_test,
                                                                         num_classes,
                                                                         input_train_shape,
                                                                         0, train)

    if reduce_samples_for_train:
        path_result = (
                '../../results/score/ruseckas/split_dataset_LOS_NLOS/' +
                label_input_type + '/' +
                connection_type + '/' +
                'less_samples_for_train/')
    else:
        path_result = (
                '../../results/score/ruseckas/split_dataset_LOS_NLOS/' +
                label_input_type + '/' +
                connection_type + '/')
    df_results_top_k.to_csv (path_result + label_input_type + '_results_top_k_ruseckas_' + connection_type + '_ok_.csv',
                             index=False)

    a=0

def read_results_conventional_evaluation(label_input_type):
    path = '../../results/score/ruseckas/split_dataset_LOS_NLOS/'
    reduce_samples_for_train = True
    add_path = 'less_samples_for_train/'

    connection_type = 'LOS'


    path_result = path + label_input_type + '/' + connection_type + '/'
    file_name = label_input_type + '_results_top_k_ruseckas_' + connection_type + '_ok_.csv'
    if reduce_samples_for_train:
        data_LOS = pd.read_csv (path_result + add_path+ file_name, delimiter=',')
    else:
        data_LOS = pd.read_csv (path_result + file_name, delimiter=',')
    LOS = data_LOS[data_LOS['top-k'] <= 10]

    connection_type = 'NLOS'
    path_result = path + label_input_type + '/' + connection_type + '/'
    file_name = label_input_type + '_results_top_k_ruseckas_' + connection_type + '_ok_.csv'
    if reduce_samples_for_train:
        data_NLOS = pd.read_csv (path_result + add_path + file_name, delimiter=',')
    else:
        data_NLOS = pd.read_csv (path_result + file_name, delimiter=',')
    NLOS = data_NLOS[data_NLOS['top-k'] <= 10]

    connection_type = 'ALL'
    path_result = path + label_input_type + '/' + connection_type + '/'
    file_name = label_input_type + '_results_top_k_ruseckas_' + connection_type + '_ok_.csv'
    if reduce_samples_for_train:
        data_ALL = pd.read_csv (path_result + add_path + file_name, delimiter=',')
    else:
       data_ALL = pd.read_csv (path_result + file_name, delimiter=',')
    ALL = data_ALL[data_ALL['top-k'] <= 10]

    return LOS, NLOS, ALL
def plot_test_LOS_NLOS():
    import matplotlib.pyplot as plt

    reduce_samples_for_train = True

    label_input_type = 'coord'
    data_LOS_coord, data_NLOS_coord, data_ALL_coord = read_results_conventional_evaluation(label_input_type)
    label_input_type = 'lidar'
    data_LOS_lidar, data_NLOS_lidar, data_ALL_lidar = read_results_conventional_evaluation(label_input_type)
    label_input_type = 'lidar_coord'
    data_LOS_lidar_coord, data_NLOS_lidar_coord, data_ALL_lidar_coord = read_results_conventional_evaluation(label_input_type)


    fig, ax = plt.subplots (1, 3, figsize=(14, 6), sharey=True)
    plt.subplots_adjust (left=0.08, right=0.98, bottom=0.1, top=0.9, hspace=0.12, wspace=0.05)
    size_of_font = 18
    ax [0].plot (data_LOS_coord['top-k'], data_LOS_coord ['score'], label='Coord LOS', marker='o')
    ax [0].text (data_LOS_coord ['top-k'].min (), data_LOS_coord ['score'] [0],
                 str (round (data_LOS_coord ['score'] [0], 3)))
    ax [0].plot (data_NLOS_coord['top-k'], data_NLOS_coord ['score'], label='Coord NLOS', marker='o')
    ax [0].text (data_NLOS_coord ['top-k'].min (), data_NLOS_coord ['score'] [0],
                 str (round (data_NLOS_coord ['score'] [0], 3)))
    ax [0].plot (data_ALL_coord['top-k'], data_ALL_coord ['score'], label='Coord ALL', marker='o')
    ax [0].text (data_ALL_coord['top-k'].min(), data_ALL_coord ['score'][0],
                 str(round(data_ALL_coord ['score'][0], 3)))
    ax [0].grid ()
    ax [0].set_xticks (data_LOS_coord ['top-k'])
    ax [0].set_xlabel ('Coordenadas \n Top-k  ', font='Times New Roman', fontsize=size_of_font)

    ax [1].plot (data_LOS_lidar['top-k'], data_LOS_lidar['score'], label='LOS', marker='o')
    ax [1].plot (data_NLOS_lidar['top-k'], data_NLOS_lidar['score'], label='NLOS', marker='o')
    ax [1].plot (data_ALL_lidar['top-k'], data_ALL_lidar['score'], label='ALL', marker='o')
    ax [1].text (data_LOS_lidar['top-k'].min(), data_LOS_lidar['score'][0],
                 str(round(data_LOS_lidar ['score'][0], 3)))
    ax [1].text (data_NLOS_lidar ['top-k'].min (), data_NLOS_lidar ['score'] [0],
                 str (round (data_NLOS_lidar['score'][0], 3)))
    ax [1].text (data_ALL_lidar ['top-k'].min (), data_ALL_lidar ['score'] [0],
                 str (round (data_ALL_lidar ['score'] [0], 3)))

    ax [1].grid ()
    ax [1].set_xticks (data_LOS_coord ['top-k'])
    ax [1].set_xlabel ('Lidar \n Top-k  ', font='Times New Roman', fontsize=size_of_font)

    ax [2].plot (data_LOS_lidar_coord['top-k'], data_LOS_lidar_coord['score'],
                 label='Lidar Coord LOS', marker='o')
    ax [2].plot (data_NLOS_lidar_coord['top-k'], data_NLOS_lidar_coord['score'],
                 label='Lidar Coord NLOS', marker='o')
    ax [2].plot (data_ALL_lidar_coord['top-k'], data_ALL_lidar_coord['score'],
                 label='Lidar Coord ALL', marker='o')
    ax [2].text (data_LOS_lidar_coord ['top-k'].min (), data_LOS_lidar_coord ['score'] [0],
                 str (round (data_LOS_lidar_coord ['score'] [0], 3)))
    ax [2].text (data_NLOS_lidar_coord ['top-k'].min (), data_NLOS_lidar_coord ['score'] [0],
                 str (round (data_NLOS_lidar_coord ['score'] [0], 3)))
    ax [2].text (data_ALL_lidar_coord ['top-k'].min (), data_ALL_lidar_coord ['score'] [0],
                 str (round (data_ALL_lidar_coord ['score'] [0], 3)))

    ax [2].grid ()
    ax [2].set_xticks (data_LOS_coord['top-k'])
    ax [2].set_xlabel ('Lidar e Coordenadas \n Top-k  ', font='Times New Roman', fontsize=size_of_font)

    ax [0].set_ylabel ('Acurácia', font='Times New Roman', fontsize=size_of_font)
    ax [1].legend ()
    plt.suptitle('Selecao de Feixe usando os modelos do Ruseckas', fontsize=size_of_font, font='Times New Roman')


    path_to_save = '../../results/score/ruseckas/split_dataset_LOS_NLOS/'
    if reduce_samples_for_train:
        file_name = 'performance_accuracy_all_LOS_NLOS_ruseckas_less_samples_for_train.png'
    else:
        file_name = 'performance_accuracy_all_LOS_NLOS_ruseckas.png'
    plt.savefig (path_to_save + file_name, dpi=300, bbox_inches='tight')


label_input_type = 'lidar_coord'
print('--------------------------------------------------------')
print('------- Beam Selection - Ruseckas - Online Learning')
print('--- Input: ', label_input_type)
print('--- Window type: Incremental Window')

#fixed_window_top_k(label_input_type, start_epi_test=0, stop_epi_test=2)
#sliding_window_top_k(label_input_type, episodes_for_test=2000, window_size=1000)

#main()
#plot_test_LOS_NLOS()
#connection = ['NLOS']#, 'NLOS']#, 'ALL']
#label_input_type = ['lidar']#, 'lidar', 'lidar_coord'] *** agora lidar_coord
#for i in connection:
#    for j in label_input_type:
#        test_LOS_NLOS(i, j)

incremental_window_top_k(label_input_type)

