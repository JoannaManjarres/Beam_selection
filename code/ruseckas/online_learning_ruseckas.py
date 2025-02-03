# Imports

import argparse

import numpy as np
import pandas as pd
import prepare_for_online_learning as prepare
import sys
sys.path.append ("../")

# Agora é possível importar o arquivo como um módulo
import tools as tls





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




def fixed_window_top_k(label_input_type, episodes_for_test):
    data_for_train, data_for_validation, data_for_test, num_classes = prepare.read_all_data()
    all_dataset_s008 = pd.concat([data_for_train, data_for_validation], axis=0)

    BATCH_SIZE = 32
    start_index_s008 = 0
    label_input_type = 'coord'
    train= False

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

    episodes_for_test = 2
    episode_for_test = np.arange(0, episodes_for_test, 1)
    df_all_results_top_k = pd.DataFrame ()
    df_all_index_predict = pd.DataFrame()
    for i in range (len (episode_for_test)):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_type', type=str, default='coord')
    episodes_for_test = 2
    args = parser.parse_args()
    fixed_window_top_k(args.input_type, episodes_for_test)

main()

