import sys
import pandas as pd
import numpy as np

sys.path.append ("../")

# Agora é possível importar o arquivo como um módulo
import tools as tls
import online_learning_batool as olb

def prepare_label_for_trainning(label_for_train):
    new_form_for_label = np.array(label_for_train)
    size_of_label = new_form_for_label.shape
    size_of_train = int(size_of_label[0] * 0.8)
    label_train = new_form_for_label[:size_of_train]
    label_validation = new_form_for_label[size_of_train:]
    return label_train, label_validation

def prepare_lidar_for_trainning(input_for_train):
    new_form_for_lidar = np.array (input_for_train)
    lidar_data = new_form_for_lidar.reshape (new_form_for_lidar.shape[0], 20, 200, 10)
    return lidar_data

def prepare_coord_for_trainning(input_for_train):
    new_form_of_input = np.zeros ((len (input_for_train), 2))
    for i in range (len (input_for_train)):
        new_form_of_input [i] = np.array (input_for_train [i])
    return new_form_of_input
def beam_selection_with_inverted_dataset(input_type='lidar_coord'):
    data_for_train, data_for_validation, s009_data, num_classes = olb.read_all_data ()
    all_dataset_s008 = pd.concat([data_for_train, data_for_validation], axis=0)
    size_for_train = int(len(s009_data)*0.8)
    train_s009 = s009_data [:size_for_train]
    val_s009 = s009_data [size_for_train:]

    label_test = np.array(all_dataset_s008['index_beams'].tolist())
    label_train = np.array(train_s009['index_beams'].tolist())
    label_val = np.array(val_s009['index_beams'].tolist())


    if input_type == 'coord':
        input_train = prepare_coord_for_trainning(train_s009['coord'].tolist())
        input_val = prepare_coord_for_trainning(val_s009['coord'].tolist())
        input_test = prepare_coord_for_trainning(all_dataset_s008['coord'].tolist())

    if input_type == 'lidar':
        input_train = prepare_lidar_for_trainning(train_s009['lidar'].tolist())
        input_val = prepare_lidar_for_trainning(val_s009['lidar'].tolist())
        input_test = prepare_lidar_for_trainning(all_dataset_s008['lidar'].tolist())

    if input_type == 'lidar_coord':
        print('input: ', input_type)
        input_train_coord = prepare_coord_for_trainning(train_s009['coord'].tolist())
        input_train_lidar = prepare_lidar_for_trainning(train_s009['lidar'].tolist())
        input_train = [input_train_lidar, input_train_coord]

        input_val_coord = prepare_coord_for_trainning(val_s009['coord'].tolist())
        input_val_lidar = prepare_lidar_for_trainning(val_s009['lidar'].tolist())
        input_val = [input_val_lidar, input_val_coord]

        input_test_coord = prepare_coord_for_trainning(all_dataset_s008['coord'].tolist())
        input_test_lidar = prepare_lidar_for_trainning(all_dataset_s008['lidar'].tolist())
        input_test = [input_test_lidar, input_test_coord]





    see_trainning_progress = False
    df_results_top_k = olb.beam_selection_Batool (input=input_type,
                                              data_train=[input_train, label_train],
                                              data_validation=[input_val, label_val],
                                              data_test=[input_test, label_test],
                                              num_classes=num_classes,
                                              episode=0,
                                              see_trainning_progress=see_trainning_progress,
                                              restore_models=False,
                                              flag_fast_experiment=False)


    path_result = ('../../results/inverter_dataset/score/Batool/') + input_type + '/ALL/'
    print (path_result + 'accuracy_' + input_type + '.csv')
    df_results_top_k.to_csv(path_result + 'accuracy_'+input_type+'.csv', index=False)

    a=0

beam_selection_with_inverted_dataset()