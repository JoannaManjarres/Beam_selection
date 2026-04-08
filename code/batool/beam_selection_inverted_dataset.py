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
def beam_selection_LOS_NLOS_inverted_dataset(connection='LOS', input_type='coord'):
    data_for_train, data_for_validation, s009_data, num_classes = olb.read_all_data ()
    all_dataset_s008 = pd.concat ([data_for_train, data_for_validation], axis=0)
    size_for_train = int (len (s009_data) * 0.8)
    s009_train = s009_data [:size_for_train]
    s009_val = s009_data [size_for_train:]

    # separar dataset em LOS e NLOS
    if connection == 'LOS':
        train_s009 = s009_train [s009_train ['LOS'] == 'LOS=1']
        val_s009 = s009_val [s009_val ['LOS'] == 'LOS=1']
        label_val = np.array (val_s009 ['index_beams'].tolist ())
        label_train = np.array (train_s009 ['index_beams'].tolist ())

        data_for_test = all_dataset_s008 [all_dataset_s008 ['LOS'] == 'LOS=1']
        label_test = np.array (data_for_test ['index_beams'].tolist ())

    if connection == 'NLOS':
        train_s009 = s009_train [s009_train ['LOS'] == 'LOS=0']
        val_s009 = s009_val [s009_val ['LOS'] == 'LOS=0']
        label_val = np.array (val_s009 ['index_beams'].tolist ())
        label_train = np.array (train_s009 ['index_beams'].tolist ())

        data_for_test = all_dataset_s008 [all_dataset_s008 ['LOS'] == 'LOS=0']
        label_test = np.array (data_for_test ['index_beams'].tolist ())

    if input_type == 'coord':
        input_train = prepare_coord_for_trainning(train_s009['coord'].tolist())
        input_val = prepare_coord_for_trainning(val_s009['coord'].tolist())
        input_test = prepare_coord_for_trainning(data_for_test['coord'].tolist())

    if input_type == 'lidar':
        input_train = prepare_lidar_for_trainning(train_s009['lidar'].tolist())
        input_val = prepare_lidar_for_trainning(val_s009['lidar'].tolist())
        input_test = prepare_lidar_for_trainning(data_for_test['lidar'].tolist())

    if input_type == 'lidar_coord':
        print('input: ', input_type)
        input_train_coord = prepare_coord_for_trainning(train_s009['coord'].tolist())
        input_train_lidar = prepare_lidar_for_trainning(train_s009['lidar'].tolist())
        input_train = [input_train_lidar, input_train_coord]

        input_val_coord = prepare_coord_for_trainning(val_s009['coord'].tolist())
        input_val_lidar = prepare_lidar_for_trainning(val_s009['lidar'].tolist())
        input_val = [input_val_lidar, input_val_coord]

        input_test_coord = prepare_coord_for_trainning(data_for_test['coord'].tolist())
        input_test_lidar = prepare_lidar_for_trainning(data_for_test['lidar'].tolist())
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

    path_result = ('../../results/inverter_dataset/score/Batool/') + input_type + '/'+connection+'/'
    print (path_result + 'accuracy_' + input_type +'_'+connection+ '.csv')
    df_results_top_k.to_csv (path_result + 'accuracy_' + input_type +'_'+connection+ '.csv', index=False)





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

def k_fold_simulations(connection, inverter_dataset):
    data_for_train, data_for_validation, data_s009, num_classes = olb.read_all_data ()
    all_dataset_s008 = pd.concat ([data_for_train, data_for_validation], axis=0)

    #connection = 'LOS'
    #inverter_dataset = True
    path = '../../results/results_article_LoS_NLoS/data_for_article/LoS_NLoS_tests/Batool/'

    result = pd.DataFrame ()

    if inverter_dataset:
        path_result = path + 'inverter_dataset/lidar_coord' + '/' + connection + '/'
        if connection == 'LOS':
            train_data = data_s009 [data_s009 ['LOS'] == 'LOS=1']
            test_data = all_dataset_s008 [all_dataset_s008 ['LOS'] == 'LOS=1']
        if connection == 'NLOS':
            train_data = data_s009 [data_s009 ['LOS'] == 'LOS=0']
            test_data = all_dataset_s008 [all_dataset_s008 ['LOS'] == 'LOS=0']
        if connection == 'ALL':
            train_data = data_s009
            test_data = all_dataset_s008
    else:
        path_result = path +'lidar_coord' + '/' + connection + '/'

        if connection == 'LOS':
            train_data = all_dataset_s008 [all_dataset_s008 ['LOS'] == 'LOS=1']
            test_data = data_s009 [data_s009 ['LOS'] == 'LOS=1']
        if connection == 'NLOS':
            train_data = all_dataset_s008 [all_dataset_s008 ['LOS'] == 'LOS=0']
            test_data = data_s009 [data_s009 ['LOS'] == 'LOS=0']
        if connection == 'ALL':
            train_data = all_dataset_s008
            test_data = data_s009

    fold_size = int (len (train_data) / 10)

    for i in range(10):
        dataset_size = len(train_data)
        if i == 0:
            print ('i: ', i)
            start = (i + 1) * fold_size
            end = dataset_size
            new_data_train = train_data [start:end]

        if i == 9:
            print ('i: ', i)
            start = 0
            end = i * fold_size
            new_data_train = train_data [start:end]

        if i != 0 and i != 9:
            print ('i: ', i)
            start_1 = 0
            end_1 = i*fold_size
            start_2 = (i+1)*fold_size
            end_2 = dataset_size
            new_data_train = pd.concat([train_data[start_1:end_1], train_data[start_2:end_2]], axis=0)

        df_results_top_k = beam_selection_LOS_NLOS_ALL_for_k_fold_with_inverter_data (new_data_train,
                                                                                      test_data,
                                                                                      num_classes)
        result = pd.concat ([result, df_results_top_k [df_results_top_k ['top-k'] == 1]], axis=0)
        # all_index_predict_order.to_csv (path_result + 'index_predict_' + input_type + '.csv', index=False)
        result.to_csv (path_result + 'k_fold_accuracy_lidar_coord_' + connection + '.csv', index=False)


def beam_selection_LOS_NLOS_ALL_for_k_fold_with_inverter_data(data_for_train, data_for_test, num_classes):

    '''
    Este método recebe apenas dados de treino e teste, ele apenas separa os dados em train/val
    a selecao de beam esta sendo realizada com dados multimodais
    Deve ser entregue a ele os dados já separados em LOS/NLOS/ALL
    Este metodo deve ser chamado dentro do k-fold, onde os dados de treino e teste já estão separados em LOS/NLOS/ALL
    Retorna un DF com os resultados da acurácia para cada top-k, onde k varia de 1 a 10,
    '''

    input_type = 'lidar_coord'
    #data_for_train, data_for_validation, s009_data, num_classes = olb.read_all_data ()
    #all_dataset_s008 = pd.concat ([data_for_train, data_for_validation], axis=0)
    size_for_train = int (len (data_for_train) * 0.8)
    train_data = data_for_train [:size_for_train]
    val_data = data_for_train [size_for_train:]
    label_val = np.array (val_data ['index_beams'].tolist ())
    label_train = np.array (train_data ['index_beams'].tolist ())

    label_test = np.array (data_for_test ['index_beams'].tolist ())

    # separar dataset em LOS e NLOS

    if input_type == 'coord':
        input_train = prepare_coord_for_trainning (train_data ['coord'].tolist ())
        input_val = prepare_coord_for_trainning (val_data ['coord'].tolist ())
        input_test = prepare_coord_for_trainning (data_for_test ['coord'].tolist ())

    if input_type == 'lidar':
        input_train = prepare_lidar_for_trainning (train_data ['lidar'].tolist ())
        input_val = prepare_lidar_for_trainning (val_data ['lidar'].tolist ())
        input_test = prepare_lidar_for_trainning (data_for_test ['lidar'].tolist ())

    if input_type == 'lidar_coord':
        print ('input: ', input_type)
        input_train_coord = prepare_coord_for_trainning (train_data ['coord'].tolist ())
        input_train_lidar = prepare_lidar_for_trainning (train_data ['lidar'].tolist ())
        input_train = [input_train_lidar, input_train_coord]

        input_val_coord = prepare_coord_for_trainning (val_data ['coord'].tolist ())
        input_val_lidar = prepare_lidar_for_trainning (val_data ['lidar'].tolist ())
        input_val = [input_val_lidar, input_val_coord]

        input_test_coord = prepare_coord_for_trainning (data_for_test ['coord'].tolist ())
        input_test_lidar = prepare_lidar_for_trainning (data_for_test ['lidar'].tolist ())
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

    return df_results_top_k

k_fold_simulations(connection='ALL', inverter_dataset=False)

#beam_selection_with_inverted_dataset()

#beam_selection_LOS_NLOS_inverted_dataset(connection='LOS', input_type='coord')
#beam_selection_LOS_NLOS_inverted_dataset(connection='NLOS', input_type='coord')
#beam_selection_LOS_NLOS_inverted_dataset(connection='LOS', input_type='lidar')
#beam_selection_LOS_NLOS_inverted_dataset(connection='NLOS', input_type='lidar')
#beam_selection_LOS_NLOS_inverted_dataset(connection='LOS', input_type='lidar_coord')
#beam_selection_LOS_NLOS_inverted_dataset(connection='NLOS', input_type='lidar_coord')

