import sys
import pandas as pd
import numpy as np

sys.path.append ("../")

# Agora é possível importar o arquivo como um módulo
import tools as tls
import online_learning_ruseckas as olb
import prepare_for_online_learning as prepare
from sklearn.model_selection import KFold



def k_fold_simulations(connection, inverter_dataset):
    data_for_train, data_for_validation, data_s009, num_classes = prepare.read_all_data ()
    data_s008 = pd.concat ([data_for_train, data_for_validation], axis=0)

    path = '../../results/results_article_LoS_NLoS/data_for_article/LoS_NLoS_tests/ruseckas/'

    result = pd.DataFrame ()


    if inverter_dataset:
        path_result = path + 'inverter_dataset/lidar_coord' + '/' + connection + '/'
        if connection == 'LOS':
            train_data = data_s009 [data_s009 ['LOS'] == 'LOS=1']
            test_data = data_s008 [data_s008 ['LOS'] == 'LOS=1']
        if connection == 'NLOS':
            train_data = data_s009 [data_s009 ['LOS'] == 'LOS=0']
            test_data = data_s008 [data_s008 ['LOS'] == 'LOS=0']
        if connection == 'ALL':
            train_data = data_s009
            test_data = data_s008

    else:
        path_result = path +'lidar_coord' + '/' + connection + '/'

        if connection == 'LOS':
            train_data = data_s008 [data_s008 ['LOS'] == 'LOS=1']
            test_data = data_s009 [data_s009 ['LOS'] == 'LOS=1']
        if connection == 'NLOS':
            train_data = data_s008 [data_s008 ['LOS'] == 'LOS=0']
            test_data = data_s009 [data_s009 ['LOS'] == 'LOS=0']
        if connection == 'ALL':
            train_data = data_s008
            test_data = data_s009

    fold_size = int(len(train_data)/10)


    for i in range(6, 10, 1):
        print(connection, ' - Fold: ', i, ' - Inverter dataset: ', inverter_dataset)
        dataset_size = len(train_data)

        if i == 0:
            #print ('i: ', i)
            start = (i + 1) * fold_size
            end = dataset_size
            new_data_train = train_data[start:end]

        if i == 9:
            #print ('i: ', i)
            start_1 = 0
            end_1 = i * fold_size
            new_data_train = train_data [start_1:end_1]

        if i != 0 and i != 9:
            #print ('i: ', i)
            start_1 = 0
            end_1 = i*fold_size
            start_2 = (i+1)*fold_size
            end_2 = dataset_size
            new_data_train = pd.concat([train_data[start_1:end_1], train_data[start_2:end_2]], axis=0)

        df_results_top_k, all_index_predict_order = beam_selection_LOS_NLOS_ALL_for_k_fold_with_inverter_data(data_for_train=new_data_train,
                                                                                                              data_for_test=test_data,
                                                                                                              num_classes=num_classes)
        result = pd.concat ([result, df_results_top_k [df_results_top_k ['top-k'] == 1]], axis=0)
        result.to_csv (path_result + 'k_fold_accuracy_lidar_coord_' + connection +'_'+ str(i) +'.csv', index=False)

    # all_index_predict_order.to_csv (path_result + 'index_predict_' + input_type + '.csv', index=False)
    #result.to_csv (path_result + 'k_fold_accuracy_lidar_coord_' + connection + '.csv', index=False)
    a=0


def beam_selection_LOS_NLOS_ALL_for_k_fold_with_inverter_data(data_for_train, data_for_test, num_classes):
    input_type = 'lidar_coord'
    size_for_train = int (len (data_for_train) * 0.8)
    train_data = data_for_train [:size_for_train]
    val_data = data_for_train [size_for_train:]

    label_train = np.array (train_data ['index_beams'].tolist ())
    label_validation = np.array (val_data ['index_beams'].tolist ())
    label_test = np.array (data_for_test ['index_beams'].tolist ())

    BATCH_SIZE = 32

    if input_type == 'coord':
        input_train = np.array (train_data ['coord'].tolist ())
        input_validation = np.array (val_data ['coord'].tolist ())
        input_test = np.array (data_for_test ['coord'].tolist ())

        train_generator = prepare.DataGenerator_coord (input_train,
                                                       label_train,
                                                       BATCH_SIZE, shuffle=True)
        val_generator = prepare.DataGenerator_coord (input_validation,
                                                     label_validation,
                                                     BATCH_SIZE, shuffle=True)
        input_train_shape = input_train.shape [1:]

    if input_type == 'lidar':
        lidar_train = np.array (train_data ['lidar'].tolist ())
        lidar_val = np.array (val_data ['lidar'].tolist ())
        lidar_test = np.array (data_for_test ['lidar'].tolist ())

        input_train = lidar_train.reshape (lidar_train.shape [0], 20, 200, 10)
        input_validation = lidar_val.reshape (lidar_val.shape [0], 20, 200, 10)
        input_test = lidar_test.reshape (lidar_test.shape [0], 20, 200, 10)

        train_generator = prepare.DataGenerator_lidar (input_train, label_train, BATCH_SIZE, shuffle=True)
        val_generator = prepare.DataGenerator_lidar (input_validation, label_validation, BATCH_SIZE, shuffle=True)
        input_train_shape = input_train.shape [1:]

    if input_type == 'lidar_coord':
        coord_train = np.array (train_data ['coord'].tolist ())
        coord_val = np.array (val_data ['coord'].tolist ())
        coord_test = np.array (data_for_test ['coord'].tolist ())

        lidar_train = np.array (train_data ['lidar'].tolist ())
        lidar_val = np.array (val_data ['lidar'].tolist ())
        lidar_test = np.array (data_for_test ['lidar'].tolist ())

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

    df_results_top_k, all_index_predict_order = olb.beam_selection_ruseckas (type_of_input=input_type,
                                                                             train_generator=train_generator,
                                                                             val_generator=val_generator,
                                                                             input_test=input_test,
                                                                             label_test=label_test,
                                                                             num_classes=num_classes,
                                                                             input_train_shape=input_train_shape,
                                                                             episode=0, train=True)

    return df_results_top_k, all_index_predict_order

def LOS_NLOS_inverted_dataset_beam_selection(data_s009, connection):
    data_for_train, data_for_validation, _, num_classes = prepare.read_all_data ()
    s008_test = pd.concat ([data_for_train, data_for_validation], axis=0)
    input_type = 'lidar_coord'

    size_for_train = int (len (data_s009) * 0.8)
    s009_train = data_s009 [:size_for_train]
    s009_val = data_s009 [size_for_train:]

    # separar dataset em LOS e NLOS
    if connection == 'LOS':
        train_s009 = s009_train [s009_train ['LOS'] == 'LOS=1']
        val_s009 = s009_val [s009_val ['LOS'] == 'LOS=1']
        label_validation = np.array (val_s009 ['index_beams'].tolist ())
        label_train = np.array (train_s009 ['index_beams'].tolist ())

        test_s008 = s008_test [s008_test ['LOS'] == 'LOS=1']
        label_test = np.array (test_s008 ['index_beams'].tolist ())

    if connection == 'NLOS':
        train_s009 = s009_train [s009_train ['LOS'] == 'LOS=0']
        val_s009 = s009_val [s009_val ['LOS'] == 'LOS=0']
        label_validation = np.array (val_s009 ['index_beams'].tolist ())
        label_train = np.array (train_s009 ['index_beams'].tolist ())

        test_s008 = s008_test [s008_test ['LOS'] == 'LOS=0']
        label_test = np.array (test_s008 ['index_beams'].tolist ())

    BATCH_SIZE = 32

    if input_type == 'coord':
        input_train = np.array (train_s009 ['coord'].tolist ())
        input_validation = np.array (val_s009 ['coord'].tolist ())
        input_test = np.array (test_s008 ['coord'].tolist ())

        train_generator = prepare.DataGenerator_coord (input_train,
                                                       label_train,
                                                       BATCH_SIZE, shuffle=True)
        val_generator = prepare.DataGenerator_coord (input_validation,
                                                     label_validation,
                                                     BATCH_SIZE, shuffle=True)
        input_train_shape = input_train.shape [1:]

    if input_type == 'lidar':
        lidar_train = np.array (train_s009 ['lidar'].tolist ())
        lidar_val = np.array (val_s009 ['lidar'].tolist ())
        lidar_test = np.array (test_s008 ['lidar'].tolist ())

        input_train = lidar_train.reshape (lidar_train.shape [0], 20, 200, 10)
        input_validation = lidar_val.reshape (lidar_val.shape [0], 20, 200, 10)
        input_test = lidar_test.reshape (lidar_test.shape [0], 20, 200, 10)

        train_generator = prepare.DataGenerator_lidar (input_train, label_train, BATCH_SIZE, shuffle=True)
        val_generator = prepare.DataGenerator_lidar (input_validation, label_validation, BATCH_SIZE, shuffle=True)
        input_train_shape = input_train.shape [1:]

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

    df_results_top_k, all_index_predict_order = olb.beam_selection_ruseckas (type_of_input=input_type,
                                                                             train_generator=train_generator,
                                                                             val_generator=val_generator,
                                                                             input_test=input_test,
                                                                             label_test=label_test,
                                                                             num_classes=num_classes,
                                                                             input_train_shape=input_train_shape,
                                                                             episode=0, train=True)

    return df_results_top_k, all_index_predict_order


def inverter_dataset_beam_selection(data_s009, test_s008,num_classes):
    #data_for_train, data_for_validation, _, num_classes = prepare.read_all_data ()
    #test_s008 = pd.concat ([data_for_train, data_for_validation], axis=0)
    input_type = 'lidar_coord'

    size_for_train = int (len (data_s009) * 0.8)
    train_s009 = data_s009 [:size_for_train]
    val_s009 = data_s009 [size_for_train:]

    label_test = np.array (test_s008 ['index_beams'].tolist ())
    label_train = np.array (train_s009 ['index_beams'].tolist ())
    label_validation = np.array (val_s009 ['index_beams'].tolist ())

    BATCH_SIZE = 32

    if input_type == 'coord':
        input_train = np.array (train_s009 ['coord'].tolist ())
        input_validation = np.array (val_s009 ['coord'].tolist ())
        input_test = np.array (test_s008 ['coord'].tolist ())

        train_generator = prepare.DataGenerator_coord (input_train,
                                                       label_train,
                                                       BATCH_SIZE, shuffle=True)
        val_generator = prepare.DataGenerator_coord (input_validation,
                                                     label_validation,
                                                     BATCH_SIZE, shuffle=True)
        input_train_shape = input_train.shape [1:]

    if input_type == 'lidar':
        lidar_train = np.array (train_s009 ['lidar'].tolist ())
        lidar_val = np.array (val_s009 ['lidar'].tolist ())
        lidar_test = np.array (test_s008 ['lidar'].tolist ())

        input_train = lidar_train.reshape (lidar_train.shape [0], 20, 200, 10)
        input_validation = lidar_val.reshape (lidar_val.shape [0], 20, 200, 10)
        input_test = lidar_test.reshape (lidar_test.shape [0], 20, 200, 10)

        train_generator = prepare.DataGenerator_lidar (input_train, label_train, BATCH_SIZE, shuffle=True)
        val_generator = prepare.DataGenerator_lidar (input_validation, label_validation, BATCH_SIZE, shuffle=True)
        input_train_shape = input_train.shape [1:]

    if input_type == 'lidar_coord':
        print (input_type)
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

    df_results_top_k, all_index_predict_order = olb.beam_selection_ruseckas (type_of_input=input_type,
                                                                             train_generator=train_generator,
                                                                             val_generator=val_generator,
                                                                             input_test=input_test,
                                                                             label_test=label_test,
                                                                             num_classes=num_classes,
                                                                             input_train_shape=input_train_shape,
                                                                             episode=0, train=True)

    return df_results_top_k, all_index_predict_order



def beam_selection_LOS_NLOS_inverted_dataset(connection='LOS', input_type='coord'):
    data_for_train, data_for_validation, data_s009, num_classes = prepare.read_all_data ()
    s008_test = pd.concat ([data_for_train, data_for_validation], axis=0)

    size_for_train = int (len (data_s009) * 0.8)
    s009_train = data_s009 [:size_for_train]
    s009_val = data_s009 [size_for_train:]

    # separar dataset em LOS e NLOS
    if connection == 'LOS':
        train_s009 = s009_train [s009_train ['LOS'] == 'LOS=1']
        val_s009 = s009_val [s009_val ['LOS'] == 'LOS=1']
        label_validation = np.array (val_s009 ['index_beams'].tolist ())
        label_train = np.array (train_s009 ['index_beams'].tolist ())

        test_s008 = s008_test [s008_test ['LOS'] == 'LOS=1']
        label_test = np.array (test_s008 ['index_beams'].tolist ())

    if connection == 'NLOS':
        train_s009 = s009_train [s009_train ['LOS'] == 'LOS=0']
        val_s009 = s009_val [s009_val ['LOS'] == 'LOS=0']
        label_validation = np.array (val_s009 ['index_beams'].tolist ())
        label_train = np.array (train_s009 ['index_beams'].tolist ())

        test_s008 = s008_test [s008_test ['LOS'] == 'LOS=0']
        label_test = np.array (test_s008 ['index_beams'].tolist ())

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

    path_result = ('../../results/inverter_dataset/score/Ruseckas/' + input_type + '/'+connection+'/')
    all_index_predict_order.to_csv (path_result + 'index_predict_'+input_type+'.csv', index=False)
    df_results_top_k.to_csv (path_result + 'accuracy_'+input_type+'_'+connection+'.csv', index=False)



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


#beam_selection_with_inverted_dataset()
k_fold_simulations(connection='ALL', inverter_dataset=False)
#k_fold_simulations(connection='ALL', inverter_dataset=False)

#beam_selection_LOS_NLOS_inverted_dataset(connection='LOS', input_type='coord')
#beam_selection_LOS_NLOS_inverted_dataset(connection='NLOS', input_type='coord')
#beam_selection_LOS_NLOS_inverted_dataset(connection='LOS', input_type='lidar')
#beam_selection_LOS_NLOS_inverted_dataset(connection='NLOS', input_type='lidar')
#beam_selection_LOS_NLOS_inverted_dataset(connection='LOS', input_type='lidar_coord')
#beam_selection_LOS_NLOS_inverted_dataset(connection='NLOS', input_type='lidar_coord')