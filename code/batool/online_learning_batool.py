import os
import time
import random

import matplotlib.pyplot as plt
import pandas as pd
import warnings


import numpy as np
from sklearn.preprocessing import normalize
from ModelHandler import add_model,load_model_structure, ModelHandler
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import Dense,concatenate, Dropout, Conv1D, Flatten, Reshape, Activation,multiply,MaxPooling1D,Add,AveragePooling1D,Lambda,Permute
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import model_from_json,Model, load_model
from custom_metrics import *


#from main import getBeamOutput
#from main import open_npz


warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
############################
# Fix the seed
############################
seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


def getBeamOutput(output_file):
    thresholdBelowMax = 6
    print("Reading dataset...", output_file)
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








def prepare_data(input):
    ###############################################################################
    # Outputs (Beams)
    ###############################################################################
    data_folder = '../../data/'
    train_data_folder = 'beams_output/beams_generate_by_me/train_val/'
    train_data_folder = 'beams_output/beam_output_baseline_raymobtime_s008/'
    file_train = 'beams_output_train.npz'
    file_val = 'beams_output_validation.npz'
    test_data_folder = 'beams_output/beam_output_baseline_raymobtime_s009/'
    file_test = 'beams_output_test.npz'

    output_train_file = data_folder + train_data_folder + file_train
    output_validation_file = data_folder + train_data_folder + file_val
    output_test_file = data_folder + test_data_folder + file_test

    y_train, num_classes = getBeamOutput (output_train_file)
    y_validation, _ = getBeamOutput (output_validation_file)
    y_test, _ = getBeamOutput (output_test_file)

    ###############################################################################
    # Inputs (GPS, Image, LiDAR)
    ###############################################################################
    Initial_labels_train = y_train  # these are same for all modalities
    Initial_labels_val = y_validation

    coord_train = open_npz (data_folder + 'coord/batool/coord_train.npz')
    coord_validation = open_npz (data_folder + 'coord/batool/coord_validation.npz')
    coord_test = open_npz (data_folder + 'coord/batool/coord_test.npz')

    lidar_train = open_npz (data_folder + 'lidar/s008/lidar_train_raymobtime.npz') / 2
    lidar_validation = open_npz (data_folder + 'lidar/s008/lidar_validation_raymobtime.npz') / 2
    lidar_test = open_npz (data_folder + 'lidar/s009/lidar_test_raymobtime.npz') / 2

    if input == 'coord' or input == 'lidar_coord':
        # train
        X_coord_train = coord_train
        # validation
        X_coord_validation = coord_validation
        # test
        X_coord_test = coord_test
        coord_train_input_shape = X_coord_train.shape
        ###############Normalize
        X_coord_train = normalize(X_coord_train, axis=1, norm='l1')
        X_coord_validation = normalize(X_coord_validation, axis=1, norm='l1')
        X_coord_test = normalize(X_coord_test, axis=1, norm='l1')
        ## Reshape for convolutional input
        X_coord_train = X_coord_train.reshape ((X_coord_train.shape[0],
                                                X_coord_train.shape[1], 1))

        X_coord_validation = X_coord_validation.reshape((X_coord_validation.shape[0],
                                                         X_coord_validation.shape[1], 1))
        X_coord_test = X_coord_test.reshape ((X_coord_test.shape[0],
                                              X_coord_test.shape[1], 1))


    if input == 'lidar' or input == 'lidar_coord':
        # train
        X_lidar_train = lidar_train
        # validation
        X_lidar_validation = lidar_validation
        # test
        X_lidar_test = lidar_test
        lidar_train_input_shape = X_lidar_train.shape

    if input == 'coord':
        data_train = [X_coord_train, y_train]
        data_validation = [X_coord_validation, y_validation]
        data_test = [X_coord_test, y_test]
    elif input == 'lidar':
        data_train = [X_lidar_train, y_train]
        data_validation = [X_lidar_validation, y_validation]
        data_test = [X_lidar_test, y_test]
    elif input == 'lidar_coord':
        data_train = [X_coord_train, X_lidar_train, y_train]
        data_validation = [X_coord_validation, X_lidar_validation, y_validation]
        data_test = [X_coord_test, X_lidar_test, y_test]

    return data_train, data_validation, data_test, num_classes

def model_configuration(input, data_train, data_validation, data_test, num_classes):
    if input == 'coord':
        X_coord_train = data_train[0]
        y_train = data_train[1]
        X_coord_validation = data_validation[0]
        y_validation = data_validation[1]
        X_coord_test = data_test[0]
        y_test = data_test[1]
    elif input == 'lidar':
        X_lidar_train = data_train[0]
        y_train = data_train[1]
        X_lidar_validation = data_validation[0]
        y_validation = data_validation[1]
        X_lidar_test = data_test[0]
        y_test = data_test[1]


    elif input == 'lidar_coord':
        input_for_train = data_train[0]
        X_lidar_train = input_for_train[0]
        X_coord_train = input_for_train[1]
        y_train = data_train[1]

        input_for_validation = data_validation[0]
        X_lidar_validation = input_for_validation[0]
        X_coord_validation = input_for_validation[1]
        y_validation = input_for_validation[1]

        input_for_test = data_test[0]
        X_lidar_test = input_for_test[0]
        X_coord_test = input_for_test[1]
        y_test = input_for_test[1]



    ##############################################################################
    # Model configuration
    ##############################################################################
    multimodal = 2 if input == 'lidar_coord' else 1
    fusion = False if multimodal == 1 else True
    train_or_test = 'test'  # default

    # multimodal = False if len(input) == 1 else len(input)
    # fusion = False if len(input) == 1 else True

    modelHand = ModelHandler()
    lr = 0.0001  # default learning rate
    opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    restore_models = False  # default
    strategy = 'one_hot'  # default
    model_folder = 'models/'
    if input == 'coord' or input == 'lidar_coord':
        if restore_models:
            coord_model = load_model_structure (model_folder + 'coord_model.json')
            coord_model.load_weights (model_folder + 'best_weights.coord.h5', by_name=True)
        else:
            coord_model = modelHand.createArchitecture ('coord_mlp',
                                                        num_classes,
                                                        X_coord_train.shape [1],
                                                        #coord_train_input_shape [1],
                                                        'complete', strategy, fusion)
            if not os.path.exists (model_folder + 'coord_model.json'):
                add_model ('coord', coord_model, model_folder)

    if input == 'lidar' or input == 'lidar_coord':
        if restore_models:
            lidar_model = load_model_structure (model_folder + 'lidar_model.json')
            lidar_model.load_weights (model_folder + 'best_weights.lidar.h5', by_name=True)
        else:
            lidar_model = modelHand.createArchitecture ('lidar_marcus', num_classes,
                                                        [X_lidar_train.shape[1], X_lidar_train.shape[2],
                                                         X_lidar_train.shape[3]], 'complete', strategy, fusion)
            if not os.path.exists (model_folder + 'lidar_model.json'):
                add_model ('lidar', lidar_model, model_folder)

    ###############################################################################
    # Fusion Models
    ###############################################################################

    epochs = 20  # default 70
    bs = 32  # default
    shuffle = False  # default

    #lidar_train = X_lidar_train
    #lidar_validation = X_lidar_validation
    #lidar_test = X_lidar_test
    #coord_train = X_coord_train
    #coord_validation = X_coord_validation
    #coord_test = X_coord_test

    if multimodal == 2:
        lidar_train = X_lidar_train
        lidar_validation = X_lidar_validation
        lidar_test = X_lidar_test
        coord_train = X_coord_train
        coord_validation = X_coord_validation
        coord_test = X_coord_test
        # if input_1 == 'coord' and input_1 == 'lidar':
        if input == 'lidar_coord':
            x_train = [lidar_train, coord_train]
            x_validation = [lidar_validation, coord_validation]
            x_test = [lidar_test, coord_test]

            combined_model = concatenate ([lidar_model.output, coord_model.output], name='cont_fusion_coord_lidar')
            z = Reshape ((2, 256)) (combined_model)
            z = Permute ((2, 1), input_shape=(2, 256)) (z)
            z = Conv1D (30, kernel_size=7, strides=1, activation="relu", name='conv1_fusion_coord_lid') (z)
            z = Conv1D (30, kernel_size=5, strides=1, activation="relu", name='conv2_fusion_coord_lid') (z)
            z = BatchNormalization () (z)
            z = MaxPooling1D (name='fusion_coord_lid_maxpool1') (z)

            z = Conv1D (30, kernel_size=7, strides=1, activation="relu", name='conv3_fusion_coord_lid') (z)
            z = Conv1D (30, kernel_size=5, strides=1, activation="relu", name='conv4_fusion_coord_lid') (z)
            z = MaxPooling1D (name='fusion_coord_lid_maxpool2') (z)

            z = Flatten (name='flat_fusion_coord_lid') (z)
            z = Dense (num_classes * 3, activation="relu", use_bias=True, name='dense1_fusion_coord_lid') (z)
            z = Dropout (0.25, name='drop1_fusion_coord_lid') (z)
            z = Dense (num_classes * 2, activation="relu", name='dense2_fusion_coord_lid',
                       kernel_regularizer=regularizers.l1_l2 (l1=1e-5, l2=1e-4)) (z)
            z = Dropout (0.25, name='drop2_fusion_coord_img') (z)
            z = Dense(num_classes, activation="softmax", name='dense3_fusion_coord_lid',
                       kernel_regularizer=regularizers.l1_l2 (l1=1e-5, l2=1e-4)) (z)

            model = Model(inputs=[lidar_model.input, coord_model.input], outputs=z)
            add_model('coord_lidar', model, model_folder)
            model.compile(loss=categorical_crossentropy,
                           optimizer=opt,
                           metrics=[metrics.categorical_accuracy,
                                    top_2_accuracy,
                                    top_5_accuracy,
                                    top_10_accuracy,
                                    top_25_accuracy,
                                    top_50_accuracy])
            model.summary()

    if input == 'coord':
        return coord_model
    elif input == 'lidar':
        return lidar_model
    elif input == 'lidar_coord':
        return model

def train_model(input, model, data_train, data_validation, see_trainning_progress):
    x_train = data_train[0]
    y_train = data_train[1]

    x_validation = data_validation[0]
    y_validation = data_validation[1]

    model_folder = 'models/'
    epochs = 20  # default 70
    bs = 32  # default
    shuffle = False  # default
    train_or_test = 'train'  # default
    strategy = 'one_hot'  # default

    lr = 0.0001  # default learning rate
    opt = Adam (learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if input == 'lidar_coord':
        print ('***************Training Lidar + Coord Model************')
        star_train = time.process_time_ns ()
        hist = model.fit (x_train,
                          y_train,
                          validation_data=(x_validation, y_validation),
                          epochs=epochs,
                          batch_size=bs,
                          callbacks=[tf.keras.callbacks.ModelCheckpoint (model_folder + 'best_weights.coord_lidar.h5',
                                                                         monitor='val_loss',
                                                                         verbose=1, save_best_only=True, mode='auto'),
                                     tf.keras.callbacks.EarlyStopping (monitor='val_loss',
                                                                       patience=25,
                                                                       verbose=2,
                                                                       mode='auto')])
        end_train = time.process_time_ns ()
        train_process_time = (star_train - end_train)
        print ("trainning Time: ", train_process_time)
        print (hist.history.keys ())
        print ('categorical_accuracy', hist.history ['categorical_accuracy'],
               'top_2_accuracy', hist.history ['top_2_accuracy'],
               'top_5_accuracy', hist.history ['top_5_accuracy'],
               'top_10_accuracy', hist.history ['top_10_accuracy'], 'top_25_accuracy', hist.history ['top_25_accuracy'],
               'top_50_accuracy', hist.history ['top_50_accuracy'],
               'val_categorical_accuracy', hist.history ['val_categorical_accuracy'],
               'val_top_2_accuracy', hist.history ['val_top_2_accuracy'],
               'val_top_5_accuracy', hist.history ['val_top_5_accuracy'],
               'val_top_10_accuracy', hist.history ['val_top_10_accuracy'],
               'val_top_25_accuracy', hist.history ['val_top_25_accuracy'],
               'val_top_50_accuracy', hist.history ['val_top_50_accuracy'])

    elif input == 'coord':
        X_coord_train = data_train[0]
        X_coord_validation = data_validation[0]

        if strategy == 'reg':
            #model = coord_model
            model.compile(loss="mse",
                          optimizer=opt,
                          metrics=[top_1_accuracy,
                                   top_2_accuracy,
                                   top_10_accuracy,
                                   top_50_accuracy,
                                   R2_metric])
            model.summary()
            if train_or_test == 'train':
                print('***************Training************')
                star_trainning = time.process_time_ns ()
                hist = model.fit(X_coord_train,
                                 y_train,
                                 validation_data=(X_coord_validation, y_validation),
                                epochs=epochs, batch_size=bs, shuffle=shuffle)
                end_trainning = time.process_time_ns ()
                trainning_process_time = (end_trainning - star_trainning)
                print('losses in train:', hist.history['loss'])

        if strategy == 'one_hot':
            #print ('All shapes', X_coord_train.shape, y_train.shape, X_coord_validation.shape, y_validation.shape)
            #       X_coord_test.shape, y_test.shape)
            #model = coord_model
            model.compile(loss=categorical_crossentropy,
                          optimizer=opt,
                          metrics=[metrics.categorical_accuracy,
                                   top_2_accuracy,
                                   top_5_accuracy,
                                   top_10_accuracy,
                                   top_25_accuracy,
                                   top_50_accuracy,
                                   precision_m,
                                   recall_m,
                                   f1_m])
            #model.summary()

            call_backs = []
            if train_or_test == 'train':
                if see_trainning_progress != 0:
                    print('***************Training************')
                star_trainning = time.process_time_ns ()

                hist = model.fit(X_coord_train, y_train,
                                 validation_data=(X_coord_validation, y_validation),
                                 epochs=epochs, batch_size=bs, shuffle=shuffle, verbose=see_trainning_progress,
                                 callbacks=[tf.keras.callbacks.ModelCheckpoint(model_folder+'best_weights.coord.h5',
                                                                               monitor='val_loss', verbose=see_trainning_progress,
                                                                               save_best_only=True, mode='auto'),
                                            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                             patience=15, verbose=0, mode='auto')])

                end_trainning = time.process_time_ns()
                trainning_process_time = (end_trainning - star_trainning)
                #print("trainning Time: ", trainning_process_time)
                '''
                print(hist.history.keys())
                print('val_loss',hist.history['val_loss'])
                print('categorical_accuracy', hist.history['categorical_accuracy'],
                      'top_2_accuracy',hist.history['top_2_accuracy'],
                      'top_5_accuracy',hist.history['top_5_accuracy'],
                      'top_10_accuracy', hist.history['top_10_accuracy'],
                      'top_25_accuracy',hist.history['top_25_accuracy'],
                      'top_50_accuracy',hist.history['top_50_accuracy'],
                      'val_categorical_accuracy',hist.history['val_categorical_accuracy'],
                      'val_top_2_accuracy',hist.history['val_top_2_accuracy'],
                      'val_top_5_accuracy',hist.history['val_top_5_accuracy'],
                      'val_top_10_accuracy',hist.history['val_top_10_accuracy'],
                      'val_top_25_accuracy',hist.history['val_top_25_accuracy'],
                      'val_top_50_accuracy',hist.history['val_top_50_accuracy'])
                '''

    elif input == 'lidar':
        X_lidar_train = data_train[0]
        X_lidar_validation = data_validation[0]

        if strategy == 'reg':
            #model = lidar_model
            model.compile(loss="mse", optimizer=opt, metrics=[top_1_accuracy,
                                                              top_2_accuracy,
                                                              top_10_accuracy,
                                                              top_50_accuracy,
                                                              R2_metric])
            model.summary()
            if train_or_test == 'train':
                print('***************Training************')
                star_trainning = time.process_time_ns ()
                hist = model.fit(X_lidar_train,
                                 y_train,
                                 validation_data=(X_lidar_validation, y_validation),
                                 epochs=epochs,
                                 batch_size=bs,
                                 shuffle=shuffle)
                end_trainning = time.process_time_ns()
                trainning_process_time = (end_trainning - star_trainning)
                print('losses in train:', hist.history['loss'])

        if strategy == 'one_hot':
            print('All shapes', X_lidar_train.shape, y_train.shape, X_lidar_validation.shape, y_validation.shape)
            #,X_lidar_test.shape,y_test.shape)
            #model = lidar_model
            model.compile(loss=categorical_crossentropy,
                          optimizer=opt,
                          metrics=[metrics.categorical_accuracy,
                                   top_2_accuracy,
                                   top_5_accuracy,
                                   top_10_accuracy,
                                   top_25_accuracy,
                                   top_50_accuracy,
                                   precision_m, recall_m, f1_m])
            model.summary()
            if train_or_test == 'train':
                print('***************Training************')
                star_trainning = time.process_time_ns ()
                hist = model.fit(X_lidar_train,
                                 y_train,
                                 validation_data=(X_lidar_validation, y_validation),
                                 epochs=epochs,
                                 batch_size=bs,
                                 shuffle=shuffle,
                                 callbacks=[tf.keras.callbacks.ModelCheckpoint(model_folder+'best_weights.lidar.h5',
                                                                               monitor='val_loss',
                                                                               verbose=2,
                                                                               save_best_only=True,mode='auto'),
                                            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                             patience=15,
                                                                             verbose=2,
                                                                             mode='auto')])
                end_trainning = time.process_time_ns()
                trainning_process_time = (end_trainning - star_trainning)
                #print("trainning Time: ", trainning_process_time)
                print(hist.history.keys())
                print('loss',hist.history['loss'],
                      'val_loss',hist.history['val_loss'],
                      'categorical_accuracy', hist.history['categorical_accuracy'],
                      'top_2_accuracy',hist.history['top_2_accuracy'],
                      'top_5_accuracy',hist.history['top_5_accuracy'],
                      'top_10_accuracy', hist.history['top_10_accuracy'],
                      'top_25_accuracy',hist.history['top_25_accuracy'],
                      'top_50_accuracy',hist.history['top_50_accuracy'],
                      'val_categorical_accuracy',hist.history['val_categorical_accuracy'],
                      'val_top_2_accuracy',hist.history['val_top_2_accuracy'],
                      'val_top_5_accuracy',hist.history['val_top_5_accuracy'],
                      'val_top_10_accuracy',hist.history['val_top_10_accuracy'],
                      'val_top_25_accuracy',hist.history['val_top_25_accuracy'],
                      'val_top_50_accuracy',hist.history['val_top_50_accuracy'])

    if see_trainning_progress != 0:
        print("trainning Time: ", trainning_process_time)
    return trainning_process_time, data_train[0].shape

def test_model(input, model, data_test, top_k, see_trainning_progress):
    #    x_test = [data_test[1], data_test[0]]
    #    y_test = data_test[2]


    lr = 0.0001  # default learning rate
    opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    strategy = 'one_hot'  # default

    model_folder = 'models/'

    if input == 'lidar_coord':
        x_test = [data_test[1], data_test[0]]
        y_test = data_test[2]
        print ('***************Testing************')
        model.load_weights (model_folder + 'best_weights.coord_lidar.h5', by_name=True)
        scores = model.evaluate (x_test, y_test)
        print ("----------------------------------")
        print ("Test results:", model.metrics_names, scores)
        print ("----------------------------------")
        print ("----------------------------------")

        # top_k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
        #top_k = np.arange (1, 51, 1)
        accuracy_top_k = []
        process_time = []
        for i in range(len(top_k)):
            model_metrics = [metrics.CategoricalAccuracy (), metrics.TopKCategoricalAccuracy (k=top_k [i])]
            model.compile (loss=categorical_crossentropy, optimizer=opt, metrics=model_metrics)
            model.load_weights (model_folder + 'best_weights.coord_lidar.h5', by_name=True)

            ### Testing
            star_test = time.process_time_ns ()
            out = model.evaluate (x_test, y_test)
            end_test = time.process_time_ns ()
            delta_time = end_test - star_test
            accuracy_top_k.append(out[2])
            process_time.append(delta_time)

        print ("----------------------------------")
        print ('top-k = ', top_k)
        print ("Acuracy =", accuracy_top_k)
        print ("Process test time =", process_time)
        print ("----------------------------------")

        '''
        all_index_predict = (model.predict (x_test, verbose=1))
        all_index_predict_order = np.zeros ((all_index_predict.shape [0], all_index_predict.shape [1]))
        for i in range (len (all_index_predict)):
            all_index_predict_order [i] = np.flip (np.argsort (all_index_predict [i]))

        ## Testanto  a acuracia calculada pelo metodo de avaliacao do keras (evaluate)
        top_1_predict = all_index_predict_order [:, 0].astype (int)
        true_label = []
        for i in range (len (y_test)):
            true_label.append (y_test [i, :].argmax ())

        acerto = 0
        nao_acerto = 0

        for sample in range (len (y_test)):
            if (true_label [sample] == top_1_predict [sample]):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        score = acerto / len (all_index_predict)
        print ('score top-1: ', score)
        '''

    elif input == 'lidar':
        X_lidar_test = data_test[0]
        y_test = data_test[1]

        if strategy == 'one_hot':
            model.compile(loss=categorical_crossentropy,
                          optimizer=opt,
                          metrics=[metrics.categorical_accuracy,
                                   top_2_accuracy,
                                   top_5_accuracy,
                                   top_10_accuracy,
                                   top_25_accuracy,
                                   top_50_accuracy,
                                   precision_m, recall_m, f1_m])
            model.summary()


            print ('***************Testing************')
            model.load_weights (model_folder + 'best_weights.lidar.h5', by_name=True)  # to be added
            scores = model.evaluate (X_lidar_test, y_test)
            print ("############ ----------------------------- ")

            print ("Test results", model.metrics_names, scores)

            #top_k = [1, 5, 10]
            # top_k = np.arange (1, 51, 1)
            accuracy_top_k = []
            process_time = []
            index_predict = []
            for i in range (len (top_k)):
                model_metrics = [metrics.CategoricalAccuracy (), metrics.TopKCategoricalAccuracy (k=top_k [i])]
                model.compile (loss=categorical_crossentropy, optimizer=opt, metrics=model_metrics)
                # model.load_weights (model_folder + 'best_weights.coord.h5', by_name=True)

                ### Testing
                star_test = time.process_time_ns ()
                out = model.evaluate (X_lidar_test, y_test)
                end_test = time.process_time_ns ()
                delta = end_test - star_test
                accuracy_top_k.append (out [2])
                process_time.append (delta)
                print ("top-k: ", top_k [i])

            print ('top-k = ', top_k)
            print ("Acuracy =", accuracy_top_k)
            print ("process time: ", process_time)

            '''
            all_index_predict = (model.predict (X_lidar_test, verbose=1))
            all_index_predict_order = np.zeros ((all_index_predict.shape [0], all_index_predict.shape [1]))
            for i in range (len (all_index_predict)):
                all_index_predict_order [i] = np.flip (np.argsort (all_index_predict [i]))

            ## Testanto  a acuracia calculada pelo metodo de avaliacao do keras (evaluate)
            top_1_predict = all_index_predict_order [:, 0].astype (int)
            true_label = []
            for i in range (len (y_test)):
                true_label.append (y_test [i, :].argmax ())

            acerto = 0
            nao_acerto = 0

            for sample in range (len (y_test)):
                if (true_label [sample] == top_1_predict [sample]):
                    acerto = acerto + 1
                else:
                    nao_acerto = nao_acerto + 1

            score = acerto / len (all_index_predict)
            print ('score top-1: ', score)
            '''

    elif input == 'coord':
        X_coord_test = data_test[0]
        y_test = np.array(data_test[1])

        if strategy == 'one_hot':
            #print ('All shapes', X_coord_train.shape, y_train.shape, X_coord_validation.shape, y_validation.shape)
            #       X_coord_test.shape, y_test.shape)
            #model = coord_model
            '''
            model.compile(loss=categorical_crossentropy,
                          optimizer=opt,
                          metrics=[metrics.categorical_accuracy,
                                   top_2_accuracy,
                                   top_5_accuracy,
                                   top_10_accuracy,
                                   top_25_accuracy,
                                   top_50_accuracy,
                                   precision_m,
                                   recall_m,
                                   f1_m])
            '''
            #model.summary()
            call_backs = []
            if see_trainning_progress != 0:
                print('***************Testing************')
            model.load_weights(model_folder + 'best_weights.coord.h5', by_name=True)  # to be added
            #scores = model.evaluate (X_coord_test, y_test)
            #print ("############ ----------------------------- ")

            #print ("Test results", model.metrics_names, scores)

            # top_k = [1, 5, 10]
            #top_k = np.arange (1, 51, 1)
            accuracy_top_k = []
            process_time = []
            index_predict = []
            for i in range(len(top_k)):
                model_metrics = [metrics.CategoricalAccuracy(), metrics.TopKCategoricalAccuracy(k=top_k[i])]
                model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=model_metrics)
                # model.load_weights (model_folder + 'best_weights.coord.h5', by_name=True)

                ### Testing
                star_test = time.process_time_ns()
                out = model.evaluate(X_coord_test, y_test, verbose=see_trainning_progress)
                end_test = time.process_time_ns()
                delta = end_test - star_test
                accuracy_top_k.append(out[2])
                process_time.append(delta)

            if see_trainning_progress != 0:
                print('top-k = ', top_k)
                print("Acuracy =", accuracy_top_k)
                print("process time: ", process_time)

            '''
            print ('usando o metodo predict: ')
            all_index_predict = (model.predict (X_coord_test, verbose=1))
            all_index_predict_order = np.zeros ((all_index_predict.shape [0], all_index_predict.shape [1]))
            print (' ordenando as predicoes: ')
            for i in range (len (all_index_predict)):
                all_index_predict_order [i] = np.flip (np.argsort (all_index_predict [i]))

            ## Testanto  a acuracia calculada pelo metodo de avaliacao do keras (evaluate)
            top_1_predict = all_index_predict_order [:, 0].astype (int)
            true_label = []
            for i in range (len (y_test)):
                true_label.append (y_test [i, :].argmax ())

            acerto = 0
            nao_acerto = 0

            for sample in range (len (y_test)):
                if (true_label [sample] == top_1_predict [sample]):
                    acerto = acerto + 1
                else:
                    nao_acerto = nao_acerto + 1

            score = acerto / len (all_index_predict)
            print ('score top-1: ', score)
            '''

    df_results_top_k = pd.DataFrame({"top-k": top_k,
                                     "score": accuracy_top_k,
                                     "test_time": process_time,
                                     "samples_tested": y_test.shape[0]})
    return df_results_top_k

def beam_selection_Batool(input,
                          data_train,
                          data_validation,
                          data_test,
                          num_classes,
                          episode,
                          see_trainning_progress):



    model = model_configuration(input, data_train, data_validation, data_test, num_classes)
    trainning_process_time, samples_shape = train_model(input, model, data_train, data_validation, see_trainning_progress)
    top_k = [1, 5, 10, 15, 20, 25, 30]
    df_results_top_k = test_model(input, model, data_test, top_k, see_trainning_progress)

    df_results_top_k['episode'] = episode
    df_results_top_k['trainning_process_time'] = trainning_process_time
    df_results_top_k['samples_trainning'] = samples_shape[0]

    return df_results_top_k


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



    data_for_train = pd.DataFrame({"EpisodeID": train_data['EpisodeID'],
                                   "x": coord_train[:, 0],
                                   "y": coord_train[:, 1],
                                   "beam": y_train.tolist(),
                                   "lidar": lidar_train_reshaped.tolist()})

    #para recuperar a matriz original do lidar
    #lidar = np.array(data_for_train ["lidar"].tolist()).reshape(9234,20,200,10)
    #a = np.array_equal (lidar_train, lidar)
    #data_for_train["lidar"] = lidar_train.tolist ()


    data_for_validation = pd.DataFrame({"EpisodeID": validation_data['EpisodeID'],
                                        "x": coord_validation[:, 0],
                                        "y": coord_validation[:, 1]})
    data_for_validation["lidar"] = lidar_validation_reshaped.tolist()
    data_for_validation["beam"] = y_validation.tolist()

    filename = '../../data/coord/CoordVehiclesRxPerScene_s009.csv'
    all_csv_data = pd.read_csv(filename)
    valid_data = all_csv_data[all_csv_data['Val'] == 'V']

    coord_for_test = np.zeros((len(valid_data), 2))
    coord_for_test[:, 0] = valid_data['x']
    coord_for_test[:, 1] = valid_data['y']
    coord_test = normalize(coord_for_test, axis=1, norm='l1')

    data_for_test = pd.DataFrame({"EpisodeID": valid_data['EpisodeID'],
                                    "x": coord_test[:, 0],
                                    "y": coord_test[:, 1]})
    data_for_test["lidar"] = lidar_test_reshaped.tolist()
    data_for_test["beam"] = y_test.tolist()

    '''
    input = 'coord'
    data_train, data_validation, data_test, num_classes = prepare_data (input='coord')
    if input == 'coord':
        dataTrain = data_train[0]
        indexTrain = data_train[1]

        TrainData = pd.DataFrame({"EpisodeID": train_data['EpisodeID'],
                                  "coord": dataTrain})
    '''
    return data_for_train, data_for_validation, data_for_test, num_classes


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

def prepare_coord_for_trainning(data, samples):
    coord_x = np.vstack(data['x'].tolist())
    coord_y = np.vstack(data['y'].tolist())
    coord = np.concatenate((coord_x, coord_y), axis=1).reshape(samples,2,1)

    return coord

def prepare_coord_for_test(data, episodio_for_test):
    coord_x = np.vstack(data[data['EpisodeID'] == episodio_for_test]['x'].tolist())
    coord_y = np.vstack(data[data['EpisodeID'] == episodio_for_test]['y'].tolist())
    coord = np.concatenate((coord_x, coord_y), axis=1).reshape(len(coord_x),2,1)
    return coord

def fit_fixed_window_top_k(label_input_type, nro_of_episodes_for_test):
    #data_train, data_validation, data_test, num_classes = prepare_data(input)
    #y_train, y_validation, y_test, num_classes = get_index_beams()

    nro_of_episodes = nro_of_episodes_for_test
    episodes_for_train = 2086
    see_trainning_progress = 0 # 0: no,
                               # 1: yes (how you an animated progress bar)
                               # 2: yes (will just mention the number of epochs completed)

    data_for_train, data_for_validation, s009_data, num_classes = read_all_data()

    episode_for_test = np.arange(0, nro_of_episodes, 1)

    label_train = np.array(data_for_train['beam'].tolist())
    label_validation = np.array(data_for_validation['beam'].tolist())
    if label_input_type == 'coord':
        input_train = prepare_coord_for_trainning(data_for_train, 9234)
        input_validation = prepare_coord_for_trainning(data_for_validation, 1960)

    elif label_input_type == 'lidar':
        print('Beam selection using ' + input)
        input_train = np.array(data_for_train["lidar"].tolist()).reshape(9234, 20, 200, 10)
        input_validation = np.array(data_for_validation["lidar"].tolist()).reshape(1960, 20, 200, 10)

    elif label_input_type == 'lidar_coord':

        coord_train = prepare_coord_for_trainning(data_for_train, 9234)
        coord_validation = prepare_coord_for_trainning(data_for_validation, 1960)

        lidar_train = np.array(data_for_train["lidar"].tolist()).reshape(9234, 20, 200, 10)
        lidar_validation = np.array(data_for_validation["lidar"].tolist()).reshape(1960, 20, 200, 10)

        input_train = [lidar_train, coord_train]
        input_validation = [lidar_validation, coord_validation]

    df_all_results_top_k = pd.DataFrame()

    print('Episode: ', end=' ', flush=True)
    for i in range(len(episode_for_test)):
        if i in s009_data['EpisodeID'].tolist():
            label_test = s009_data[s009_data['EpisodeID'] == i]['beam'].tolist()
            if label_input_type == 'coord':
                input_test = prepare_coord_for_test(data=s009_data, episodio_for_test=i)

            elif label_input_type == 'lidar':
                input_test = np.array(s009_data[s009_data['EpisodeID'] == i]['lidar'].tolist()).reshape(len(label_test),
                                                                                                        20, 200, 10)
            elif label_input_type == 'lidar_coord':
                coord_test = prepare_coord_for_test(data=s009_data, episodio_for_test=i)
                lidar_test = np.array(s009_data[s009_data['EpisodeID'] == i]['lidar'].tolist()).reshape(len(label_test),
                                                                                                        20, 200, 10)
                input_test = [lidar_test, coord_test]

            print(i, end=' ', flush=True)
            df_results_top_k = beam_selection_Batool(input=label_input_type,
                                                     data_train=[input_train, label_train],
                                                     data_validation=[input_validation, label_validation],
                                                     data_test=[input_test, label_test],
                                                     num_classes=num_classes,
                                                     episode=i,
                                                     see_trainning_progress= see_trainning_progress)
            df_all_results_top_k = pd.concat([df_all_results_top_k, df_results_top_k], ignore_index=True)

            a=0
            b=0
    path_result = ('../../results/score/Batool/online/top_k/') + label_input_type + '/fixed_window/'
    df_all_results_top_k.to_csv(path_result + 'scores_with_fixed_window_top_k.csv', index=False)

def fit_sliding_window_top_k(label_input_type,
                             nro_of_episodes,
                             window_size,
                             s008_data,
                             s009_data):

    data_for_train, data_for_validation, s009_data, num_classes = read_all_data ()

    episode_for_test = np.arange(0, nro_of_episodes, 1)
    start_index_s009 = 0
    nro_episodes_s008 = 2085

    for i in range(len(episode_for_test)):
    #for i in tqdm(range(len(episode_for_test))):
        if i in s009_data['Episode'].tolist():
            if i == 0:
                start_index_s008 = nro_episodes_s008 - window_size
                input_train, label_train = extract_training_data_from_s008(s008_data, start_index_s008, label_input_type)
                input_test, label_test = extract_test_data_from_s009(i, label_input_type, s009_data)
            else:
                start_index_s008 = (nro_episodes_s008 - window_size)+i
                if start_index_s008 < nro_episodes_s008:
                    start_index_s009 = 0
                    end_index_s009 = window_size - (nro_episodes_s008 - start_index_s008)

                    input_train_s008, label_train_s008 = extract_training_data_from_s008(s008_data,
                                                                                         start_index_s008,
                                                                                         label_input_type)
                    input_train_s009, label_train_s009 = extract_training_data_from_s009(s009_data,
                                                                                         start_index_s009,
                                                                                         end_index_s009,
                                                                                         label_input_type)
                    input_train = input_train_s008 + input_train_s009
                    label_train = label_train_s008 + label_train_s009

                    input_test, label_test = extract_test_data_from_s009(i, label_input_type, s009_data)

                else:
                    end_index_s009 = start_index_s009 + window_size
                    input_train, label_train = extract_training_data_from_s009(s009_data,
                                                                               start_index_s009,
                                                                               end_index_s009,
                                                                               label_input_type)
                    input_test, label_test = extract_test_data_from_s009(i, label_input_type, s009_data)
                    start_index_s009 += 1



def extract_training_data_from_s008(s008_data, start_index, input_type):
    initial_data_for_trainning = s008_data [s008_data ['Episode'] > start_index]
    label_train = initial_data_for_trainning ['index_beams'].tolist ()
    input_train = []

    if input_type == 'coord':
        input_train = initial_data_for_trainning ['encoding_coord'].tolist ()
    elif input_type == 'lidar':
        input_train = initial_data_for_trainning ['lidar'].tolist ()
    elif input_type == 'lidar_coord':
        input_train = initial_data_for_trainning ['lidar_coord'].tolist ()
    else:
        print('error: deve especificar o tipo de entrada')

    return input_train, label_train
def extract_training_data_from_s009(s009_data, start_index, end_index, input_type):
    data_for_trainnig = s009_data.loc[(s009_data['Episode'] >= start_index) & (s009_data['Episode'] < end_index)]

    label_train = data_for_trainnig['index_beams'].tolist()

    input_train = []
    if input_type == 'coord':
        input_train = data_for_trainnig['encoding_coord'].tolist()
    elif input_type == 'lidar':
        input_train = data_for_trainnig['lidar'].tolist()
    elif input_type == 'lidar_coord':
        input_train = data_for_trainnig['lidar_coord'].tolist()

    return input_train, label_train
def extract_test_data_from_s009(episode, label_input_type, s009_data):
    label_test = s009_data [s009_data ['Episode'] == episode] ['index_beams'].tolist ()

    input_test = []

    if label_input_type == 'coord':
        input_test = s009_data [s009_data ['Episode'] == episode] ['encoding_coord'].tolist ()
    elif label_input_type == 'lidar':
        input_test = s009_data [s009_data ['Episode'] == episode] ['lidar'].tolist ()
    elif label_input_type == 'lidar_coord':
        input_test = s009_data [s009_data ['Episode'] == episode] ['lidar_coord'].tolist ()
    else:
        print ('error: deve especificar o tipo de entrada')

    return input_test, label_test




def calculate_mean_score(data):
    #all_score = data ['Score'].tolist ()
    average_score = []
    for i in range(len(data)):
        i = i + 1
        average_score.append(np.mean(data[0:i]))
    return average_score
def plot_results():
    filename = '../../results/score/Batool/online/top_k/coord/fixed_window/scores_with_fixed_window_top_k.csv'
    all_csv_data = pd.read_csv(filename)

    metric = 'score' #'time_trainning'
    top_k = [1, 5, 10, 15, 20, 25, 30]
    color = ['blue', 'red', 'green', 'purple', 'orange', 'maroon', 'teal']  # 'maroon', 'teal', 'black', 'gray', 'brown', 'cyan', 'magenta', 'yellow', 'olive', 'navy', 'lime', 'aqua', 'fuchsia', 'silver', 'white']

    if metric == 'score':
        pos_x = [10, 50, 100, 150, 200, 250, 300]
        for i in range(len(top_k)):
            top_1 = all_csv_data[all_csv_data['top-k'] == top_k[i]]
            all_score_top_1 = top_1['score']
            mean_accum_top_1 = calculate_mean_score(all_score_top_1)

            plt.plot(top_1['episode'], mean_accum_top_1, '.', color=color[i],
                     label='Top-'+str(top_k[i]))
            plt.text(pos_x[i], 0.3,
                     str(np.round(np.mean(mean_accum_top_1), 3)),
                     fontsize=8, color=color[i])

        plt.xlabel ('Episode', fontsize=12, fontweight='bold', fontname='Myanmar Sangam MN')
        plt.ylabel ('Accumulative score', fontsize=12, fontweight='bold', fontname='Myanmar Sangam MN')
        plt.title ('Beam Selection using MPL with Coord \n Reference: Batool',
                   fontsize=14, fontweight='bold', fontname='Myanmar Sangam MN')
        plt.legend(ncol=3, loc='lower right')
        plt.show()

    elif metric == 'time_trainning':
        top_1 = all_csv_data[all_csv_data['top-k'] == 1]
        trainning_time = top_1['trainning_process_time']*1e-9
        plt.plot(top_1['episode'], trainning_time, marker=',', label='fixed window')
        plt.text (200, 80, 'Mean: '+str(np.round(np.mean(trainning_time), 3)),
                  fontsize=12)
        plt.xlabel('Episode', fontsize=12, fontweight='bold', fontname='Myanmar Sangam MN')
        plt.ylabel('Trainning Time [s]', fontsize=12, fontweight='bold', fontname='Myanmar Sangam MN')
        plt.title('Beam Selection using MPL with Coord \n Reference: Batool',
                  fontsize=14, fontweight='bold', fontname='Myanmar Sangam MN')
        plt.legend()
        plt.show()

def main():

    print("+-------------------------------------"
          "\n|    online learning - Batool "
          "\n|          Fixed window"
          "\n+----------------------------------")
    input = 'coord'
    top_k = [1, 5, 10, 15, 20, 25, 30]


    #print(tf.__file__)
    #import keras
    #print(keras.__version__)



    fit_fixed_window_top_k(input, nro_of_episodes_for_test=5)

    #plot_results()

main()


