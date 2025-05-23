import os
import time
import random

import matplotlib.pyplot as plt
import pandas as pd
import warnings
import seaborn as sns


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

def model_configuration(input, data_train, data_validation, data_test, num_classes, restore_models):
    #quando nao usar o modelo saltitante tirar a variavel restore_models dos parametros,
    # e descomentar a linha 212
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

    #restore_models = False  # default
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
            #model.summary()

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
        print('***************Training Lidar + Coord Model************')
        star_train = time.process_time_ns ()
        hist = model.fit (x_train,
                          y_train,
                          validation_data=(x_validation, y_validation),
                          epochs=epochs,
                          batch_size=bs,
                          verbose=0,
                          callbacks=[tf.keras.callbacks.ModelCheckpoint (model_folder + 'best_weights.coord_lidar.h5',
                                                                         monitor='val_loss',
                                                                         verbose=0, save_best_only=True, mode='auto'),
                                     tf.keras.callbacks.EarlyStopping (monitor='val_loss',
                                                                       patience=25,
                                                                       verbose=0,
                                                                       mode='auto')])
        end_train = time.process_time_ns ()
        trainning_process_time = (end_train- star_train)
        if see_trainning_progress != 0:
            print ("trainning Time: ", trainning_process_time)
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

            #model.summary()
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
            #model.summary()
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
                if see_trainning_progress != 0:
                    print('losses in train:', hist.history['loss'])

        if strategy == 'one_hot':
            if see_trainning_progress != 0:
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
             #model.summary()
            if train_or_test == 'train':
                if see_trainning_progress != 0:
                    print('***************Training************')
                star_trainning = time.process_time_ns ()
                hist = model.fit(X_lidar_train,
                                 y_train,
                                 validation_data=(X_lidar_validation, y_validation),
                                 epochs=epochs,
                                 batch_size=bs,
                                 shuffle=shuffle,
                                 verbose=0,
                                 callbacks=[tf.keras.callbacks.ModelCheckpoint(model_folder+'best_weights.lidar.h5',
                                                                               monitor='val_loss',
                                                                               verbose=0, #2,
                                                                               save_best_only=True,mode='auto'),
                                            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                             patience=15,
                                                                             verbose=0, #2,
                                                                             mode='auto')])
                end_trainning = time.process_time_ns()
                trainning_process_time = (end_trainning - star_trainning)
                if see_trainning_progress != 0:
                    print("trainning Time: ", trainning_process_time)
                    print(hist.history.keys())
                if see_trainning_progress != 0:
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
    if input == 'lidar_coord':
        return trainning_process_time, data_train[0][0].shape
    else:
        return trainning_process_time, data_train[0].shape

def test_model(input, model, data_test, top_k, see_trainning_progress):
    #    x_test = [data_test[1], data_test[0]]
    #    y_test = data_test[2]


    lr = 0.0001  # default learning rate
    opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    strategy = 'one_hot'  # default

    model_folder = 'models/'

    if input == 'lidar_coord':
        x_test = [data_test[0][0], data_test[0][1]]
        y_test = data_test[1]
        print ('***************Testing************')
        model.load_weights (model_folder + 'best_weights.coord_lidar.h5', by_name=True)
        scores = model.evaluate (x_test, y_test, verbose=0)
        if see_trainning_progress != 0:
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
            out = model.evaluate (x_test, y_test, verbose=0)
            end_test = time.process_time_ns ()
            delta_time = end_test - star_test
            accuracy_top_k.append(out[2])
            process_time.append(delta_time)

        if see_trainning_progress != 0:
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
            #model.summary()

            if see_trainning_progress != 0:
                print ('***************Testing************')
            model.load_weights (model_folder + 'best_weights.lidar.h5', by_name=True)  # to be added
            scores = model.evaluate (X_lidar_test, y_test, verbose=0)
            #print ("############ ----------------------------- ")

            #print ("Test results", model.metrics_names, scores)

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
                out = model.evaluate (X_lidar_test, y_test, verbose=0)
                end_test = time.process_time_ns ()
                delta = end_test - star_test
                accuracy_top_k.append (out [2])
                process_time.append (delta)
                #print ("top-k: ", top_k [i])

            if see_trainning_progress != 0:
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
                          see_trainning_progress,
                          restore_models,
                          flag_fast_experiment=False):


    if flag_fast_experiment:
        print('entre')

        if episode == 0 or episode == 200 or episode == 400 or episode == 600 or episode == 800  or episode == 1000  or episode == 1200 \
                or episode == 1400 or episode == 1600 or episode == 1800:
            '''
            print('train - test', episode)
            print('data train: ', np.shape(data_train[0][0])[0], 'data test', np.shape(data_test[0][0])[0])
            df_results_top_k['episode'] = episode
            df_results_top_k['trainning_process_time'] = 0
            df_results_top_k['samples_trainning'] = np.shape(data_train[0][0])[0]
            df_results_top_k['samples_test'] = np.shape(data_test[0][0])[0]
            '''
            model = model_configuration (input, data_train, data_validation, data_test, num_classes, restore_models)
            trainning_process_time, samples_shape = train_model (input, model,
                                                                 data_train, data_validation,
                                                                 see_trainning_progress)
            #top_k = [1, 5, 10, 15, 20, 25, 30]
            top_k = np.arange (1, 31, 1)
            df_results_top_k = test_model (input, model, data_test, top_k, see_trainning_progress)

        else:
            '''
            print('just test' , episode)
            print ('data test', np.shape(data_test[0][0])[0])
            df_results_top_k ['episode'] = episode
            df_results_top_k ['trainning_process_time'] = 0
            df_results_top_k ['samples_trainning'] =0
            df_results_top_k ['samples_test'] = np.shape(data_test[0][0])[0]
            '''
            restore_models = False
            model = model_configuration (input, data_train, data_validation,
                                         data_test, num_classes, restore_models)
            trainning_process_time = 0
            samples_shape = [0, 0, 0]
            top_k = [1, 5, 10, 15, 20, 25, 30]
            df_results_top_k = test_model (input, model, data_test, top_k, see_trainning_progress)




    else:
        print('No entre')
        restore_models = False
        top_k = np.arange (1, 31, 1)
        #print(top_k)
        model = model_configuration(input, data_train, data_validation, data_test, num_classes, restore_models)
        trainning_process_time, samples_shape = train_model (input, model,
                                                             data_train, data_validation,
                                                             see_trainning_progress)
        #top_k = np.arange(1, 31, 1)
        df_results_top_k = test_model(input, model, data_test, top_k, see_trainning_progress)

    df_results_top_k['episode'] = episode
    df_results_top_k['trainning_process_time'] = trainning_process_time
    df_results_top_k['samples_trainning'] = samples_shape[0]

    return df_results_top_k

def beam_selection_Batool_for_fit_jumpy(input,
                          data_train,
                          data_validation,
                          data_test,
                          num_classes,
                          episode,
                          see_trainning_progress, flag_fit_jumpy):


    if flag_fit_jumpy:
        restore_models = False
        model = model_configuration(input, data_train, data_validation, data_test, num_classes, restore_models)
        trainning_process_time, samples_shape = train_model(input, model, data_train, data_validation, see_trainning_progress)
        top_k = [1, 5, 10, 15, 20, 25, 30]
        df_results_top_k = test_model(input, model, data_test, top_k, see_trainning_progress)

    else:
        restore_models = False
        model = model_configuration(input, data_train, data_validation, data_test, num_classes, restore_models)
        trainning_process_time = 0
        samples_shape = [0,0,0]
        top_k = [1, 5, 10, 15, 20, 25, 30]
        df_results_top_k = test_model (input, model, data_test, top_k, see_trainning_progress)


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



    data_for_train = pd.DataFrame({"Episode": train_data['EpisodeID'],
                                   "coord": coord_train.tolist(),
                                   "x": coord_train[:, 0],
                                   "y": coord_train[:, 1],
                                   "LOS": train_data['LOS'],
                                   "index_beams": y_train.tolist(),
                                   "lidar": lidar_train_reshaped.tolist()})

    #para recuperar a matriz original do lidar
    #lidar = np.array(data_for_train ["lidar"].tolist()).reshape(9234,20,200,10)
    #a = np.array_equal (lidar_train, lidar)
    #data_for_train["lidar"] = lidar_train.tolist ()


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


def fit_fixed_window_top_k(label_input_type, episodes_for_test):
    import sys
    import os

    # Adiciona o caminho do diretório do arquivo que você quer importar
    # sys.path.append(os.path.abspath("../"))
    sys.path.append ("../")

    # Agora é possível importar o arquivo como um módulo
    import tools as tls

    data_for_train, data_for_validation, s009_data, num_classes = read_all_data ()
    all_dataset_s008 = pd.concat ([data_for_train, data_for_validation], axis=0)

    episode_for_test = np.arange(0, episodes_for_test, 1)
    see_trainning_progress = 0
    nro_episodes_s008 = 2086  # from 0 to 1564

    start_index_s008 = 0
    input_for_train, label_for_train = tls.extract_training_data_from_s008_sliding_window (all_dataset_s008,
                                                                                           start_index_s008,
                                                                                           label_input_type)
    label_train, label_validation = sliding_prepare_label_for_trainning (label_for_train)

    df_all_results_top_k = pd.DataFrame()
    for i in range(len(episode_for_test)):
        # i=101
        if i in s009_data['Episode'].tolist ():
            input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (i,
                                                                                             label_input_type,
                                                                                             s009_data)

            label_test = np.array (label_for_test)

            if label_input_type == 'coord':
                input_train, input_validation = sliding_prepare_coord_for_trainning (input_for_train)
                input_test = np.array (input_for_test).reshape (len (input_for_test), 2, 1)
            if label_input_type == 'lidar':
                input_train, input_validation = sliding_prepare_lidar_for_trainning (input_for_train)
                input_test = np.array (input_for_test).reshape (len (input_for_test), 20, 200, 10)
            if label_input_type == 'lidar_coord':
                input_coord_train, input_coord_validation = sliding_prepare_coord_for_trainning (input_for_train [1])
                input_lidar_train, input_lidar_validation = sliding_prepare_lidar_for_trainning (input_for_train [0])
                input_train = [input_lidar_train, input_coord_train]
                input_validation = [input_lidar_validation, input_coord_validation]
                input_coord_test = np.array (input_for_test [0]).reshape (len (input_for_test [0]), 2, 1)
                input_lidar_test = np.array (input_for_test [1]).reshape (len (input_for_test [1]), 20, 200, 10)
                input_test = [input_lidar_test, input_coord_test]

            print(i, end=' ', flush=True)

            df_results_top_k = beam_selection_Batool(input=label_input_type,
                                                 data_train=[input_train, label_train],
                                                 data_validation=[input_validation, label_validation],
                                                 data_test=[input_test, label_test],
                                                 num_classes=num_classes,
                                                 episode=i,
                                                 see_trainning_progress=see_trainning_progress,
                                                 restore_models=False,
                                                 flag_fast_experiment=True)
            df_all_results_top_k = pd.concat([df_all_results_top_k, df_results_top_k], ignore_index=True)
            path_result = ('../../results/score/Batool/online/top_k/') + label_input_type + '/fixed_window/'
            df_all_results_top_k.to_csv(path_result + 'all_results_fixed_window_top_k.csv', index=False)
    a=0


def fit_fixed_window_top_k_old(label_input_type, nro_of_episodes_for_test):
    #data_train, data_validation, data_test, num_classes = prepare_data(input)
    #y_train, y_validation, y_test, num_classes = get_index_beams()

    nro_of_episodes = nro_of_episodes_for_test
    episodes_for_train = 2086
    see_trainning_progress = 0 # 0: no,
                               # 1: yes (how you an animated progress bar)
                               # 2: yes (will just mention the number of epochs completed)

    data_for_train, data_for_validation, s009_data, num_classes = read_all_data()

    episode_for_test = np.arange(0, nro_of_episodes, 1)

    label_train = np.array(data_for_train['index_beams'].tolist())
    label_validation = np.array(data_for_validation['index_beams'].tolist())
    if label_input_type == 'coord':
        input_train = prepare_coord_for_trainning(data_for_train, 9234)
        input_validation = prepare_coord_for_trainning(data_for_validation, 1960)

    elif label_input_type == 'lidar':
        #print('Beam selection using ' + label_input_type)
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
        if i in s009_data['Episode'].tolist():
            #label_test = np.array(s009_data[s009_data['EpisodeID'] == i]['beam'].tolist()) #se funciona, colocar um na if lidar con o np.array e um no coord sem o np.array
            if label_input_type == 'coord':
                label_test = s009_data [s009_data ['EpisodeID'] == i] ['beam'].tolist()
                input_test = prepare_coord_for_test(data=s009_data, episodio_for_test=i)

            elif label_input_type == 'lidar':
                label_test = np.array (s009_data [s009_data ['EpisodeID'] == i] ['beam'].tolist ())
                input_test = np.array(s009_data[s009_data['EpisodeID'] == i]['lidar'].tolist()).reshape(len(label_test),
                                                                                                        20, 200, 10)
            elif label_input_type == 'lidar_coord':
                label_test = np.array(s009_data[s009_data['EpisodeID'] == i] ['beam'].tolist ())
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
                                                     see_trainning_progress= see_trainning_progress,
                                                     restore_models=False,
                                                     flag_fast_experiment=True)
            df_all_results_top_k = pd.concat([df_all_results_top_k, df_results_top_k], ignore_index=True)

            a=0
            b=0
    path_result = ('../../results/score/Batool/online/top_k/') + label_input_type + '/fixed_window/'
    df_all_results_top_k.to_csv(path_result + 'all_results_fixed_window_top_k.csv', index=False)

def sliding_prepare_coord_for_trainning(input_for_train):
    new_form_of_input = np.zeros ((len (input_for_train), 2))
    for i in range (len (input_for_train)):
        new_form_of_input[i] = np.array(input_for_train [i])

    size_of_input = new_form_of_input.shape
    size_of_train = int (size_of_input [0] * 0.8)
    train = new_form_of_input[:size_of_train]
    input_train = train.reshape(train.shape[0], 2, 1)
    val = new_form_of_input[size_of_train:]
    input_validation = val.reshape(val.shape[0], 2, 1)

    return input_train, input_validation

def sliding_prepare_lidar_for_trainning(lidar_for_train):
    new_form_for_lidar = np.array(lidar_for_train)
    size_of_lidar = new_form_for_lidar.shape
    size_of_train = int(size_of_lidar[0] * 0.8)
    input_train = new_form_for_lidar[:size_of_train]
    input_validation = new_form_for_lidar[size_of_train:]

    lidar_train = input_train.reshape(input_train.shape[0], 20, 200, 10)
    lidar_validation = input_validation.reshape(input_validation.shape[0], 20, 200, 10)
    return lidar_train, lidar_validation
def sliding_prepare_label_for_trainning(label_for_train):
    new_form_for_label = np.array(label_for_train)
    size_of_label = new_form_for_label.shape
    size_of_train = int(size_of_label[0] * 0.8)
    label_train = new_form_for_label[:size_of_train]
    label_validation = new_form_for_label[size_of_train:]
    return label_train, label_validation
def fit_incremental_window_top_k(label_input_type, episodes_for_test, start_epi = 0):
    import sys
    import os

    # Adiciona o caminho do diretório do arquivo que você quer importar
    # sys.path.append(os.path.abspath("../"))
    sys.path.append ("../")

    # Agora é possível importar o arquivo como um módulo
    import tools as tls

    data_for_train, data_for_validation, s009_data, num_classes = read_all_data()

    all_dataset_s008 = pd.concat([data_for_train, data_for_validation], axis=0)

    episode_for_test = np.arange (0, episodes_for_test, 1)
    see_trainning_progress = False
    df_all_results_top_k = pd.DataFrame ()


    stop_epi = len(episode_for_test)

    for i in range(start_epi, stop_epi, 1):

        if i in s009_data['Episode'].tolist():
            if i == 0:
                start_index_s008 = i
                input_for_train, label_for_train = tls.extract_training_data_from_s008_sliding_window (all_dataset_s008,
                                                                                                       start_index_s008,
                                                                                                       label_input_type)
                input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (i,
                                                                                                 label_input_type,
                                                                                                 s009_data)
                label_test = np.array (label_for_test)
                label_train, label_validation = sliding_prepare_label_for_trainning (label_for_train)

                if label_input_type == 'coord':
                    input_train, input_validation = sliding_prepare_coord_for_trainning(input_for_train)
                    input_test = np.array(input_for_test).reshape(len(input_for_test), 2, 1)
                if label_input_type == 'lidar':
                    input_train, input_validation = sliding_prepare_lidar_for_trainning(input_for_train)
                    input_test = np.array(input_for_test).reshape(len(input_for_test), 20, 200, 10)

            else:
                start_index_s008 = 0
                input_for_train_s008, label_for_train_s008 = tls.extract_training_data_from_s008_sliding_window(all_dataset_s008,
                                                                                                                start_index_s008,
                                                                                                                label_input_type)
                input_for_train_s009, label_for_train_s009 = tls.extract_training_data_from_s009_sliding_window (s009_data=s009_data,
                                                                                                                 start_index=0,
                                                                                                                 end_index=i,
                                                                                                                 input_type=label_input_type)
                input_for_train = np.concatenate ((input_for_train_s008, input_for_train_s009), axis=0)
                label_for_train = np.concatenate ((label_for_train_s008, label_for_train_s009), axis=0)

                input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (i,
                                                                                                 label_input_type,
                                                                                                 s009_data)
                label_test = np.array(label_for_test)
                label_train, label_validation = sliding_prepare_label_for_trainning (label_for_train)

                if label_input_type == 'coord':
                    input_train, input_validation = sliding_prepare_coord_for_trainning(input_for_train)
                    input_test = np.array(input_for_test).reshape(len(input_for_test), 2, 1)
                if label_input_type == 'lidar':
                    input_train, input_validation = sliding_prepare_lidar_for_trainning(input_for_train)
                    input_test = np.array(input_for_test).reshape(len(input_for_test), 20, 200, 10)



            print(i, end=' ', flush=True)
            df_results_top_k = beam_selection_Batool (input=label_input_type,
                                                      data_train=[input_train, label_train],
                                                      data_validation=[input_validation, label_validation],
                                                      data_test=[input_test, label_test],
                                                      num_classes=num_classes,
                                                      episode=i,
                                                      see_trainning_progress=see_trainning_progress,
                                                      restore_models=False)

            df_all_results_top_k = pd.concat([df_all_results_top_k, df_results_top_k], ignore_index=True)
            path_result = ('../../results/score/Batool/online/top_k/') + label_input_type + '/incremental_window/'
            df_all_results_top_k.to_csv(path_result + 'all_results_incremental_window' + '_top_k.csv', index=False)


def fit_jumpy_sliding_window_top_k(label_input_type,
                                        episodes_for_test,
                                        window_size):

    import sys
    import os

    # Adiciona o caminho do diretório do arquivo que você quer importar
    # sys.path.append(os.path.abspath("../"))
    sys.path.append ("../")

    # Agora é possível importar o arquivo como um módulo
    import tools as tls


    data_for_train, data_for_validation, s009_data, num_classes = read_all_data ()
    all_dataset_s008 = pd.concat ([data_for_train, data_for_validation], axis=0)


    episode_for_test = np.arange (0, episodes_for_test, 1)
    see_trainning_progress = 0
    monitor_ep_train =0
    start_index_s009 = 0
    nro_episodes_s008 = 2086  # from 0 to 1564

    df_all_results_top_k = pd.DataFrame ()
    episode_monitor = 0
    for i in range (len (episode_for_test)):
        if i in s009_data ['Episode'].tolist ():
            if i == 0:
                start_index_s008 = nro_episodes_s008 - window_size

                #print('------ TRAIN : ', start_index_s008 , 'To', nro_episodes_s008)
                #print(i)

                input_for_train, label_for_train = tls.extract_training_data_from_s008_sliding_window (all_dataset_s008,
                                                                                                       start_index_s008,
                                                                                                       label_input_type)

                input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (i,
                                                                                                 label_input_type,
                                                                                                 s009_data)




                if label_input_type == 'coord':
                    input_train, input_validation = sliding_prepare_coord_for_trainning(input_for_train)
                    input_test = np.array(input_for_test).reshape(len(input_for_test), 2, 1)

                if label_input_type == 'lidar':
                    input_train_1, input_validation_1 = sliding_prepare_lidar_for_trainning (input_for_train)
                    shape_of_input_train_1 = input_train_1.shape
                    shape_of_input_validation_1 = input_validation_1.shape

                    input_train = input_train_1.reshape(shape_of_input_train_1[0], 20, 200, 10)
                    input_validation = input_validation_1.reshape(shape_of_input_validation_1[0], 20, 200, 10)

                    input_test = np.array(input_for_test).reshape(len(input_for_test),20, 200, 10)

                if label_input_type == 'lidar_coord':
                    input_coord_s008_train, input_coord_s008_validation = sliding_prepare_coord_for_trainning (
                        input_for_train[1])
                    input_lidar_s008_train, input_lidar_s008_validation = sliding_prepare_lidar_for_trainning (
                        input_for_train[0])

                    input_train = [input_lidar_s008_train, input_coord_s008_train]
                    input_validation = [input_lidar_s008_validation, input_coord_s008_validation]

                    input_coord_test = np.array (input_for_test [0]).reshape (len (input_for_test [0]), 2, 1)
                    input_lidar_test = np.array (input_for_test [1]).reshape (len (input_for_test [1]), 20, 200, 10)
                    input_test = [input_lidar_test, input_coord_test]



                #if label_input_type == 'lidar_coord':
                #    input_train, input_validation = sliding_prepare_coord_for_trainning (input_for_train)

                label_train, label_validation = sliding_prepare_label_for_trainning (label_for_train)
                label_test = np.array (label_for_test)

                #print ('Test Episode: ', i)
                flag_fit_jumpy = True


            else:
                episode_monitor = episode_monitor+1
                if episode_monitor <40:
                    #Test
                    #print(i)
                    input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (i,
                                                                                                     label_input_type,
                                                                                                     s009_data)
                    if label_input_type == 'coord':
                        input_test = np.array (input_for_test).reshape (len (input_for_test), 2, 1)
                    if label_input_type == 'lidar':
                        input_test = np.array(input_for_test).reshape(len(input_for_test), 20, 200, 10)
                    if label_input_type == 'lidar_coord':
                        input_coord_test = np.array (input_for_test [0]).reshape (len (input_for_test [0]), 2, 1)
                        input_lidar_test = np.array (input_for_test [1]).reshape (len (input_for_test [1]), 20, 200, 10)
                        input_test = [input_lidar_test, input_coord_test]

                    label_test = np.array (label_for_test)
                    flag_fit_jumpy = False

                else:
                    start_index_s008 = (nro_episodes_s008-window_size) + monitor_ep_train + 1
                    if start_index_s008 < nro_episodes_s008:
                        episode_monitor=0
                        start_index_s009 = 0
                        end_index_s009 = window_size - (nro_episodes_s008 - start_index_s008)

                        #print ('------ TRAIN s008: ', start_index_s008, 'To', nro_episodes_s008, '=', nro_episodes_s008-start_index_s008)
                        #print ('------ TRAIN s009: ', start_index_s009, 'To', end_index_s009, '=', end_index_s009-start_index_s009)

                        input_for_train_s008, label_for_train_s008 = tls.extract_training_data_from_s008_sliding_window (
                            all_dataset_s008,
                            start_index_s008,
                            label_input_type)

                        input_for_train_s009, label_for_train_s009 = tls.extract_training_data_from_s009_sliding_window (
                            s009_data=s009_data,
                            start_index=start_index_s009,
                            end_index=end_index_s009,
                            input_type=label_input_type)

                        label_train_s008, label_val_s008 = sliding_prepare_label_for_trainning (label_for_train_s008)
                        label_train_s009, label_val_s009 = sliding_prepare_label_for_trainning (label_for_train_s009)
                        label_train = np.concatenate ((label_train_s008, label_train_s009), axis=0)
                        label_validation = np.concatenate ((label_val_s008, label_val_s009), axis=0)

                        #print('Test episode: ', i)

                        input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (i,
                                                                                                         label_input_type,
                                                                                                         s009_data)

                        if label_input_type == 'coord':

                            input_train_s008, input_val_s008 = sliding_prepare_coord_for_trainning(input_for_train_s008)
                            input_train_s009, input_val_s009 = sliding_prepare_coord_for_trainning(input_for_train_s009)
                            input_for_train = np.concatenate((input_train_s008, input_train_s009), axis=0)
                            input_for_validation = np.concatenate((input_val_s008, input_val_s009), axis=0)

                            input_train = input_for_train.reshape(input_for_train.shape[0], 2, 1)
                            input_validation = input_for_validation.reshape(input_for_validation.shape[0], 2, 1)
                            input_test = np.array(input_for_test).reshape(len (input_for_test), 2, 1)

                        if label_input_type == 'lidar':
                            input_train_s008, input_validation_s008 = sliding_prepare_lidar_for_trainning (input_for_train_s008)
                            input_train_s009, input_validation_s009 = sliding_prepare_lidar_for_trainning (input_for_train_s009)
                            input_for_train = np.concatenate ((input_train_s008, input_train_s009), axis=0)
                            input_for_validation = np.concatenate ((input_validation_s008, input_validation_s009), axis=0)

                            input_train = np.array(input_for_train).reshape(input_for_train.shape[0], 20, 200, 10)
                            input_validation = np.array(input_for_validation).reshape(input_for_validation.shape[0], 20, 200, 10)

                            input_test = np.array(input_for_test).reshape(len(input_for_test), 20, 200, 10)

                        if label_input_type == 'lidar_coord':
                            input_coord_s008_train, input_coord_s008_validation = sliding_prepare_coord_for_trainning (
                                input_for_train_s008 [1])
                            input_lidar_s008_train, input_lidar_s008_validation = sliding_prepare_lidar_for_trainning (
                                input_for_train_s008 [0])

                            input_coord_s009_train, input_coord_s009_validation = sliding_prepare_coord_for_trainning (
                                input_for_train_s009 [1])
                            input_lidar_s009_train, input_lidar_s009_validation = sliding_prepare_lidar_for_trainning (
                                input_for_train_s009 [0])

                            input_lidar_train = np.concatenate ((input_lidar_s008_train, input_lidar_s009_train), axis=0)
                            input_coord_train = np.concatenate ((input_coord_s008_train, input_coord_s009_train), axis=0)

                            input_lidar_validation = np.concatenate ((input_lidar_s008_validation, input_lidar_s009_validation),
                                                                    axis=0)
                            input_coord_validation = np.concatenate ((input_coord_s008_validation, input_coord_s009_validation),
                                                                    axis=0)

                            input_train = [input_lidar_train, input_coord_train]
                            input_validation = [input_lidar_validation, input_coord_validation]

                            input_coord_test = np.array (input_for_test [0]).reshape (len (input_for_test [0]), 2, 1)
                            input_lidar_test = np.array (input_for_test [1]).reshape (len (input_for_test [1]), 20, 200, 10)
                            input_test = [input_lidar_test, input_coord_test]

                        label_test = np.array(label_for_test)
                        #label_train = np.array(label_for_train)
                        #label_validation = np.array(label_for_validation)
                        flag_fit_jumpy = True
                        monitor_ep_train += 1
                        '''

                        df_results = beam_selection_Batool_for_fit_jumpy (input=label_input_type,
                                                                          data_train=[input_train,
                                                                                      label_train],
                                                                          data_validation=[input_validation,
                                                                                           label_validation],
                                                                          data_test=[input_test, label_test],
                                                                          num_classes=num_classes,
                                                                          episode=i,
                                                                          see_trainning_progress=see_trainning_progress,
                                                                          flag_fit_jumpy=True)


                        '''
                    else:
                        episode_monitor=0
                        end_index_s009 = start_index_s009 + window_size + monitor_ep_train + 1
                        #print ('------ TRAIN s009: ', start_index_s009, 'To', end_index_s009, '=', end_index_s009-start_index_s009)

                        input_for_train_s009, label_for_train_s009 = tls.extract_training_data_from_s009_sliding_window (
                            s009_data=s009_data,
                            start_index=start_index_s009,
                            end_index=end_index_s009,
                            input_type=label_input_type)
                        input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (i,
                                                                                                         label_input_type,
                                                                                                         s009_data)
                        label_train, label_validation = sliding_prepare_label_for_trainning (label_for_train_s009)

                        if label_input_type == 'coord':
                            input_train_s009, input_val_s009 = sliding_prepare_coord_for_trainning(input_for_train_s009)

                            input_for_train = input_train_s009
                            input_for_validation = input_val_s009

                            input_train = input_for_train.reshape (input_for_train.shape [0], 2, 1)
                            input_validation = input_for_validation.reshape (input_for_validation.shape [0], 2, 1)

                            input_test = np.array (input_for_test).reshape (len (input_for_test), 2, 1)

                        if label_input_type == 'lidar':
                            input_train_s009, input_validation_s009 = sliding_prepare_lidar_for_trainning (input_for_train_s009)
                            input_for_train = input_train_s009
                            input_for_validation = input_validation_s009

                            input_train = np.array(input_for_train).reshape(input_for_train.shape[0], 20, 200, 10)
                            input_validation = np.array(input_for_validation).reshape(input_for_validation.shape[0], 20, 200, 10)
                            input_test = np.array(input_for_test).reshape(len(input_for_test), 20, 200, 10)
                        if label_input_type == 'lidar_coord':
                            input_coord_s009_train, input_coord_s009_validation = sliding_prepare_coord_for_trainning (
                                input_for_train_s009 [1])
                            input_lidar_s009_train, input_lidar_s009_validation = sliding_prepare_lidar_for_trainning (
                                input_for_train_s009 [0])

                            input_lidar_train = input_lidar_s009_train
                            input_coord_train = input_coord_s009_train

                            input_lidar_validation = input_lidar_s009_validation
                            input_coord_validation = input_coord_s009_validation

                            input_train = [input_lidar_train, input_coord_train]
                            input_validation = [input_lidar_validation, input_coord_validation]

                            input_coord_test = np.array (input_for_test [0]).reshape (len (input_for_test [0]), 2, 1)
                            input_lidar_test = np.array (input_for_test [1]).reshape (len (input_for_test [1]), 20, 200, 10)
                            input_test = [input_lidar_test, input_coord_test]

                        #print('Test episode: ', i)
                        label_test = np.array (label_for_test)
                        #label_train = np.array (label_for_train)
                        #label_validation = np.array (label_for_validation)

                        start_index_s009 += 1
                        flag_fit_jumpy = True
                        monitor_ep_train += 1

        #print (i, end=' ', flush=True)

        df_results = beam_selection_Batool_for_fit_jumpy (input=label_input_type,
                                                          data_train=[input_train,
                                                                      label_train],
                                                          data_validation=[input_validation,
                                                                           label_validation],
                                                          data_test=[input_test, label_test],
                                                          num_classes=num_classes,
                                                          episode=i,
                                                          see_trainning_progress=see_trainning_progress,
                                                          flag_fit_jumpy=flag_fit_jumpy)

        df_all_results_top_k = pd.concat([df_all_results_top_k, df_results], ignore_index=True)
        path_result = ('../../results/score/Batool/online/top_k/') + label_input_type + '/jumpy_sliding_window/'
        df_all_results_top_k.to_csv(path_result + 'all_results_jumpy_sliding_window_top_k.csv', index=False)


def fit_sliding_window_top_k(label_input_type,
                             episodes_for_test,
                             window_size):
    import sys
    import os

    # Adiciona o caminho do diretório do arquivo que você quer importar
    # sys.path.append(os.path.abspath("../"))
    sys.path.append ("../")

    # Agora é possível importar o arquivo como um módulo
    import tools as tls


    data_for_train, data_for_validation, s009_data, num_classes = read_all_data()
    all_dataset_s008 = pd.concat([data_for_train, data_for_validation], axis=0)


    episode_for_test = np.arange(0, episodes_for_test, 1)
    see_trainning_progress = 0
    start_index_s009 = 0
    nro_episodes_s008 = 2086 #from 0 to 1564

    df_all_results_top_k = pd.DataFrame()
    for i in range(len(episode_for_test)):
        # for i in tqdm(range(len(episode_for_test))):
        #i=101
        if i in s009_data['Episode'].tolist():
            if i == 0:
                start_index_s008 = nro_episodes_s008 - window_size
                input_for_train, label_for_train = tls.extract_training_data_from_s008_sliding_window(all_dataset_s008,
                                                                                                      start_index_s008,
                                                                                                      label_input_type)

                input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (i,
                                                                                                 label_input_type,
                                                                                                 s009_data)
                label_train, label_validation = sliding_prepare_label_for_trainning (label_for_train)
                label_test = np.array (label_for_test)

                if label_input_type == 'coord':
                    input_train, input_validation = sliding_prepare_coord_for_trainning(input_for_train)
                    input_test = np.array(input_for_test).reshape(len(input_for_test), 2, 1)
                if label_input_type == 'lidar':
                    input_train, input_validation = sliding_prepare_lidar_for_trainning(input_for_train)
                    input_test = np.array(input_for_test).reshape(len(input_for_test), 20, 200, 10)
                if label_input_type == 'lidar_coord':
                    input_coord_train, input_coord_validation = sliding_prepare_coord_for_trainning(input_for_train[1])
                    input_lidar_train, input_lidar_validation = sliding_prepare_lidar_for_trainning(input_for_train[0])
                    input_train = [input_lidar_train, input_coord_train]
                    input_validation = [input_lidar_validation, input_coord_validation]
                    input_coord_test = np.array(input_for_test[0]).reshape(len(input_for_test[0]), 2, 1)
                    input_lidar_test = np.array(input_for_test[1]).reshape(len(input_for_test[1]), 20, 200, 10)
                    input_test = [input_lidar_test, input_coord_test]
                    a=0

            else:
                start_index_s008 = (nro_episodes_s008 - window_size) + i
                #start_index_s008 = 1565
                if start_index_s008 < nro_episodes_s008:
                    start_index_s009 = 0
                    end_index_s009 = window_size - (nro_episodes_s008 - start_index_s008)
                    input_for_train_s008, label_for_train_s008 = tls.extract_training_data_from_s008_sliding_window(
                                                                                          all_dataset_s008,
                                                                                          start_index_s008,
                                                                                          label_input_type)
                    label_train_s008, label_val_s008 = sliding_prepare_label_for_trainning(label_for_train_s008)

                    input_for_train_s009, label_for_train_s009 = tls.extract_training_data_from_s009_sliding_window(
                                                                    s009_data=s009_data,
                                                                    start_index=start_index_s009,
                                                                    end_index=end_index_s009,
                                                                    input_type=label_input_type)
                    label_train_s009, label_val_s009 = sliding_prepare_label_for_trainning(label_for_train_s009)

                    label_train = np.concatenate((label_train_s008, label_train_s009), axis=0)
                    label_validation = np.concatenate((label_val_s008, label_val_s009), axis=0)


                    input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window(i,
                                                                                                    label_input_type,
                                                                                                    s009_data)
                    label_test = np.array(label_for_test)

                    if label_input_type == 'coord':
                        input_train_s008, input_val_s008 = sliding_prepare_coord_for_trainning (input_for_train_s008)
                        input_train_s009, input_val_s009 = sliding_prepare_coord_for_trainning (input_for_train_s009)

                        input_train = np.concatenate((input_train_s008, input_train_s009), axis=0)
                        input_validation = np.concatenate((input_val_s008, input_val_s009), axis=0)

                        #input_train = input_for_train.reshape(input_for_train.shape[0], 2, 1)
                        #input_validation = input_for_validation.reshape(input_for_validation.shape[0], 2, 1)

                        input_test = np.array(input_for_test).reshape(len(input_for_test), 2, 1)

                    if label_input_type == 'lidar':
                        input_train_s008, input_validation_s008 = sliding_prepare_lidar_for_trainning(input_for_train_s008)
                        input_train_s009, input_validation_s009 = sliding_prepare_lidar_for_trainning(input_for_train_s009)

                        input_train = np.concatenate((input_train_s008, input_train_s009), axis=0)
                        input_validation = np.concatenate((input_validation_s008, input_validation_s009), axis=0)

                        input_test = np.array(input_for_test).reshape(len(input_for_test), 20, 200, 10)

                    if label_input_type == 'lidar_coord':
                        input_coord_s008_train, input_coord_s008_validation = sliding_prepare_coord_for_trainning(input_for_train_s008[1])
                        input_lidar_s008_train, input_lidar_s008_validation = sliding_prepare_lidar_for_trainning(input_for_train_s008[0])
                        #input_s008_train = [input_lidar_s008_train, input_coord_s008_train]
                        #input_s008_validation = [input_lidar_s008_validation, input_coord_s008_validation]

                        input_coord_s009_train, input_coord_s009_validation = sliding_prepare_coord_for_trainning(input_for_train_s009[1])
                        input_lidar_s009_train, input_lidar_s009_validation = sliding_prepare_lidar_for_trainning(input_for_train_s009[0])
                        #input_s009_train = [input_lidar_s009_train, input_coord_s009_train]
                        #input_s009_validation = [input_lidar_s009_validation, input_coord_s009_validation]

                        input_lidar_train = np.concatenate((input_lidar_s008_train, input_lidar_s009_train), axis=0)
                        input_coord_train = np.concatenate((input_coord_s008_train, input_coord_s009_train), axis=0)

                        input_lidar_validation = np.concatenate((input_lidar_s008_validation, input_lidar_s009_validation), axis=0)
                        input_coord_validation = np.concatenate((input_coord_s008_validation, input_coord_s009_validation), axis=0)

                        input_train = [input_lidar_train, input_coord_train]
                        input_validation = [input_lidar_validation, input_coord_validation]

                        input_coord_test = np.array(input_for_test[0]).reshape(len(input_for_test[0]), 2, 1)
                        input_lidar_test = np.array(input_for_test[1]).reshape(len(input_for_test[1]), 20, 200, 10)
                        input_test = [input_lidar_test, input_coord_test]


                else:
                    end_index_s009 = start_index_s009 + window_size
                    input_for_train_s009, label_for_train_s009 = tls.extract_training_data_from_s009_sliding_window(s009_data=s009_data,
                                                                                                   start_index=start_index_s009,
                                                                                                   end_index=end_index_s009,
                                                                                                   input_type=label_input_type)

                    label_train, label_validation = sliding_prepare_label_for_trainning (label_for_train_s009)
                    input_for_test, label_for_test = tls.extract_test_data_from_s009_sliding_window (i,
                                                                                                     label_input_type,
                                                                                                     s009_data)

                    label_test = np.array (label_for_test)
                    if label_input_type == 'coord':
                        input_train, input_validation = sliding_prepare_coord_for_trainning (input_for_train_s009)
                        input_test = np.array(input_for_test).reshape (len (input_for_test), 2, 1)
                    if label_input_type == 'lidar':
                        input_train, input_validation = sliding_prepare_lidar_for_trainning (input_for_train_s009)
                        input_test = np.array(input_for_test).reshape (len (input_for_test), 20, 200, 10)
                    if label_input_type == 'lidar_coord':
                        input_coord_train, input_coord_validation = sliding_prepare_coord_for_trainning (input_for_train_s009[1])
                        input_lidar_train, input_lidar_validation = sliding_prepare_lidar_for_trainning (input_for_train_s009[0])
                        input_train = [input_lidar_train, input_coord_train]
                        input_validation = [input_lidar_validation, input_coord_validation]
                        input_coord_test = np.array(input_for_test[0]).reshape(len(input_for_test[0]), 2, 1)
                        input_lidar_test = np.array(input_for_test[1]).reshape(len(input_for_test[1]), 20, 200, 10)
                        input_test = [input_lidar_test, input_coord_test]


                    start_index_s009 += 1

            print (i, end=' ', flush=True)
            df_results_top_k = beam_selection_Batool (input=label_input_type,
                                                      data_train=[input_train, label_train],
                                                      data_validation=[input_validation, label_validation],
                                                      data_test=[input_test, label_test],
                                                      num_classes=num_classes,
                                                      episode=i,
                                                      see_trainning_progress=see_trainning_progress,
                                                      restore_models=False)
            df_all_results_top_k = pd.concat([df_all_results_top_k, df_results_top_k], ignore_index=True)
            path_result = ('../../results/score/Batool/online/top_k/') + label_input_type + '/sliding_window/'#window_size_var/'
            df_all_results_top_k.to_csv (path_result + 'all_results_sliding_window_'+str(window_size)+'_top_k.csv', index=False)





def read_results_for_plot(type_of_input, type_of_window, window_size, model):
    #type_of_input = 'coord'
    #type_of_window = 'fixed_window'
    window_size = str(window_size)
    flag_servidor = 3
    filename = 'scores_with_' + type_of_window + '_top_k.csv'
    metric = 'time_trainning'  # 'score'

    if flag_servidor == 1:  # servidor local
        path = '../../results/score/Batool/online/top_k/' + type_of_input + '/' + type_of_window + '/'
        servidor = 'servidor_local'

        pos_y = 70
        pos_x = [10, 50, 100, 150, 200, 250, 300]
    elif flag_servidor == 2:  # servidor portugal
        servidor = 'servidor_portugal'
        pos_y = 525
        pos_x = [10, 250, 500, 750, 1000, 1250, 1500]
        path = '../../results/score/Batool/online/top_k/' + type_of_input + '/' + type_of_window + '/' + servidor + '/'
    elif flag_servidor == 3:  # servidor land
        servidor = 'servidor_land'

        pos_x = [10, 250, 500, 750, 1000, 1250, 1500]

        if type_of_input == 'coord':
            pos_y = 0.3
        elif type_of_input == 'lidar' or type_of_input == 'lidar_coord':
            pos_y = 0.6
        if type_of_window == 'fixed_window':
            #path = '../../results/score/Batool/online/top_k/' + type_of_input + '/' + type_of_window + '/' + servidor + '/'
            path = '../../results/score/Batool/servidor_land/online/' + type_of_input + '/' + type_of_window + '/'
            filename = 'all_results_'+type_of_window+'_top_k.csv'
            title = 'Beam Selection using ' + model + ' with ' + type_of_input + ' and ' + type_of_window + '\n Reference: Batool -' + servidor

        elif type_of_window == 'sliding_window':
            #path = '../../results/score/Batool/online/top_k/' + type_of_input + '/' + type_of_window + '/window_size_var/' + servidor + '/window_size_var/'+ window_size +'/'
            #path = '../../results/score/Batool/online/top_k/' + type_of_input + '/' + type_of_window + '/window_size_var/' + servidor + '/window_size_var/'
            path = '../../results/score/Batool/servidor_land/online/' + type_of_input + '/' + type_of_window + '/'

            title = 'Beam Selection using ' + model + ' with ' + type_of_input + ' and ' + type_of_window + window_size + '\n Reference: Batool -' + servidor
            filename = 'all_results_sliding_window_'+ window_size +'_top_k.csv'

        elif type_of_window == 'incremental_window':
            #path = '../../results/score/Batool/online/top_k/' + type_of_input + '/' + type_of_window + '/' + servidor + '/'
            path = '../../results/score/Batool/servidor_land/online/' + type_of_input + '/' + type_of_window + '/'

            title = 'Beam Selection using ' + model + ' with ' + type_of_input + ' and ' + type_of_window + '\n Reference: Batool -' + servidor
            filename = 'all_results_incremental_window_top_k.csv'
        elif type_of_window == 'jumpy_sliding_window':
            #path = '../../results/score/Batool/online/top_k/' + type_of_input + '/' + type_of_window + '/' + servidor + '/'
            path = '../../results/score/Batool/servidor_land/online/' + type_of_input + '/' + type_of_window + '/'

            title = 'Beam Selection using ' + model + ' with ' + type_of_input + ' and ' + type_of_window + '\n Reference: Batool -' + servidor
            filename = 'all_results_jumpy_sliding_window_top_k.csv'
            pos_x = [10, 250, 500, 750, 1000, 1250, 1500]
            #pos_x = [0, 10, 20, 40, 60, 70, 80, 100]
            pos_y = 0.8

    return path, servidor, filename, pos_x, pos_y, title
def plot_results__(type_of_input, type_of_window, window_size=0):
    import sys
    import os

    # Adiciona o caminho do diretório do arquivo que você quer importar
    #sys.path.append(os.path.abspath("../"))
    sys.path.append("../")

    # Agora é possível importar o arquivo como um módulo
    import plot_results as plot


    if type_of_input == 'coord':
        model = 'MLP'
    elif type_of_input == 'lidar' or type_of_input == 'lidar_coord':
        model = 'DNN'
    path, servidor, filename, pos_x, pos_y, title = read_results_for_plot(type_of_input,
                                                                          type_of_window,
                                                                          window_size,
                                                                          model)
    if type_of_window == 'jumpy_sliding_window':
        plot_score (type_of_input, type_of_window, model, window_size)
        plot.plot_analyses_time_process_jumpy_sliding (path, filename, title)
        plot.plot_analyses_hist_of_jumpy_sliding (path=path, filename=filename, graph_type='hist', title=title)
        plot.plot_analyses_hist_of_jumpy_sliding (path=path, filename=filename, graph_type='ecdf', title=title)
        plot.plot_analyses_hist_of_jumpy_sliding (path=path, filename=filename, graph_type='kde', title=title)

    else:
        plot_score(type_of_input, type_of_window, model, window_size)
        plot.plot_time_process_vs_samples_online_learning(path=path, filename=filename, title=title, ref='batool', window_size=window_size, flag_fast_experiment=True)
        plot.plot_histogram_of_trainning_time(path=path, filename=filename, title=title, graph_type='hist', window_size=window_size)
        plot.plot_histogram_of_trainning_time(path=path, filename=filename, title=title, graph_type='ecdf', window_size=window_size)
        plot.plot_time_process_online_learning(path=path, filename=filename,  title=title, window_size=window_size, window_type=type_of_window)


def plot_compare_score(type_of_input, type_of_window, window_size):
    if type_of_input == 'coord':
        model = 'MLP'
    elif type_of_input == 'lidar':
        model = 'DNN'

    type_of_window = 'fixed_window'
    path, servidor, filename, pos_x, pos_y, title = read_results_for_plot (type_of_input,
                                                                           type_of_window,
                                                                           window_size,
                                                                           model)
    all_results_fixed_window = pd.read_csv (path + filename)

    type_of_window ='jumpy_sliding_window'
    path, servidor, filename, pos_x, pos_y, title = read_results_for_plot (type_of_input,
                                                                           type_of_window,
                                                                           window_size,
                                                                           model)
    all_results_jumpy_sliding_window = pd.read_csv (path + filename)


    '''
    type_of_window = 'sliding_window'
    path, servidor, filename, pos_x, pos_y, title = read_results_for_plot (type_of_input,
                                                                           type_of_window,
                                                                           window_size,
                                                                           model)
    
    type_of_window = 'incremental_window'
    path, servidor, filename, pos_x, pos_y, title = read_results_for_plot (type_of_input,
                                                                           type_of_window,
                                                                           window_size,
                                                                           model)
    '''

    top_k = [1, 5, 10, 15, 20, 25, 30]
    all_mean_score_top_k_jumpy= []

    for i in range(len(top_k)):
        score_top_k = all_results_jumpy_sliding_window[all_results_jumpy_sliding_window['top-k'] == top_k[i]]
        mean_score_top_k = np.mean(score_top_k['score'])
        all_mean_score_top_k_jumpy.append(mean_score_top_k)

    all_mean_score_top_k_fixed = []
    for i in range(len(top_k)):
        score_top_k = all_results_fixed_window[all_results_fixed_window['top-k'] == top_k[i]]
        mean_score_top_k = np.mean(score_top_k['score'])
        all_mean_score_top_k_fixed.append(mean_score_top_k)

    plt.plot(top_k, all_mean_score_top_k_jumpy, 'o-', label='Jumpy Sliding window')
    plt.plot(top_k, all_mean_score_top_k_fixed, 'o-', label='Fixed window')
    for i in range (len (top_k)):
        plt.text (top_k [i], all_mean_score_top_k_jumpy [i] - 0.02, str (np.round (all_mean_score_top_k_jumpy [i], 3)), color='blue')
        plt.text (top_k [i], all_mean_score_top_k_fixed [i] - 0.02, str (np.round (all_mean_score_top_k_fixed [i], 3)), color='orange')
    plt.legend()
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(top_k)
    plt.xlabel('Top-k')
    plt.ylabel('Score')
    plt.title('Comparison between types of Windows \nfor Batool Reference with '+type_of_input)
    plt.grid()
    plt.show()

    a=0

def plot_score(type_of_input, type_of_window, model, window_size):
    import sys
    import os
    sys.path.append ("../")
    import plot_results as plot
    path, servidor, filename, pos_x, pos_y, title = read_results_for_plot (type_of_input,
                                                                           type_of_window,
                                                                           window_size,
                                                                           model)
    plot.plot_accum_score_top_k(pos_x=pos_x, pos_y=pos_y, path=path, title=title, filename=filename, window_size=window_size)
    plot.plot_score_top_k(path, filename, title, window_size)
    plot.plot_score_top_1(path, filename, title, window_size)



def plot_comparition_time_process(type_of_input, type_of_window, model):
    import seaborn as sns

    path, servidor, filename, pos_x, pos_y, title = read_results_for_plot (type_of_input,
                                                                           type_of_window='fixed_window',
                                                                           window_size=2000,
                                                                           model=model)
    all_results_fixed_window = pd.read_csv (path + 'scores_with_fixed_window_top_k.csv')

    path, servidor, filename, pos_x, pos_y, title = read_results_for_plot (type_of_input,
                                                                           type_of_window='incremental_window',
                                                                           window_size=2000,
                                                                           model=model)

    all_results_incremental_window = pd.read_csv (path + 'all_results_incremental_window_top_k.csv')



    size_of_window = [100, 500, 1000, 1500, 2000]
    color = ['orange', 'blue', 'red', 'green', 'purple']
    flag = 'training'
    sns.set_theme (style="darkgrid")
    fig, ax1 = plt.subplots (figsize=(15, 7))
    plt.plot (all_results_fixed_window['episode'],
              all_results_fixed_window['trainning_process_time'] * 1e-9,
              color='grey', alpha=0.3)
    plt.plot (all_results_incremental_window['episode'],
              all_results_incremental_window['trainning_process_time'] * 1e-9,
              color='magenta', alpha=0.3)

    for i in range(len(size_of_window)):
        path, servidor, filename, pos_x, pos_y, title = read_results_for_plot (type_of_input,
                                                                               type_of_window='sliding_window',
                                                                               window_size=size_of_window[i],
                                                                               model=model)
        all_results_sliding_window = pd.read_csv (path + 'all_results_sliding_window_'+str(size_of_window[i])+'_top_k.csv')
        plt.plot (all_results_sliding_window ['episode'],
                  all_results_sliding_window ['trainning_process_time'] * 1e-9,
                  color=color[i], alpha=0.3)



    ax1.set_ylabel (flag + ' time [s]', fontsize=12, color='black', labelpad=10, fontweight='bold',
                    fontname='Myanmar Sangam MN')
    ax1.set_xlabel ('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold', fontname='Myanmar Sangam MN')

    # Criando um segundo eixo
    ax2 = ax1.twinx ()
    plt.plot (all_results_fixed_window ['episode'],
              all_results_fixed_window ['samples_trainning'],
              marker=',', color='grey',  label='fixed window')

    plt.plot (all_results_incremental_window ['episode'],
              all_results_incremental_window ['samples_trainning'],
              marker=',', color='magenta',  label='incremental window')

    for i in range(len(size_of_window)):
        path, servidor, filename, pos_x, pos_y, title = read_results_for_plot (type_of_input,
                                                                               type_of_window='sliding_window',
                                                                               window_size=size_of_window [i],
                                                                               model=model)
        all_results_sliding_window = pd.read_csv (path + 'all_results_sliding_window_' + str (size_of_window [i]) + '_top_k.csv')

        plt.plot (all_results_sliding_window ['episode'],
                  all_results_sliding_window ['samples_trainning'],
                  marker=',', color=color[i],  label='sliding window ' + str(size_of_window[i]))



    ax2.set_ylabel ('training samples', fontsize=12, color='black', labelpad=12, fontweight='bold',
                    fontname='Myanmar Sangam MN')  # , color='red')

    # Adicionando título e legendas
    title = "Relationship between trained samples and training time \n usign data: " + type_of_input
    plt.title (title, fontsize=15, color='black', fontweight='bold')
    # plt.xticks(all_results_traditional['Episode'])
    plt.xlabel ('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')
    plt.legend (loc='best', ncol=3)  # loc=(0,-0.4), ncol=3)#loc='best')
    #plt.savefig (path_result + input_type + '/time_and_samples_train_comparation.png', dpi=300)
    plt.show ()
    #plt.close ()


    plt.plot (all_results_fixed_window ['episode'], all_results_fixed_window ['trainning_process_time'] / 1e9, '.',
              color='purple', label='Fixed window')

    plt.plot (all_results_sliding_window ['episode'], all_results_sliding_window ['trainning_process_time'] / 1e9, '.',
              color='red', label='Sliding window')
    '''
    plt.plot (all_results_incremental_window ['Episode'], all_results_incremental_window ['Trainning Time'] / 1e9, '.',
              color='green', label='Incremental window')
              '''
    plt.xlabel ('Episode', fontsize=12, color='black', fontname='Myanmar Sangam MN', fontweight='bold')
    plt.ylabel ('Trainning Time', fontsize=12, color='black', fontname='Myanmar Sangam MN', fontweight='bold')
    plt.legend (loc='best')  # , bbox_to_anchor=(1.04, 0))
    plt.title ('Trainning Time using '+ model +' with \n' + type_of_input + ' in online learning - Ref Batool')
    #plt.savefig (path_result + input_type + '/time_train_comparation.png', dpi=300)
    plt.show()
    #plt.close ()

    '''
    plt.plot (all_results_fixed_window ['Episode'], all_results_fixed_window ['Test Time'] / 1e9, '.', color='purple',
              label='Fixed window')
    plt.plot (all_results_sliding_window ['Episode'], all_results_sliding_window ['Test Time'] / 1e9, '.', color='red',
              label='Sliding window')
    plt.plot (all_results_incremental_window ['Episode'], all_results_incremental_window ['Test Time'] / 1e9, '.',
              color='green', label='Incremental window')
    plt.xlabel ('Episode', fontsize=12, color='black', fontname='Myanmar Sangam MN', fontweight='bold')
    plt.ylabel ('Test Time', fontsize=12, color='black', fontname='Myanmar Sangam MN', fontweight='bold')
    plt.legend (loc='best')  # , bbox_to_anchor=(1.04, 0))
    plt.title ('Test Time using WiSARD with \n' + input_type + ' in online learning', fontsize=12, color='black')
    plt.savefig (path_result + input_type + '/time_test_comparation.png', dpi=300)
    plt.close ()
    '''


def read_incremental_data_results_lidar():
    path = '../../results/score/Batool/servidor_land/online/lidar/incremental_window/'


    for i in range(1,46,1):
        file = 'all_results_incremental_window_top_k_parte_'+str(i)+'.csv'
        data = pd.read_csv(path + file)
        if i == 1:
            all_data = data
        else:
            all_data = pd.concat([all_data, data], axis=0)

    all_data.to_csv(path + 'all_results_incremental_window_top_k.csv', index=False)

def main():
    import sys
    import os
    sys.path.append ("../")

    import plot_results as plot
    #import plots_for_jornal_paper/plot_by_window_type as plot_1

    run_simulation = False
    input = 'lidar'
    type_of_window = 3
    plot_compare_results = False

        #1 = 'fixed_window'
        #2 = 'sliding_window'
        #3 = 'incrmental_window'

    print("+-------------------------------------"
          "\n|    online learning - Batool ")
    if type_of_window == 1:
        print("|          Fixed window")
        window = 'fixed_window'
    elif type_of_window == 2:
        print("|          Sliding window")
        window = 'sliding_window'
    elif type_of_window == 3:
        print("|          Incremental window")
        window = 'incremental_window'
    elif type_of_window == 4:
        print("|          Jumpy sliding window")
        window = 'jumpy_sliding_window'

    print("|            " + input +
          "\n+----------------------------------")


    if run_simulation:
        if type_of_window == 1:
                fit_fixed_window_top_k(input, episodes_for_test=2000)
        elif type_of_window == 2:
            window_size = [100]#, 500, 1000, 1500, 2000]
            for i in range(len(window_size)):
                print('window_size:', window_size[i])
                fit_sliding_window_top_k(label_input_type=input,
                                         episodes_for_test=2000,
                                         window_size=window_size[i])
        elif type_of_window == 3:
            fit_incremental_window_top_k(label_input_type=input,
                                         episodes_for_test=2000,
                                         start_epi = 5)
        elif type_of_window == 4:
            fit_jumpy_sliding_window_top_k(label_input_type=input,
                                           episodes_for_test=2000,
                                           window_size=1500)

    else:
        if plot_compare_results:
            plot.plot_compare_types_of_windows (input_name=input, ref='Batool')
            plot.plot_compare_windows_size_in_window_sliding(input_name=input, ref='Batool')
            #plot_1.calculate_statis(input=input, window_type=window, ref='Batool')

        else:
            if type_of_window == 2:
                window_size = [1000]#[100, 500, 1000, 1500, 2000]
                for i in range (len (window_size)):
                    plot_results__(type_of_input=input, type_of_window=window, window_size=window_size[i])
            else:
                plot_results__(type_of_input=input, type_of_window=window)

    # print(tf.__file__)
    # import keras
    # print(keras.__version__)

def test_LOS_NLOS(connection_type, label_input_type):

    #connection_type = 'NLOS'
    #label_input_type = 'lidar'
    reduce_samples_for_train = True
    data_for_train, data_for_validation, data_for_test, num_classes = read_all_data()

    if connection_type == 'LOS':
        if reduce_samples_for_train:
            data_train = data_for_train[data_for_train['LOS'] == 'LOS=1'].sample(n=4285, random_state=1)
            data_val = data_for_validation[data_for_validation['LOS'] == 'LOS=1'].sample(n=427, random_state=1)
        else:
            data_train = data_for_train[data_for_train['LOS'] == 'LOS=1']
            data_val = data_for_validation[data_for_validation['LOS'] == 'LOS=1']
        data_test = data_for_test[data_for_test['LOS'] == 'LOS=1']
    elif connection_type == 'NLOS':
        if reduce_samples_for_train:
            data_train = data_for_train[data_for_train['LOS'] == 'LOS=0'].sample(n=4285, random_state=1)
            data_val = data_for_validation[data_for_validation['LOS'] == 'LOS=0'].sample(n=427, random_state=1)
        else:
            data_train = data_for_train[data_for_train['LOS'] == 'LOS=0']
            data_val = data_for_validation[data_for_validation['LOS'] == 'LOS=0']
        data_test = data_for_test[data_for_test['LOS'] == 'LOS=0']
    elif connection_type == 'ALL':
        if reduce_samples_for_train:
            data_train = data_for_train.sample(n=4285, random_state=1)
            data_val = data_for_validation.sample(n=427, random_state=1)
        else:
            data_train = data_for_train
            data_val = data_for_validation
        data_test = data_for_test

    if label_input_type == 'coord':
        input_train = prepare_coord_for_trainning(data_train, data_train.shape[0])

        input_validation = prepare_coord_for_trainning(data_val, data_val.shape[0])
        input_test = prepare_coord_for_trainning(data_test, data_test.shape[0])

    elif label_input_type == 'lidar':
        input_train = data_train['lidar'].tolist()
        input_train = np.array (input_train).reshape(len(input_train), 20, 200, 10)
        input_validation = data_val['lidar'].tolist()
        input_validation = np.array (input_validation).reshape (len (input_validation), 20, 200, 10)
        input_test = data_test['lidar'].tolist()
        input_test = np.array (input_test).reshape (len (input_test), 20, 200, 10)

    elif label_input_type == 'lidar_coord':
        input_lidar_train = data_train['lidar'].tolist()
        input_lidar_train = np.array (input_lidar_train).reshape(len (input_lidar_train), 20, 200, 10)
        input_lidar_val = data_val['lidar'].tolist()
        input_lidar_val = np.array(input_lidar_val).reshape(len(input_lidar_val), 20, 200, 10)
        input_lidar_test = data_test['lidar'].tolist()
        input_lidar_test = np.array(input_lidar_test).reshape(len(input_lidar_test), 20, 200, 10)

        input_coord_train = prepare_coord_for_trainning (data_train, data_train.shape [0])
        input_coord_validation = prepare_coord_for_trainning (data_val, data_val.shape [0])
        input_coord_test = prepare_coord_for_trainning (data_test, data_test.shape [0])

        input_train = [input_lidar_train, input_coord_train]
        input_validation = [input_lidar_val, input_coord_validation]
        input_test = [input_lidar_test, input_coord_test]

    label_train = np.array(data_train['index_beams'].tolist ())
    label_validation = np.array(data_val['index_beams'].tolist ())
    label_test = np.array(data_test['index_beams'].tolist ())



    df_results_top_k = beam_selection_Batool (input=label_input_type,
                                              data_train=[input_train, label_train],
                                              data_validation=[input_validation, label_validation],
                                              data_test=[input_test, label_test],
                                              num_classes=num_classes,
                                              episode=0,
                                              see_trainning_progress=0,
                                              restore_models=False,
                                              flag_fast_experiment=False)

    if reduce_samples_for_train:
        path_result = (
                    '../../results/score/Batool/split_dataset_LOS_NLOS/' +
                    label_input_type + '/' + connection_type + '/' + 'less_samples_for_train/')

    else:
        path_result = ('../../results/score/Batool/split_dataset_LOS_NLOS/'+label_input_type+'/'+connection_type+'/')
    df_results_top_k.to_csv (path_result + label_input_type +'_results_top_k_batool_'+connection_type+'_ok.csv', index=False)


def read_results_conventional_evaluation(label_input_type):
    reduce_samples_for_train = True
    add_path = 'less_samples_for_train/'

    path = '../../results/score/Batool/split_dataset_LOS_NLOS/'



    connection_type = 'LOS'
    path_result = path + label_input_type + '/' + connection_type + '/'
    file_name = label_input_type + '_results_top_k_batool_' + connection_type + '_ok.csv'
    if reduce_samples_for_train:
        data_LOS = pd.read_csv (path_result + add_path +file_name, delimiter=',')
    else:
        data_LOS = pd.read_csv (path_result + file_name, delimiter=',')
    LOS = data_LOS[data_LOS['top-k'] <= 10]

    connection_type = 'NLOS'
    path_result = path + label_input_type + '/' + connection_type + '/'
    file_name = label_input_type + '_results_top_k_batool_' + connection_type + '_ok.csv'

    if reduce_samples_for_train:
        data_NLOS = pd.read_csv (path_result + add_path + file_name, delimiter=',')
    else:
        data_NLOS = pd.read_csv (path_result + file_name, delimiter=',')
    NLOS = data_NLOS[data_NLOS['top-k'] <= 10]

    connection_type = 'ALL'
    path_result = path + label_input_type + '/' + connection_type + '/'
    file_name = label_input_type + '_results_top_k_batool_' + connection_type + '_ok.csv'
    if reduce_samples_for_train:
        data_ALL = pd.read_csv (path_result + add_path + file_name, delimiter=',')
    else:
        data_ALL = pd.read_csv (path_result + file_name, delimiter=',')
    ALL = data_ALL[data_ALL['top-k'] <= 10]

    return LOS, NLOS, ALL
def plot_test_LOS_NLOS():
    import matplotlib.pyplot as plt
    reduce_samples_for_train = True
    add_path = 'less_samples_for_train/'

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
    plt.suptitle('Selecao de Feixe usando os modelos da Batool', fontsize=size_of_font, font='Times New Roman')

    path_to_save = '../../results/score/Batool/split_dataset_LOS_NLOS/'

    if reduce_samples_for_train:
        file_name = 'performance_accuracy_all_LOS_NLOS_batool_less_samples_for_train_ok.png'
    else:

        file_name = 'performance_accuracy_all_LOS_NLOS_batool.png'
    plt.savefig (path_to_save + file_name, dpi=300, bbox_inches='tight')


#main()
#read_incremental_data_results_lidar()
#plot_compare_score(type_of_input='lidar', type_of_window='fixed_window', window_size=2000)
#window = 'fixed_window'
#input = 'coord'
#plot_comparition_time_process(type_of_input=input, type_of_window=window, model='MLP')
#plot_results__('lidar', 'jumpy_sliding_window')
#fit_fixed_window_top_k(label_input_type='coord', episodes_for_test=2000)
#test_LOS_NLOS(connection_type='NLOS', label_input_type='coord')

#test_LOS_NLOS(connection_type='NLOS', label_input_type='lidar_coord')

#test_LOS_NLOS(connection_type='LOS', label_input_type='coord')
#test_LOS_NLOS(connection_type='NLOS', label_input_type='coord')




plot_test_LOS_NLOS()

