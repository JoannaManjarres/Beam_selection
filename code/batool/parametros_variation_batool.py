import pandas as pd
import os
import time
import random
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

from read_data import read_all_data

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

def train_model(input, model, data_train, data_validation,
                see_trainning_progress, epochs, bs):
    x_train = data_train[0]
    y_train = data_train[1]

    x_validation = data_validation[0]
    y_validation = data_validation[1]

    model_folder = 'models/'
    #epochs = 20  # default 70
    #bs = 64  # default
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
                          epochs, bs,
                          flag_fast_experiment=False):




    top_k = np.arange (1, 31, 1)
    #print(top_k)
    model = model_configuration(input, data_train, data_validation, data_test, num_classes, restore_models)
    trainning_process_time, samples_shape = train_model (input, model,
                                                         data_train, data_validation,
                                                         see_trainning_progress, epochs, bs)
    df_results_top_k = test_model(input, model, data_test, top_k, see_trainning_progress)


    df_results_top_k['episode'] = episode
    df_results_top_k['trainning_process_time'] = trainning_process_time
    df_results_top_k['samples_trainning'] = samples_shape[0]

    return df_results_top_k

def prepare_coord_for_trainning(data, samples):
    coord_x = np.vstack(data['x'].tolist())
    coord_y = np.vstack(data['y'].tolist())
    coord = np.concatenate((coord_x, coord_y), axis=1).reshape(samples,2,1)

    return coord


def parameters_variation(label_input_type, bs, connection_type):
    #connection_type = 'NLOS'
    parameters_of_variation = 'epochs'#'batch_size' #'epochs'
    data_for_train, data_for_validation, data_for_test, num_classes = read_all_data()

    if connection_type == 'LOS':
        data_train = data_for_train [data_for_train ['LOS'] == 'LOS=1'].sample (n=4285, random_state=1)
        data_val = data_for_validation [data_for_validation ['LOS'] == 'LOS=1'].sample (n=427, random_state=1)
        data_test = data_for_test [data_for_test ['LOS'] == 'LOS=1']
    elif connection_type == 'NLOS':
        data_train = data_for_train [data_for_train ['LOS'] == 'LOS=0'].sample (n=4285, random_state=1)
        data_val = data_for_validation [data_for_validation ['LOS'] == 'LOS=0'].sample (n=427, random_state=1)
        data_test = data_for_test [data_for_test ['LOS'] == 'LOS=0']
    elif connection_type == 'ALL':
        data_train = data_for_train.sample (n=4285, random_state=1)
        data_val = data_for_validation.sample (n=427, random_state=1)
        data_test = data_for_test


    if label_input_type == 'coord':
        input_train = prepare_coord_for_trainning (data_train, data_train.shape [0])

        input_validation = prepare_coord_for_trainning (data_val, data_val.shape [0])
        input_test = prepare_coord_for_trainning (data_test, data_test.shape [0])

    elif label_input_type == 'lidar':
        input_train = data_train ['lidar'].tolist ()
        input_train = np.array (input_train).reshape (len (input_train), 20, 200, 10)
        input_validation = data_val ['lidar'].tolist ()
        input_validation = np.array (input_validation).reshape (len (input_validation), 20, 200, 10)
        input_test = data_test ['lidar'].tolist ()
        input_test = np.array (input_test).reshape (len (input_test), 20, 200, 10)

    elif label_input_type == 'lidar_coord':
        input_lidar_train = data_train ['lidar'].tolist ()
        input_lidar_train = np.array (input_lidar_train).reshape (len (input_lidar_train), 20, 200, 10)
        input_lidar_val = data_val ['lidar'].tolist ()
        input_lidar_val = np.array (input_lidar_val).reshape (len (input_lidar_val), 20, 200, 10)
        input_lidar_test = data_test ['lidar'].tolist ()
        input_lidar_test = np.array (input_lidar_test).reshape (len (input_lidar_test), 20, 200, 10)

        input_coord_train = prepare_coord_for_trainning (data_train, data_train.shape [0])
        input_coord_validation = prepare_coord_for_trainning (data_val, data_val.shape [0])
        input_coord_test = prepare_coord_for_trainning (data_test, data_test.shape [0])

        input_train = [input_lidar_train, input_coord_train]
        input_validation = [input_lidar_val, input_coord_validation]
        input_test = [input_lidar_test, input_coord_test]

    label_train = np.array (data_train ['index_beams'].tolist ())
    label_validation = np.array (data_val ['index_beams'].tolist ())
    label_test = np.array (data_test ['index_beams'].tolist ())

    #epochs = [20,30,40,50,60,70,80,90,100, 110, 120,130, 150]# default 70
    epochs = [20, 70,  100]
    #bs=16
    #epochs =[100]
    score = []
    for i in range(len(epochs)):
        df_results_top_k = beam_selection_Batool (input=label_input_type,
                                              data_train=[input_train, label_train],
                                              data_validation=[input_validation, label_validation],
                                              data_test=[input_test, label_test],
                                              num_classes=num_classes,
                                              episode=0,
                                              see_trainning_progress=0,
                                              restore_models=False,
                                              flag_fast_experiment=False,
                                              epochs=epochs[i], bs=bs)
        score.append(df_results_top_k['score'][0])
    print('epochs', epochs)
    print('score', score)

    data = pd.DataFrame([epochs, score])
    data_T = data.transpose()
    data_T.columns = ['epochs', 'score']


    if parameters_of_variation == 'epochs':
        path = ('../../results/score/Batool/parameters_variation/epochs/'+label_input_type+'/'+connection_type+'/')
    if parameters_of_variation == 'batch_size':
        path = ('../../results/score/Batool/parameters_variation/batch_size/'+label_input_type+'/'+connection_type+'/'+str(bs)+'/')

    name = 'results_batool_epochs_variation.csv'
    data_T.to_csv(path + name, index=False)


    a=0
def read_results_param_var_epochs(label_input_type='coord'):
    reduce_samples_for_train = True
    connection_type = 'ALL'
    path = ('../../results/score/Batool/parameters_variation/epochs/' + label_input_type + '/' + connection_type + '/')
    name = 'results_batool_epochs_variation.csv'

    df_ALL = pd.read_csv(path + name, delimiter=',')
    connection_type = 'LOS'
    path = ('../../results/score/Batool/parameters_variation/epochs/' + label_input_type + '/' + connection_type + '/')

    df_LOS = pd.read_csv(path + name, delimiter=',')
    connection_type = 'NLOS'
    path = ('../../results/score/Batool/parameters_variation/epochs/' + label_input_type + '/' + connection_type + '/')

    df_NLOS = pd.read_csv(path + name, delimiter=',')

    return df_ALL, df_LOS, df_NLOS

def read_results_param_var_batch_size_epochs(label_input_type='coord'):
    connection_type = ['ALL', 'LOS', 'NLOS']
    batch_size = [16, 32, 64, 128]
    name = 'results_batool_epochs_variation.csv'
    path = ('../../results/score/Batool/parameters_variation/batch_size/' + label_input_type + '/' )

    for i in range(len(connection_type)):
        all_path = (path + str(connection_type[i]) + '/')
        score = pd.DataFrame ()
        for j in range(len(batch_size)):
            batch_path = (all_path + str(batch_size[j]) + '/')
            df = pd.read_csv(batch_path + name, delimiter=',')
            score = pd.concat([score, df['score']], axis=1)
            #score.columns = batch_size
        if i == 0:
            score.columns = batch_size
            all_score = score
            all_score['epochs'] = df['epochs']
        elif i == 1:
            score.columns = batch_size
            los_score = score
            los_score ['epochs'] = df ['epochs']
        elif i == 2:
            score.columns = batch_size
            nlos_score = score
            nlos_score ['epochs'] = df ['epochs']

    return all_score, los_score, nlos_score

def plot_results_param_var_batch_size_epochs(label_input_type='coord'):
    import matplotlib.pyplot as plt
    df_ALL, df_LOS, df_NLOS = read_results_param_var_batch_size_epochs(label_input_type=label_input_type)
    connection_type = 'ALL'
    if connection_type == 'ALL':
        data = df_ALL
    if connection_type == 'LOS':
        data = df_LOS
    if connection_type == 'NLOS':
        data = df_NLOS

    plt.plot (data['epochs'], data[16], marker='o', label='bs=16')
    plt.plot (data['epochs'], data[32], marker='o', label='bs=32')
    plt.plot (data['epochs'], data[64], marker='o', label='bs=64')
    plt.plot (data['epochs'], data[128], marker='o', label='bs=128')



    plt.legend()
    plt.grid( linestyle = '--', linewidth = 0.5)
    #plt.ylim(0, 0.3)
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.title('Modelo Batool com coord e dados '+ connection_type +'. \n - Variação de Épocas e tamanho do batch -')
    plt.ylim(0, 0.25)

    plt.xticks(data['epochs'])
    #plt.show()
    plt.savefig('../../results/score/Batool/parameters_variation/batch_size/' + label_input_type + '/'
                + connection_type +'batch_size_variation.png', dpi=300, bbox_inches='tight')


def plot_results_param_var_epochs(label_input_type='coord'):
    import matplotlib.pyplot as plt
    df_ALL, df_LOS, df_NLOS = read_results_param_var_epochs(label_input_type=label_input_type)

    plt.plot (df_ALL['epochs'], df_ALL['score'], marker='o', label='ALL')
    plt.plot (df_LOS['epochs'], df_LOS['score'], marker='o', label='LOS')
    plt.plot (df_NLOS['epochs'], df_NLOS['score'], marker='o', label='NLOS')
    plt.legend()
    plt.grid( linestyle = '--', linewidth = 0.5)
    plt.ylim(0, 0.3)
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.title('Modelo Batool com coord. \n - Variação de Épocas -')
    #plt.show()
    plt.savefig('../../results/score/Batool/parameters_variation/epochs/' + label_input_type + '/' + 'epochs_variation.png', dpi=300, bbox_inches='tight')


#plot_results_param_var_batch_size_epochs()
#plot_results_param_var_epochs(label_input_type='coord')
parameters_variation(label_input_type='lidar', connection_type='NLOS',bs=32)
parameters_variation(label_input_type='lidar', connection_type='ALL',bs=32)