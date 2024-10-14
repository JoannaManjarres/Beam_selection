import os
import time
import random
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
        X_coord_train = data_train[0]
        X_lidar_train = data_train[1]
        y_train = data_train[2]
        X_coord_validation = data_validation[0]
        X_lidar_validation = data_validation[1]
        y_validation = data_validation[2]
        X_coord_test = data_test[0]
        X_lidar_test = data_test[1]
        y_test = data_test[2]



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

    lidar_train = X_lidar_train
    lidar_validation = X_lidar_validation
    lidar_test = X_lidar_test
    coord_train = X_coord_train
    coord_validation = X_coord_validation
    coord_test = X_coord_test

    if multimodal == 2:
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

def train_model(input, model, data_train, data_validation):
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
            print ('All shapes', X_coord_train.shape, y_train.shape, X_coord_validation.shape, y_validation.shape)
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
            model.summary()

            call_backs = []
            if train_or_test == 'train':
                print('***************Training************')
                star_trainning = time.process_time_ns ()

                hist = model.fit(X_coord_train, y_train,
                                 validation_data=(X_coord_validation, y_validation),
                                 epochs=epochs, batch_size=bs, shuffle=shuffle,
                                 callbacks=[tf.keras.callbacks.ModelCheckpoint(model_folder+'best_weights.coord.h5',
                                                                               monitor='val_loss', verbose=1,
                                                                               save_best_only=True,mode='auto'),
                                            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                             patience=15, verbose=2, mode='auto')])

                end_trainning = time.process_time_ns()
                trainning_process_time = (end_trainning - star_trainning)
                #print("trainning Time: ", trainning_process_time)
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

    print("trainning Time: ", trainning_process_time)

def test_model(input, model, data_test):
    x_test = [data_test[1], data_test[0]]
    y_test = data_test[2]

    lr = 0.0001  # default learning rate
    opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model_folder = 'models/'

    if input == 'lidar_coord':
        print ('***************Testing************')
        model.load_weights (model_folder + 'best_weights.coord_lidar.h5', by_name=True)
        scores = model.evaluate (x_test, y_test)
        print ("----------------------------------")
        print ("Test results:", model.metrics_names, scores)
        print ("----------------------------------")
        print ("----------------------------------")

        # top_k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
        top_k = np.arange (1, 51, 1)
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

    elif input == 'lidar':
        X_lidar_test = data_test[0]
        y_test = data_test[1]

        print ('***************Testing************')
        model.load_weights (model_folder + 'best_weights.lidar.h5', by_name=True)  # to be added
        scores = model.evaluate (X_lidar_test, y_test)
        print ("############ ----------------------------- ")

        print ("Test results", model.metrics_names, scores)

        top_k = [1, 5, 10]
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

    elif input == 'coord':
        X_coord_test = data_test[0]
        y_test = data_test[1]

        print ('***************Testing************')
        model.load_weights (model_folder + 'best_weights.coord.h5', by_name=True)
        print ('***************Testing************')
        model.load_weights (model_folder + 'best_weights.coord.h5', by_name=True)  # to be added
        scores = model.evaluate (X_coord_test, y_test)
        print ("############ ----------------------------- ")

        print ("Test results", model.metrics_names, scores)

        # top_k = [1, 5, 10]
        top_k = np.arange (1, 51, 1)
        accuracy_top_k = []
        process_time = []
        index_predict = []
        for i in range (len (top_k)):
            model_metrics = [metrics.CategoricalAccuracy (), metrics.TopKCategoricalAccuracy (k=top_k [i])]
            model.compile (loss=categorical_crossentropy, optimizer=opt, metrics=model_metrics)
            # model.load_weights (model_folder + 'best_weights.coord.h5', by_name=True)

            ### Testing
            star_test = time.process_time_ns ()
            out = model.evaluate (X_coord_test, y_test)
            end_test = time.process_time_ns ()
            delta = end_test - star_test
            accuracy_top_k.append (out [2])
            process_time.append (delta)
            print ("top-k: ", top_k [i])

        print ('top-k = ', top_k)
        print ("Acuracy =", accuracy_top_k)
        print ("process time: ", process_time)

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


print("online learning Batool...")
input = 'lidar_coord'
data_train, data_validation, data_test, num_classes = prepare_data(input)
model = model_configuration(input, data_train, data_validation, data_test, num_classes)
test_model(input, model, data_test)
a=0