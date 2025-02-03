# Imports

import argparse
from pathlib import Path

import callbacks
import numpy as np
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import layers, metrics, optimizers, losses, callbacks, regularizers
from tensorflow.keras.models import load_model, save_model
import tensorflow.keras.backend as K

from utils import set_seed, OneCycleLR, show_history, tic, toc
from resnet import conv_block, residual_body, stem
from beam_utils import load_data, get_beams_output

from tensorflow.keras.models import model_from_json
from resnet import AddRelu
import pandas as pd


# set random seeds and return numpy random generator:
set_seed (123)
def normalize_data(X, means=None, stds=None):
    if means is None: means = np.mean(X, axis=0)
    if stds is None: stds = np.std(X, axis=0)
    X_norm = (X - means) / stds
    return X_norm, means, stds
def process_coordinates(X, means=None, stds=None):
    X_xyz, means, stds = normalize_data(X[:, :3], means, stds)
    X = np.concatenate((X_xyz, X[:, 3:]), axis=1)
    return X, means, stds

def load_data_test(filename):
    cache = np.load (filename, allow_pickle=True)
    keys = list (cache.keys())
    X = cache[keys[0]]

    return X
class DataGenerator_coord(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, shuffle=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = self.x[indexes]
        Y = self.y[indexes]
        return X, Y
class DataGenerator_both(keras.utils.Sequence):
    def __init__(self, x1, x2, y, batch_size, shuffle=False):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x1))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x1) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = (self.x1[indexes], self.x2[indexes])
        Y = self.y[indexes]

        return X, Y
class DataGenerator_lidar(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, shuffle=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = self.x[indexes]
        Y = self.y[indexes]
        return X, Y


def pre_process_data_train(type_input):
    data_folder = '../../data/ruseckas/Raymobtime_s008'
    DATA_DIR = data_folder + '/baseline_data/'
    LIDAR_DATA_DIR = DATA_DIR + 'lidar_input/'
    COORDINATES_DATA_DIR = DATA_DIR + 'coord_input/'
    IMAGES_DATA_DIR = DATA_DIR + 'image_input/'
    BEAMS_DATA_DIR = DATA_DIR + 'beam_output/'

    LIDAR_TRAIN_FILE = LIDAR_DATA_DIR + 'lidar_train.npz'
    LIDAR_VAL_FILE = LIDAR_DATA_DIR + 'lidar_validation.npz'

    COORDINATES_TRAIN_FILE = COORDINATES_DATA_DIR + 'my_coord_train.npz'
    COORDINATES_VAL_FILE = COORDINATES_DATA_DIR + 'my_coord_validation.npz'

    BEAMS_TRAIN_FILE = BEAMS_DATA_DIR + 'beam_output_train.npz'
    BEAMS_VAL_FILE = BEAMS_DATA_DIR + 'beam_output_validation.npz'

    BATCH_SIZE = 32


    # set random seeds and return numpy random generator:
    set_seed (123)


    if type_input == 'lidar':
        X_lidar_train = load_data (LIDAR_TRAIN_FILE, 'input')
        X_lidar_val = load_data (LIDAR_VAL_FILE, 'input')

        Y_train, num_classes = get_beams_output (BEAMS_TRAIN_FILE)
        Y_val, _ = get_beams_output (BEAMS_VAL_FILE)

        train_generator = DataGenerator_lidar (X_lidar_train, Y_train, BATCH_SIZE, shuffle=True)
        val_generator = DataGenerator_lidar (X_lidar_val, Y_val, BATCH_SIZE)

        return train_generator, val_generator, num_classes, X_lidar_train.shape [1:]
    if type_input == 'coord':
        X_coord_train = load_data (COORDINATES_TRAIN_FILE, 'coordinates')
        X_coord_val = load_data (COORDINATES_VAL_FILE, 'coordinates')

        X_coord_train, coord_means, coord_stds = process_coordinates (X_coord_train)

        np.savez ('coord_train_stats.npz', coord_means=coord_means, coord_stds=coord_stds)

        X_coord_val, _, _ = process_coordinates (X_coord_val, coord_means, coord_stds)

        Y_train, num_classes = get_beams_output (BEAMS_TRAIN_FILE)
        Y_val, _ = get_beams_output (BEAMS_VAL_FILE)

        train_generator = DataGenerator_coord (X_coord_train, Y_train, BATCH_SIZE, shuffle=True)
        val_generator = DataGenerator_coord (X_coord_val, Y_val, BATCH_SIZE)

        return train_generator, val_generator, num_classes, X_coord_train.shape [1:]
    elif type_input == 'lidar_coord':
        X_lidar_train = load_data (LIDAR_TRAIN_FILE, 'input')
        X_lidar_val = load_data (LIDAR_VAL_FILE, 'input')

        X_coord_train = load_data (COORDINATES_TRAIN_FILE, 'coordinates')
        X_coord_val = load_data (COORDINATES_VAL_FILE, 'coordinates')

        X_coord_train, coord_means, coord_stds = process_coordinates (X_coord_train)

        np.savez ('coord_train_stats.npz', coord_means=coord_means, coord_stds=coord_stds)

        X_coord_val, _, _ = process_coordinates (X_coord_val, coord_means, coord_stds)

        Y_train, num_classes = get_beams_output (BEAMS_TRAIN_FILE)
        Y_val, _ = get_beams_output (BEAMS_VAL_FILE)

        train_generator = DataGenerator_both (X_lidar_train, X_coord_train, Y_train, BATCH_SIZE, shuffle=True)
        val_generator = DataGenerator_both (X_lidar_val, X_coord_val, Y_val, BATCH_SIZE)

        return train_generator, val_generator, num_classes, [X_lidar_train.shape [1:], X_coord_train.shape [1:]]
def create_model_lidar(inp, params):
    y = stem([32, 32, 64], strides=(1, 2), **params)(inp)
    y = residual_body(64, [2, 2], [2, 2], **params)(y)

    y = conv_block(8, kernel_size=1, **params)(y)
    y = layers.Flatten()(y)
    y = layers.Dropout(0.25)(y)
    y = layers.Dense(256, activation='relu', **params)(y)

    return y
def create_model_coord(inp, params):
    y = layers.Dense(8, activation='relu', **params)(inp)
    y = layers.Dense(16, activation='relu', **params)(y)
    y = layers.Dense(64, activation='relu', **params)(y)
    y = layers.Dense(256, activation='relu', **params)(y)

    return y
def create_model_for_both(input_shape_lidar, input_shape_coord, classes, params):
    inp_lidar = keras.Input(shape=input_shape_lidar)
    inp_coord = keras.Input(shape=input_shape_coord)

    out_lidar = create_model_lidar(inp_lidar, params)
    out_coord = create_model_coord(inp_coord, params)

    y = layers.Concatenate()([out_lidar, out_coord])
    y = layers.Dropout(0.5)(y)
    out = layers.Dense(classes, activation='softmax', **params)(y)

    model = keras.Model(inputs = [inp_lidar, inp_coord], outputs = out)
    return model
def create_model_for_lidar(input_shape_lidar, classes, params):
    inp_lidar = keras.Input(shape=input_shape_lidar)
    out_lidar = create_model_lidar(inp_lidar, params)

    y = layers.Dropout(0.25)(out_lidar)
    out = layers.Dense(classes, activation='softmax', **params)(y)
    model = keras.Model(inputs = inp_lidar, outputs = out)
    return model
def create_model_for_coord(input_shape_coord, classes, params):
    inp_coord = keras.Input(shape=input_shape_coord)
    out_coord = create_model_coord(inp_coord, params)
    y = layers.Dropout(0.25)(out_coord)
    out = layers.Dense(classes, activation='softmax', **params)(y)
    model = keras.Model(inputs = inp_coord, outputs = out)
    return model
def read_data_for_test(type_input):
    DATA_DIR = '../../data/'
    LIDAR_DATA_DIR = DATA_DIR + 'lidar/s009/'
    COORDINATES_DATA_DIR = DATA_DIR + 'coord/ruseckas/'

    BEAMS_DATA_DIR = DATA_DIR + 'beams_output/beam_output_baseline_raymobtime_s009/'

    LIDAR_TEST_FILE = LIDAR_DATA_DIR + 'lidar_test_raymobtime.npz'
    COORDINATES_TEST_FILE = COORDINATES_DATA_DIR + 'my_coord_test.npz'

    BEAMS_TEST_FILE = BEAMS_DATA_DIR + 'beams_output_test.npz'
    index_true, _ = get_beams_output (BEAMS_TEST_FILE)

    if type_input == 'coord':
        X_coord_test = load_data_test (COORDINATES_TEST_FILE)
        cache = np.load (COORDINATES_DATA_DIR + 'coord_train_stats.npz', )
        coord_means = cache['coord_means']
        coord_stds = cache ['coord_stds']
        a=0
        X_coord_test, _, _ = process_coordinates (X_coord_test, coord_means, coord_stds)
        X_data = X_coord_test
    if type_input == 'lidar':
        X_lidar_test = load_data_test(LIDAR_TEST_FILE)
        X_data = X_lidar_test

    if type_input == 'lidar_coord':
        X_lidar_test = load_data_test(LIDAR_TEST_FILE)

        X_coord_test = load_data_test(COORDINATES_TEST_FILE)

        cache = np.load (COORDINATES_DATA_DIR + 'coord_train_stats.npz')
        coord_means = cache ['coord_means']
        coord_stds = cache ['coord_stds']
        X_coord_test, _, _ = process_coordinates (X_coord_test, coord_means, coord_stds)
        X_data = (X_lidar_test, X_coord_test)

    return X_data, index_true
def criate_model_for_test(type_input):
    folder_model = 'model_' + type_input + '/'

    if type_input == 'lidar':
        model = 'my_model_weights_lidar.h5'
        model_json = 'my_model_lidar.json'
    if type_input == 'coord':
        model = 'my_model_weights_coord.h5'
        model_json = 'my_model_coord.json'
    elif type_input == 'lidar_coord':
        model = 'my_model_weights_both.h5'
        model_json = 'my_model_both.json'

    BEST_WEIGTHS = folder_model + model
    with open (folder_model + model_json, 'r') as json_file:
        loaded_model_json = json_file.read ()
        model = model_from_json (loaded_model_json, custom_objects={'AddRelu': AddRelu})

    return model, BEST_WEIGTHS


def train_model_1(type_input):

    KERNEL_REG = 1.e-4
    params = {"kernel_regularizer": regularizers.l2(KERNEL_REG)}

    train_generator, val_generator, num_classes, x_train_shape = pre_process_data_train(type_input)
    if type_input == 'lidar':
        model = create_model_for_lidar(x_train_shape, num_classes, params)
        BEST_WEIGTHS = 'model_lidar/my_model_weights_lidar.h5'
    if type_input == 'coord':
        model = create_model_for_coord (x_train_shape, num_classes, params)
        BEST_WEIGTHS = 'model_coord/my_model_weights_coord.h5'
    elif type_input == 'lidar_coord':
        model = create_model_for_both (x_train_shape[0], x_train_shape[1], num_classes, params)
        BEST_WEIGTHS = 'model_lidar_coord/my_model_weights_both.h5'

    #model.summary(line_length=128)

    # ## Training
    K.clear_session()

    # optim = optimizers.Adam()
    optim = optimizers.legacy.Adam()

    model_metrics = [metrics.CategoricalAccuracy(), metrics.TopKCategoricalAccuracy(k=3)]

    model_loss = losses.CategoricalCrossentropy()

    model.compile(loss=model_loss, optimizer=optim, metrics=model_metrics)

    EPOCHS = 50

    one_cycle_sheduler = OneCycleLR (max_lr=1e-2, total_steps=EPOCHS * len (train_generator))

    checkpoint = keras.callbacks.ModelCheckpoint(BEST_WEIGTHS, monitor='val_top_k_categorical_accuracy',
                                            verbose=0, save_best_only=True,
                                            save_weights_only=True, mode='max', save_freq='epoch')


    tb_log = keras.callbacks.TensorBoard(log_dir='./logs')

    callbacks = [one_cycle_sheduler, checkpoint]

    tic()
    hist = model.fit (train_generator,
                      validation_data=val_generator,
                      epochs=EPOCHS,
                      callbacks=callbacks,
                      verbose=0)
    delta_time = toc()

    model.save_weights(BEST_WEIGTHS, save_format='h5')

    # ## Evaluation
    model.load_weights(BEST_WEIGTHS)

    model.evaluate(val_generator, verbose=1)

    model_json = model.to_json()
    if type_input == 'lidar':
        with open('model_lidar/my_model_lidar.json', "w") as json_file:
            json_file.write(model_json)
    if type_input == 'coord':
        with open('model_coord/my_model_coord.json', "w") as json_file:
            json_file.write(model_json)
    elif type_input == 'lidar_coord':
        with open('model_lidar_coord/my_model_both.json', "w") as json_file:
            json_file.write(model_json)

    print ("Training time:", delta_time)

def read_all_data():

    filename = '../../data/coord/CoordVehiclesRxPerScene_s008.csv'
    all_csv_data = pd.read_csv (filename)
    valid_data = all_csv_data [all_csv_data ['Val'] == 'V']
    limit_ep_train = 1564

    train_data = valid_data [valid_data ['EpisodeID'] <= limit_ep_train]
    coord_for_train = np.zeros ((len (train_data), 3))
    coord_for_train[:, 0] = train_data['x']
    coord_for_train[:, 1] = train_data['y']
    coord_for_train[:, 2] = train_data['z']
    coord_train, _coord_means, _coord_stds = process_coordinates(coord_for_train)

    validation_data = valid_data [valid_data ['EpisodeID'] > limit_ep_train]
    coord_for_validation = np.zeros ((len (validation_data), 3))
    coord_for_validation [:, 0] = validation_data ['x']
    coord_for_validation [:, 1] = validation_data ['y']
    coord_for_validation [:, 2] = validation_data ['z']
    coord_validation, _, _ = process_coordinates (coord_for_validation, _coord_means, _coord_stds)

    filename = '../../data/coord/CoordVehiclesRxPerScene_s009.csv'
    all_csv_data = pd.read_csv(filename)
    valid_data_test = all_csv_data[all_csv_data['Val'] == 'V']

    coord_for_test = np.zeros((len(valid_data_test), 3))


    coord_for_test[:, 0] = valid_data_test['x']
    coord_for_test[:, 1] = valid_data_test['y']
    coord_for_test[:, 2] = valid_data_test['z']
    coord_test, _, _ = process_coordinates(coord_for_test, _coord_means, _coord_stds)


    path = '../../data/lidar/s008/'
    LIDAR_TRAIN_FILE = path + 'lidar_train_raymobtime.npz'
    lidar_train = load_data (LIDAR_TRAIN_FILE, 'input')
    LIDAR_VAL_FILE = path + 'lidar_validation_raymobtime.npz'
    lidar_validation = load_data (LIDAR_VAL_FILE, 'input')
    path = '../../data/lidar/s009/'
    LIDAR_TEST_FILE = path + 'lidar_test_raymobtime.npz'
    lidar_test = load_data (LIDAR_TEST_FILE, 'input')

    lidar_train_reshaped = lidar_train.reshape (9234, -1)
    lidar_validation_reshaped = lidar_validation.reshape (1960, -1)
    lidar_test_reshaped = lidar_test.reshape (9638, -1)

    BEAMS_TRAIN_FILE = '../../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_train.npz'
    Y_train, num_classes = get_beams_output(BEAMS_TRAIN_FILE)

    BEAMS_VAL_FILE = '../../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_validation.npz'
    Y_val, _ = get_beams_output(BEAMS_VAL_FILE)

    BEAMS_TEST_FILE = '../../data/beams_output/beam_output_baseline_raymobtime_s009/beams_output_test.npz'
    Y_test, _ = get_beams_output(BEAMS_TEST_FILE)



    data_for_train = pd.DataFrame ({"Episode": train_data ['EpisodeID'],
                                    "coord": coord_train.tolist(),
                                    "lidar": lidar_train_reshaped.tolist(),
                                    "index_beams": Y_train.tolist()})


    data_for_validation = pd.DataFrame ({"Episode": validation_data['EpisodeID'],
                                         "coord": coord_validation.tolist(),
                                         "lidar": lidar_validation_reshaped.tolist(),
                                         "index_beams": Y_val.tolist()})

    data_for_test = pd.DataFrame ({"Episode": valid_data_test['EpisodeID'],
                                   "coord": coord_test.tolist(),
                                   "lidar": lidar_test_reshaped.tolist(),
                                   "index_beams": Y_test.tolist()})


    return data_for_train, data_for_validation, data_for_test, num_classes









def train_model(type_input, train_generator, val_generator, num_classes, x_train_shape):
    KERNEL_REG = 1.e-4
    params = {"kernel_regularizer": regularizers.l2 (KERNEL_REG)}

    if type_input == 'lidar':
        model = create_model_for_lidar(x_train_shape, num_classes, params)
        BEST_WEIGTHS = 'model_lidar/my_model_weights_lidar.h5'
    if type_input == 'coord':
        model = create_model_for_coord (x_train_shape, num_classes, params)
        BEST_WEIGTHS = 'model_coord/my_model_weights_coord.h5'
    elif type_input == 'lidar_coord':
        model = create_model_for_both (x_train_shape[0], x_train_shape[1], num_classes, params)
        BEST_WEIGTHS = 'model_lidar_coord/my_model_weights_both.h5'

    #model.summary(line_length=128)

    # ## Training
    K.clear_session()

    # optim = optimizers.Adam()
    optim = optimizers.legacy.Adam()

    model_metrics = [metrics.CategoricalAccuracy(), metrics.TopKCategoricalAccuracy(k=3)]

    model_loss = losses.CategoricalCrossentropy()

    model.compile(loss=model_loss, optimizer=optim, metrics=model_metrics)

    EPOCHS = 50

    one_cycle_sheduler = OneCycleLR (max_lr=1e-2, total_steps=EPOCHS * len (train_generator))

    checkpoint = keras.callbacks.ModelCheckpoint(BEST_WEIGTHS, monitor='val_top_k_categorical_accuracy',
                                            verbose=0, save_best_only=True,
                                            save_weights_only=True, mode='max', save_freq='epoch')


    tb_log = keras.callbacks.TensorBoard(log_dir='./logs')

    callbacks = [one_cycle_sheduler, checkpoint]

    tic()
    hist = model.fit (train_generator,
                      validation_data=val_generator,
                      epochs=EPOCHS,
                      callbacks=callbacks,
                      verbose=0)
    delta_time = toc()

    model.save_weights(BEST_WEIGTHS, save_format='h5')

    # ## Evaluation
    model.load_weights(BEST_WEIGTHS)

    model.evaluate(val_generator, verbose=0)

    model_json = model.to_json()
    if type_input == 'lidar':
        with open('model_lidar/my_model_lidar.json', "w") as json_file:
            json_file.write(model_json)
    if type_input == 'coord':
        with open('model_coord/my_model_coord.json', "w") as json_file:
            json_file.write(model_json)
    elif type_input == 'lidar_coord':
        with open('model_lidar_coord/my_model_both.json', "w") as json_file:
            json_file.write(model_json)

    print ("Training time:", delta_time)

    return delta_time, x_train_shape




def test_model_1(type_input):

    save_index_predict = 'False'
    X_data, index_true = read_data_for_test(type_input)
    model, BEST_WEIGTHS = criate_model_for_test(type_input)
    optim = optimizers.legacy.Adam()

    top_k = np.arange (1, 31, 1)

    all_score = []
    process_time = []
    for i in range (len (top_k)):
        model_metrics = [metrics.CategoricalAccuracy (), metrics.TopKCategoricalAccuracy (k=top_k [i])]
        model_loss = losses.CategoricalCrossentropy ()
        model.compile (loss=model_loss, optimizer=optim, metrics=model_metrics)
        model.load_weights(BEST_WEIGTHS)
        #model.summary()

        tic ()
        out = model.evaluate (x=X_data, y=index_true, verbose=0)
        delta_time = toc ()
        all_score.append (out [2])
        process_time.append (delta_time)

    all_index_predict = model.predict(X_data, verbose=0)
    all_index_predict_order = np.zeros ((all_index_predict.shape [0], all_index_predict.shape [1]))

    for i in range (len (all_index_predict)):
        all_index_predict_order [i] = np.flip (np.argsort (all_index_predict [i]))

    if save_index_predict:
        path_index_predict = '../../results/index_beams_predict/ruseckas/top_k/' + type_input + '/'
        file_name = 'index_beams_predict_' + type_input + '_top_k.npz'
        npz_index_predict = path_index_predict + file_name
        np.savez (npz_index_predict, index_predict=all_index_predict_order)

    df_results_top_k = pd.DataFrame ({"top-k": top_k,
                                      "score": all_score,
                                      "test_time": process_time,
                                      "samples_tested": index_true.shape [0]})
    return df_results_top_k, all_index_predict_order

def test_model(type_input, X_data, index_true, episode):
    save_index_predict = 'False'
    #X_data, index_true = read_data_for_test (type_input)
    model, BEST_WEIGTHS = criate_model_for_test(type_input)
    optim = optimizers.legacy.Adam()

    top_k = np.arange(1, 31, 1)

    all_score = []
    process_time = []
    for i in range (len (top_k)):
        model_metrics = [metrics.CategoricalAccuracy (), metrics.TopKCategoricalAccuracy (k=top_k [i])]
        model_loss = losses.CategoricalCrossentropy ()
        model.compile (loss=model_loss, optimizer=optim, metrics=model_metrics)
        model.load_weights (BEST_WEIGTHS)
        # model.summary()

        tic ()
        out = model.evaluate (x=X_data, y=index_true, verbose=0)
        delta_time = toc ()
        all_score.append (out [2])
        process_time.append (delta_time)

    all_index_predict = model.predict (X_data, verbose=0)
    all_index_predict_order = np.zeros ((all_index_predict.shape [0], all_index_predict.shape [1]))

    for i in range (len (all_index_predict)):
        all_index_predict_order[i] = np.flip(np.argsort(all_index_predict [i]))

    if save_index_predict:
        path_index_predict = '../../results/index_beams_predict/ruseckas/top_k/' + type_input + '/'
        file_name = 'index_beams_predict_' + type_input + '_top_k.npz'
        npz_index_predict = path_index_predict + file_name
        np.savez (npz_index_predict, index_predict=all_index_predict_order)

    df_results_top_k = pd.DataFrame ({"top-k": top_k,
                                      "score": all_score,
                                      "test_time": process_time,
                                      "samples_tested": index_true.shape [0],
                                      "episode": episode})

    df_all_index_predict_order = pd.DataFrame({"index_predict": all_index_predict_order.tolist(),
                                               "episode": episode})
    return df_results_top_k, df_all_index_predict_order
#pre_process_data_train('lidar')