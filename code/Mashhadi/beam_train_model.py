import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np

from dataloader import LidarDataset2D
from models import Lidar2D

import argparse

'''
parser = argparse.ArgumentParser()
parser.add_argument("--lidar_training_data",  '+', type=str, help="LIDAR training data file, if you want to merge multiple"
                                                                       " datasets, simply provide a list of paths, as follows:"
                                                                       " --lidar_training_data path_a.npz path_b.npz")
parser.add_argument("--beam_training_data",  '+', type=str, help="Beam training data file, if you want to merge multiple"
                                                                      " datasets, simply provide a list of paths, as follows:"
                                                                      " --beam_training_data path_a.npz path_b.npz")
parser.add_argument("--lidar_validation_data",  '+', type=str, help="LIDAR validation data file, if you want to merge multiple"
                                                                         " datasets, simply provide a list of paths, as follows:"
                                                                         " --lidar_test_data path_a.npz path_b.npz")
parser.add_argument("--beam_validation_data",  '+', type=str, help="Beam validation data file, if you want to merge multiple"
                                                                        " datasets, simply provide a list of paths, as follows:"
                                                                        " --beam_test_data path_a.npz path_b.npz")
parser.add_argument("--model_path", type=str, default='test_model', help="Path, where the trained model will be saved") = parser.parse )
'''

lidar_training_data='lidar_train_raymobtime.npz'
beam_training_data = 'beams_output_train.npz'
lidar_validation_data = 'lidar_validation_raymobtime.npz'
beam_validation_data = 'beams_output_validation.npz'
model_path = 'model/'


if __name__ == '__main__':
    train_data = LidarDataset2D(lidar_training_data, beam_training_data)

    if lidar_validation_data is None and beam_validation_data is None:
        lidar_validation_data = lidar_training_data
        beam_validation_data = beam_training_data

    validation_data = LidarDataset2D(lidar_validation_data, beam_validation_data)

    train_data.lidar_data = np.transpose(train_data.lidar_data, (0, 2, 3, 1))
    validation_data.lidar_data = np.transpose(validation_data.lidar_data, (0, 2, 3, 1))

    model = Lidar2D
    loss_fn = lambda y_true, y_pred: -tf.reduce_sum(tf.reduce_mean(y_true[y_pred>0] * tf.math.log(y_pred[y_pred>0]), axis=0))

    # metrics
    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_categorical_accuracy', dtype=None)
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_categorical_accuracy', dtype=None)
    top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_categorical_accuracy', dtype=None)

    optim = optimizers.Adam(lr=1e-3, epsilon=1e-8)

    scheduler = lambda epoch, lr: lr if epoch < 10 else lr/10.
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model.compile(optimizer=optim, loss=loss_fn, metrics=[top1, top5, top10])
    #model.fit(train_data.lidar_data, train_data.beam_output, callbacks=callback, batch_size=16, epochs=20)
    #model.evaluate(validation_data.lidar_data, validation_data.beam_output)
    model.fit (train_data.lidar_data, train_data.beamOutput, callbacks=callback, batch_size=16, epochs=20)
    model.evaluate (validation_data.lidar_data, validation_data.beamOutput)

    model.save_weights(model_path+'model_weights_new.h5', save_format='h5')
