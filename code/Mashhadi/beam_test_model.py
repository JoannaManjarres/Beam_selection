import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np

from dataloader import LidarDataset2D
from models import Lidar2D
import pandas as pd
import timeit

import argparse

def tic():
    global tic_s
    tic_s = timeit.default_timer()
def toc():
    global tic_s
    toc_s = timeit.default_timer()
    return (toc_s - tic_s)

def beam_output_correction(beam_output):
    beam_output_matrix = np.zeros ([beam_output.shape [0], 32, 8], dtype=np.float32)
    # y[i, rx * Tx_size + tx] = codebook [rx, tx]

    yMatrix = np.zeros (([beam_output.shape [0], 8,32]))
    for i in range (0, beam_output.shape [0], 1):  # go over all examples
        codebook = beam_output [i, :]  # read vector
        Rx_size = 8
        Tx_size = 32
        for tx in range(0, Tx_size, 1):
            for rx in range(0, Rx_size, 1):  # inner loop goes over receiver
                yMatrix[i, rx, tx] = beam_output[i, tx * Rx_size + rx]  # impose ordering

    y_vector = yMatrix.reshape(yMatrix.shape[0], 256)


    return y_vector
'''
parser = argparse.ArgumentParser()

parser.add_argument("--lidar_test_data", nargs='+', type=str, help="LIDAR test data file, if you want to merge multiple"
                                                                   " datasets, simply provide a list of paths, as follows:"
                                                                   " --lidar_training_data path_a.npz path_b.npz")
parser.add_argument("--beam_test_data", nargs='+', type=str, default=None,
                    help="Beam test data file, if you want to merge multiple"
                         " datasets, simply provide a list of paths, as follows:"
                         " --beam_training_data path_a.npz path_b.npz")
parser.add_argument("--model_path", type=str, help="Path, where the model is saved")
parser.add_argument("--preds_csv_path", type=str, default="unnamed_preds.csv",
                    help="Path, where the .csv file with the predictions will be saved")

args = parser.parse_args()
'''
input = 'lidar'
#lidar_test_data = 'lidar_test_raymobtime.npz'
#beam_test_data = 'beams_output_test.npz'

lidar_test_data = '../../data/lidar/s009/lidar_test_raymobtime.npz'
#test_data_folder = '../../data/beams_output/beams_generate_by_me/'
#file_test = 'beams_output_8x32_test.npz'
#beam_test_data = test_data_folder + file_test

test_data_folder = '../../data/beams_output/beam_output_baseline_raymobtime_s009/'
file_test = 'beams_output_test.npz'
beam_test_data = test_data_folder + file_test

model_path = 'model/'
preds_npz_path = '../../results/index_beams_predict/Mashhadi/top_k/'+input+'/'
score_csv_path = '../../results/score/Mashhadi/top_k/'+input+'/'
path_to_save_process_time = '../../results/processingTime/Mashhadi/'+input+'/'

if __name__ == '__main__':
    #test_data = LidarDataset2D(args.lidar_test_data, args.beam_test_data)
    test_data = LidarDataset2D (lidar_test_data, beam_test_data)

    test_data.lidar_data = np.transpose(test_data.lidar_data, (0, 2, 3, 1))

    model = Lidar2D
    #model.load_weights(args.model_path)
    model.load_weights(model_path+'model_weights.h5')

    # metrics
    '''
    # Original dos autores
    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_categorical_accuracy', dtype=None)
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_categorical_accuracy', dtype=None)
    top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_categorical_accuracy', dtype=None)

    model.compile(metrics=[top1, top5, top10])
    model.evaluate(test_data.lidar_data, test_data.beam_output)
    '''
    top_k = np.arange (1, 51, 1)
    #top_k = [1, 5, 10]
    accuracy_top_k = []
    process_time = []
    index_predict = []

    for i in range(len(top_k)):
        all_k = tf.keras.metrics.TopKCategoricalAccuracy(k=top_k[i], name='top_'+str(top_k[i])+'_categorical_accuracy', dtype=None)
        model.compile(metrics=[all_k])
        tic()
        out = model.evaluate(test_data.lidar_data, test_data.beam_output)
        #out_1 = model.evaluate(test_data.lidar_data, test_data.beamOutput)
        delta = toc()
        accuracy_top_k.append(out[1])
        process_time.append(delta)

    #y_test = test_data.beam_output
    #y_test = test_data.beamOutput

    all_data_predict = model.predict(test_data.lidar_data, batch_size=100)
    all_data_predict_correct_shape = beam_output_correction(all_data_predict)

    all_index_predict_order = np.zeros ((all_data_predict_correct_shape.shape [0], all_data_predict_correct_shape.shape [1]))
    for i in range (len (all_data_predict_correct_shape)):
        all_index_predict_order[i] = np.flip (np.argsort (all_data_predict_correct_shape [i]))

    test_data_correct_shape = beam_output_correction(test_data.beam_output)


    ## Testanto  a acuracia calculada pelo metodo de avaliacao do keras (evaluate)
    top_1_predict = all_index_predict_order [:, 0].astype (int)
    top_5_predict = all_index_predict_order [:, 0:5].astype (int)
    true_label = []
    for i in range (len(test_data_correct_shape)):
        true_label.append(test_data_correct_shape [i, :].argmax ())

    acerto = 0
    nao_acerto = 0
    acerto_top_5 = 0
    nao_acerto_top_5 = 0

    for sample in range (len (test_data_correct_shape)):
        if (true_label[sample] == top_1_predict [sample]):
            acerto = acerto + 1
        else:
            nao_acerto = nao_acerto + 1

    for sample in range(len(test_data_correct_shape)):
        if (true_label[sample] in top_5_predict[sample]):
            acerto_top_5 = acerto_top_5 + 1
        else:
            nao_acerto_top_5 = nao_acerto_top_5 + 1

    score = acerto / len(all_index_predict_order)
    print ('score top-1: ', score)
    print('score top-5: ', acerto_top_5 / len(all_index_predict_order))


    df_score_top_k = pd.DataFrame ({"Top-K": top_k, "Acuracia": accuracy_top_k})
    df_score_top_k.to_csv (score_csv_path + 'score_' + input + '_top_k.csv', index=False)

    df_test_time = pd.DataFrame ({"test_time": process_time})
    df_test_time.to_csv (path_to_save_process_time + 'test_time_' + input + '.csv', index=False)

    file_name = 'index_beams_predict_top_k.npz'
    npz_index_predict = preds_npz_path + file_name
    np.savez (npz_index_predict, index_predict=all_index_predict_order)