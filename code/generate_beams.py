import numpy as np
import csv
import seaborn as sns
import pandas as pd
import mimo_best_beams
import random

random.seed(0)

def generated_beams_output_from_ray_tracing():
    # BEAMS DE TREINAMENTO
    #inputPath = '../data/beams_output/dataset_s008/beam_output_generate_rt_s008/ray_tracing_data_s008_carrier60GHz/rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e'
    inputPath = '../data/ray_tracing_data_s008_carrier60GHz/rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e'
    insiteCSVFile = '../data/coord/CoordVehiclesRxPerScene_s008.csv'
    numEpisodes = 2086  # total number of episodes
    outputFolder = '../data/beams_output/beams_generate_by_me/'

    #mimo_best_beams.processBeamsOutput(csvFile=insiteCSVFile, num_episodes=numEpisodes, outputFolder=outputFolder, inputPath=inputPath)

    # BEAMS DE TESTE
    inputPath = '../data/ray_tracing_data_s009_carrier60GHz/rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e'
    insiteCSVFile = '../data/coord/CoordVehiclesRxPerScene_s009.csv'
    numEpisodes = 2000  # total number of episodes
    outputFolder = '../data/beams_output/beams_generate_by_me/'

    mimo_best_beams.processBeamsOutput(csvFile=insiteCSVFile, num_episodes=numEpisodes, outputFolder=outputFolder, inputPath=inputPath)

def read_beams_output_generated_by_ray_tracing():
    path = '../data/beams_output/beams_generate_by_me/'
    beam_output_test = np.load(path + "beams_output_8x32_test.npz", allow_pickle=True)['output_classification']
    print("\t\tGeracao de Beams pelo Ray-tracing para teste ")
    index_beams_test = calculate_index_beams(beam_output_test)

    path = '../data/beams_output/beams_generate_by_me/'
    beam_output_train = np.load (path + "beams_output_8x32_train.npz", allow_pickle=True) ['output_classification']
    print ("\t\tGeracao de Beams pelo Ray-tracing para treinamento")
    index_beams_train = calculate_index_beams (beam_output_train)

    path = '../data/beams_output/dataset_s008/beam_output_ref_17/'
    beam_output = np.load (path + "beams_output_train.npz", allow_pickle=True)
    keys = list (beam_output.keys ())
    index_beams_r3_train = calculate_index_beams(beam_output [keys [0]])

    path = '../data/beams_output/dataset_s008/beam_output_ref_17/'
    beam_output = np.load (path + "beams_output_validation.npz", allow_pickle=True)
    keys = list (beam_output.keys ())
    index_beams_r3_val = calculate_index_beams (beam_output [keys [0]])

    index_beams_r3 = np.concatenate((index_beams_r3_train, index_beams_r3_val), axis=0).tolist()

    #beam_output = np.load (path + "best_beam_index.npz", allow_pickle=True) ['output_classification']
    #print ("\t\tGeracao de Beams pelo Ray-tracing")
    #index_beams_r3 = calculate_index_beams (beam_output)

    return index_beams_train, index_beams_test
def calculate_index_beams(beam_output):
    # calculate the index of the best beam
    tx_size = beam_output.shape[2]
    #print(beam_output.shape)

    # Reshape beam pair index
    num_classes = beam_output.shape[1] * beam_output.shape[2]
    beams = beam_output.reshape(beam_output.shape[0], num_classes)

    # Beams distribution
    best_beam_index = []
    for sample in range(beams.shape[0]):
        best_beam_index.append(np.argmax(beams[sample, :]))

    return(best_beam_index)


def compare_index_generate():
    index_beams_r1, index_beams_r2 = read_beams_output_generated_by_ray_tracing ()

    a = 0
    for i in range (len(index_beams_r1)):
        if index_beams_r1 [i] != index_beams_r2[i]: # != index_beams_r3 [i]:
            a = a+1

    print (a, "Beams Diferentes")

    a =0

compare_index_generate()
#generated_beams_output_from_ray_tracing()