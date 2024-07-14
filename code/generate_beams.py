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
    dataset_name = 'train'

    mimo_best_beams.processBeamsOutput(csvFile=insiteCSVFile,
                                       num_episodes=numEpisodes,
                                       outputFolder=outputFolder,
                                       inputPath=inputPath,
                                       dataset_name=dataset_name)
    print('Generated training beams in'+ outputFolder)

    # BEAMS DE TESTE
    inputPath = '../data/ray_tracing_data_s009_carrier60GHz/rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e'
    insiteCSVFile = '../data/coord/CoordVehiclesRxPerScene_s009.csv'
    numEpisodes = 2000  # total number of episodes
    outputFolder = '../data/beams_output/beams_generate_by_me/'
    dataset_name = 'test'

    mimo_best_beams.processBeamsOutput(csvFile=insiteCSVFile,
                                       num_episodes=numEpisodes,
                                       outputFolder=outputFolder,
                                       inputPath=inputPath,
                                       dataset_name=dataset_name)
    print('Generated test beams'+ outputFolder)

def read_valid_coordinates():
    # filename = '/Users/Joanna/git/Analise_de_dados/data/coordinates/CoordVehiclesRxPerScene_s008.csv'
    filename = '../data/coord/CoordVehiclesRxPerScene_s008.csv'
    limit_ep_train = 1564

    with open (filename) as csvfile:
        reader = csv.DictReader (csvfile)
        number_of_rows = len (list (reader))

    all_info_coord_val = np.zeros ([11194, 5], dtype=object)

    with open (filename) as csvfile:
        reader = csv.DictReader (csvfile)
        cont = 0
        for row in reader:
            if row ['Val'] == 'V':
                all_info_coord_val [cont] = int (row ['EpisodeID']), float (row ['x']), float (row ['y']), float (
                    row ['z']), row ['LOS']
                cont += 1

    # all_info_coord = np.array(all_info_coord)

    coord_train = all_info_coord_val [(all_info_coord_val [:, 0] < limit_ep_train + 1)]
    coord_test = all_info_coord_val [(all_info_coord_val [:, 0] > limit_ep_train)]

    return all_info_coord_val, coord_train, coord_test


def divide_beams_in_train_validation():
    save_data = True
    beams_output_train, beams_output_test = read_beams_output_generated_by_ray_tracing()

    beamoutput_train = beams_output_train[0:9234]
    beamoutput_validation = beams_output_train[9234::]

    '''
    #all_data = np.column_stack((all_coord, beams_output_train))
    coord_and_beam_output = np.zeros((all_coord.shape[0], 2),dtype=object)
    for i in range(len(all_coord)):
        coord_and_beam_output[i] = all_coord[i][0], beams_output_train[i]

    limit_ep_train = 1564

    data_train = coord_and_beam_output [(coord_and_beam_output [:, 0] < limit_ep_train + 1)]
    data_validation = coord_and_beam_output [(coord_and_beam_output [:, 0] > limit_ep_train)]

    beam_output_train = data_train [:, -1]
    beam_output_validation = data_validation [:, -1]

    for i in range(len(beam_output_train)):
        if beam_output_train[i].all() != beamoutput_train[i].all():
            print('not ok')

    for i in range(len(beamoutput_validation)):
        if beam_output_validation[i].all() != beamoutput_validation[i].all():
            print('not ok')
    
    '''

    if save_data:

        save_path = '../data/beams_output/beams_generate_by_me/train_val/'
        np.savez (save_path + 'beam_output_train' + '.npz', output_classification=beamoutput_train)
        np.savez (save_path + 'beam_output_validation' + '.npz', output_classification=beamoutput_validation)



def read_beams_output_generated_by_ray_tracing():
    print ("\t\tRead Beams output generated from Ray-tracing ")
    path = '../data/beams_output/beams_generate_by_me/'
    beam_output_train = np.load (path + "beams_output_8x32_train.npz", allow_pickle=True) ['output_classification']

    path = '../data/beams_output/beams_generate_by_me/'
    beam_output_test = np.load(path + "beams_output_8x32_test.npz", allow_pickle=True)['output_classification']


    return beam_output_train, beam_output_test

def read_index_beams_generated_by_ray_tracing():
    print ("\t\tRead Beams output generated from Ray-tracing ")
    path = '../data/beams_output/beams_generate_by_me/'
    beam_output_train = np.load (path + "beams_output_8x32_train.npz", allow_pickle=True) ['output_classification']
    print("\t\tCalculated Beams Index for training ")
    index_beams_train = calculate_index_beams (beam_output_train)
    index_beams_train_str = [str (i) for i in index_beams_train]

    path = '../data/beams_output/beams_generate_by_me/'
    beam_output_test = np.load(path + "beams_output_8x32_test.npz", allow_pickle=True)['output_classification']
    print("\t\tCalculated Beams Index for test ")
    index_beams_test = calculate_index_beams(beam_output_test)
    index_beams_test_str = [str (i) for i in index_beams_test]



    path = '../data/beams_output/dataset_s008/beam_output_ref_17/'
    beam_output = np.load (path + "beams_output_train.npz", allow_pickle=True)
    keys = list (beam_output.keys ())
    index_beams_r3_train = calculate_index_beams(beam_output [keys [0]])

    path = '../data/beams_output/dataset_s008/beam_output_ref_17/'
    beam_output = np.load (path + "beams_output_validation.npz", allow_pickle=True)
    keys = list (beam_output.keys ())
    index_beams_r3_val = calculate_index_beams (beam_output [keys [0]])

    index_beams_r3 = np.concatenate((index_beams_r3_train, index_beams_r3_val), axis=0).tolist()


    return index_beams_train, index_beams_test, index_beams_train_str, index_beams_test_str,

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


#divide_beams_in_train_validation()
#compare_index_generate()
#generated_beams_output_from_ray_tracing()