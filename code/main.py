import pre_process_coord
import pre_process_lidar
import beam_selection_wisard
import autoencoder
from beam_selection_wisard import beam_selection_top_k_wisard
from beam_selection_wisard import select_best_beam
import analyse_s009, analyse_s008
import generate_beams
import oversampling
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import random
import pca

random.seed(0)

def beam_selection_wisard_with_article_data():
    #Este metodo faz selecao de beam usando wisard com dados já preprocessadas pro artigo


    path = '../data/beams_output/dataset_s008/beam_output_s008_article/'

    beam_output_train = np.load (path + "index_beams_combined_train.npz", allow_pickle=True)
    keys_train = list(beam_output_train.keys())
    index_beam_article_train = beam_output_train[keys_train[0]]
    index_beam_article_train_str = [str(i) for i in index_beam_article_train]

    beam_output_test = np.load(path + "index_beams_combined_test.npz", allow_pickle=True)
    keys_test = list(beam_output_test.keys())
    index_beam_article_test = beam_output_test[keys_test[0]]
    index_beam_article_test_str = [str(i) for i in index_beam_article_test]

    path_coord = '../data/coord/'
    coord_train = np.load (path_coord + "coord_in_Thermomether_x_y_unbalanced_8_train.npz", allow_pickle=True)['coord_train']
    coord_test = np.load (path_coord + "coord_in_Thermomether_x_y_unbalanced_8_test.npz", allow_pickle=True)['coord_test']

    beam_selection_wisard.select_best_beam(input_train=coord_train,
                                           input_validation=coord_test,
                                           label_train=index_beam_article_train_str,
                                           label_validation=index_beam_article_test_str,
                                           figure_name='accuracy_s008_article_1',
                                           antenna_config='8x32',
                                           type_of_input='coord',
                                           titulo_figura='Accuracy [Train: s008 - Test: s008] - Article Data resolution 8  ',
                                           user='all')
def beam_selection_wisard_with_s008_from_ray_tracing():
    #Este metodo faz selecao de beam usando wisard com dados gerados por ray tracing.
    #e as coordenadas sao tomadas das, já preprocessadas pro artigo
    print("-------------------------------------------")
    print("\t\t\tDataset Raymobtime s008")
    print("-------------------------------------------")

    data_of_article = False
    data_of_RT = False
    data_of_baseline = True
    data_of_ailton_script = False

    if data_of_article:
        index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_used_in_article ()
        flag_beam = "_article"
    if data_of_RT:
        index_beams_train, index_beam_validation, index_beam_RT_all,_,_ = analyse_s008.read_beams_output_generated_by_rt()
        flag_beam = "_RT"
    if data_of_baseline:
        index_beams_train, index_beam_validation,_,_ = analyse_s008.read_beams_output_from_baseline ()
        flag_beam = "_baseline"
    if data_of_ailton_script:
        index_beams_train, index_beam_validation = analyse_s008.read_beams_output_ailton ()
        flag_beam = "_ailton"



    print("\t\t\tTrain \t\tValidation")
    print("\t\t\t", len(index_beams_train), "\t\t", len(index_beam_validation))
    print ("-------------------------------------------")
    #escala = [1, 2, 4, 8, 16, 32, 64]
    escala = [8]

    #analyse_s008.generated_beams_output_from_ray_tracing ()
    #analyse_s008.analyse_index_beams_s008()



    for i in range(len(escala)):
        encoding_coord_train, encoding_coord_validation, encoding_all_coord = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s008(escala=escala[i])
        #encoding_coord_train, encoding_coord_validation, encoding_all_coord = pre_process_coord.new_themomether_coord_x_y_unbalanced_for_s008 (escala=escala[i])
        #encoding_coord_train, encoding_coord_validation, encoding_all_coord =pre_process_coord.sum_digits_of_each_coord_x_y_for_s008(escala=escala[i])
        #encoding_coord_train, encoding_coord_validation, encoding_all_coord =pre_process_coord.new_sum_digits_of_each_coord(escala=escala[i])
        #encoding_coord_train, encoding_coord_validation, encoding_all_coord = pre_process_coord.Thermomether_from_BS_coord_x_y_unbalanced_for_s008(escala=escala[i])

        print ("-------------------------------------------")
        print ("\t\t\t EXPERIMENTOS ")
        print ("-------------------------------------------")
        # Select the beams using WISARD
        beam_selection_wisard.select_best_beam(input_train=encoding_coord_train,
                                               input_validation=encoding_coord_validation,
                                               label_train=index_beams_train,
                                               label_validation=index_beam_validation,
                                               figure_name='s008_train_s008_test_'+str(escala[i])+'_baseLine',
                                               antenna_config='8x32',
                                               type_of_input='coord',
                                               titulo_figura='Accuracy [Train: s008 - Test: s008]',
                                               user='all')
def beam_selection_wisard_with_s008_and_s009_from_ray_tracing():
    # Este medoto faz selecao de beam usando wisard com dados gerados por ray tracing.
    # e as coordenas foram pre-processadas para s008 e s009 devido a diferenca de tamanhas na entrada dos dataset
    # é usado o dataset s008 completo para train e o s009 para test

    data_of_article = False
    data_of_RT = False
    data_of_baseline = True
    data_of_ailton_script = False

    if data_of_article:
        index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_used_in_article ()
        flag_beam = "_article"
    if data_of_RT:
        index_beams_train, index_beam_validation, index_beam_RT_all, _, _ = analyse_s008.read_beams_output_generated_by_rt ()
        flag_beam = "_RT"
    if data_of_baseline:
        index_beams_train_1, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline ()
        index_beams_train = np.concatenate ((index_beams_train_1, index_beam_validation), axis=0)

        train, val, index_beams_test_batool = analyse_s008.read_beams_output_generated_by_batool ()
        index_beams_train_batool = np.concatenate ((train, val), axis=0)

        index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline ()
        index_beam_train_ref_17_str, index_beam_validation_ref_17_str, index_beam_test_ref17_str = analyse_s008.read_beams_output_used_in_ref_17 ()

        print ('Beams para Treinamento')
        for i in range (len (index_beams_train)):
            if index_beams_train [i] != index_beams_train_batool [i] != index_beam_train_ref_17_str [i]:
                print ("Diferente")

        print ('Beams para Teste')
        for i in range (len (index_beam_test_ref17_str)):
            if index_beam_test_ref17_str [i] != index_beams_test_batool [i] != index_beams_test [i]:
                print ("Diferente")

        flag_beam = "_baseline"
    if data_of_ailton_script:
        index_beams_train, index_beam_validation = analyse_s008.read_beams_output_ailton ()
        flag_beam = "_ailton"

    print ("\t\t\tTrain \t\tTest")
    print ("\t\t\t", len (index_beams_train), "\t\t", len (index_beams_test))
    print ("-------------------------------------------")
    # escala = [1, 2, 4, 8, 16, 32, 64]
    escala = [8]

    # analyse_s008.generated_beams_output_from_ray_tracing ()
    # analyse_s008.analyse_index_beams_s008()

    for i in range (len (escala)):
        encoding_coord_train, encoding_coord_validation, encoding_all_coord = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s008 (
            escala=escala [i])
        encoding_coord_test = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s009 (escala=escala [i])
        # encoding_coord_train, encoding_coord_validation, encoding_all_coord = pre_process_coord.new_themomether_coord_x_y_unbalanced_for_s008 (escala=escala[i])
        # encoding_coord_train, encoding_coord_validation, encoding_all_coord =pre_process_coord.sum_digits_of_each_coord_x_y_for_s008(escala=escala[i])
        # encoding_coord_train, encoding_coord_validation, encoding_all_coord =pre_process_coord.new_sum_digits_of_each_coord(escala=escala[i])
        # encoding_coord_train, encoding_coord_validation, encoding_all_coord = pre_process_coord.Thermomether_from_BS_coord_x_y_unbalanced_for_s008(escala=escala[i])

        print ("-------------------------------------------")
        print ("\t\t\t EXPERIMENTOS ")
        print ("-------------------------------------------")
        # Select the beams using WISARD

        beam_selection_wisard.select_best_beam (input_train=encoding_all_coord,
                                                input_validation=encoding_coord_test,
                                                label_train=index_beams_train,
                                                label_validation=index_beams_test,
                                                figure_name='s008_train_s009_test_' + str (escala [i]) + '_baseLine',
                                                antenna_config='8x32',
                                                type_of_input='coord',
                                                titulo_figura='Accuracy [Train: s008 - Test: s009]',
                                                user='all')

        beam_selection_top_k_wisard (x_train=encoding_all_coord,
                                     x_test=encoding_coord_test,
                                     y_train=index_beams_train,
                                     y_test=index_beams_test,
                                     data_input='coord')


def beam_selection_wisard(input_type):
    only_s008 = False
    only_s009 = True
    s_008_and_s009 = False
    top_k_accuracy = False
    input_type = input_type
    preprocess_resolution = 8

    print("-------------------------------------------")
    print("\t\tCalculate the best beam using WISARD  \n\t\t\t with ", input_type, " data")
    print("-------------------------------------------")

    if only_s009:
        name_of_figure = 's009_train_s009_test_'+ input_type + '_'
        title_of_figure = 'Accuracy [Train: s009 - Test: s009] \n - Input[', input_type, '] -'
        data_set = 's009'
        flag_test = 'Validation'

        all_index_beams_s009 = analyse_s009.read_beam_output_generated_by_raymobtime_baseline()

        if input_type == 'coord':

            encoding_coord_s009 = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s009(escala=preprocess_resolution)

            x_train, x_test, y_train, y_test = train_test_split(encoding_coord_s009,
                                                                all_index_beams_s009,
                                                                shuffle=False,
                                                                test_size=0.2)
            input_train = x_train
            input_test = x_test

            label_train = y_train
            label_test = y_test
            size_of_address = 64

        if input_type == 'lidar':
            _, data_lidar_s009 = pre_process_lidar.read_pre_processed_data_rx_like_cube()
            x_train, x_test, y_train, y_test = train_test_split(data_lidar_s009,
                                                                 all_index_beams_s009,
                                                                 shuffle=False,
                                                                 test_size=0.2)
            input_train = x_train
            input_test = x_test

            label_train = y_train
            label_test = y_test

            size_of_address = 64

        if input_type == 'coord_lidar':
            encoding_coord_s009 = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s009(escala=preprocess_resolution)
            _, data_lidar_s009 = pre_process_lidar.read_pre_processed_data_rx_like_cube()

            x = np.concatenate((encoding_coord_s009, data_lidar_s009), axis=1)

            x_train, x_test, y_train, y_test = train_test_split(x,
                                                                 all_index_beams_s009,
                                                                 shuffle=False,
                                                                 test_size=0.2)
            input_train = x_train
            input_test = x_test

            label_train = y_train
            label_test = y_test

            size_of_address = 64




    if only_s008:
        name_of_figure = 's008_train_s008_test_'+input_type+'_'
        title_of_figure = 'Accuracy [Train: s008 - Test: s008] \n - Input[',input_type,'] -'
        data_set = 's008'
        flag_test = 'Validation'

        # ------- Get Beams
        index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline ()
        label_train = index_beams_train
        label_test = index_beam_validation

        if input_type == 'coord':
            # ------ Get preprocess coordinates
            encoding_coord_train, encoding_coord_val, _ = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s008 (
                escala=preprocess_resolution)

            input_train = encoding_coord_train
            input_test = encoding_coord_val
            size_of_address = 54

        if input_type == 'lidar':
            # ----- Get Lidar data
            data_lidar_train, data_lidar_val = pre_process_lidar.read_lidar_data_of_s008 ()
            input_train = data_lidar_train
            input_test = data_lidar_val

        if input_type =='coord_lidar':
            # ----- Get Lidar data
            data_lidar_train, data_lidar_val = pre_process_lidar.read_lidar_data_of_s008 ()

            # ------ Get preprocess coordinates
            encoding_coord_train, encoding_coord_validation, _ = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s008 (
                escala=preprocess_resolution)

            input_train = np.concatenate ((encoding_coord_train, data_lidar_train), axis=1)
            input_test = np.concatenate((encoding_coord_validation, data_lidar_val), axis=1)

            size_of_address = 64


    if s_008_and_s009:
        name_of_figure = 's008_train_s009_test_'+ input_type + '_'
        title_of_figure = 'Accuracy [Train: s008 - Test: s009] \n - Input[', input_type, '] -'
        data_set = 's008-s009'
        flag_test = 'Test'

        # ------- Get Beams
        index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline ()
        index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline ()

        label_train = np.concatenate((index_beams_train, index_beam_validation), axis=0)
        label_test = index_beams_test

        if input_type == 'coord':
            _, _, encoding_coord_train = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s008(escala=preprocess_resolution)
            encoding_coord_test = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s009(escala=preprocess_resolution)

            input_train = encoding_coord_train
            input_test = encoding_coord_test

            size_of_address = 42

        if input_type == 'lidar':
            data_lidar_train, data_lidar_test = pre_process_lidar.read_pre_processed_data_rx_like_cube()

            input_train = data_lidar_train
            input_test = data_lidar_test

        if input_type == 'coord_lidar':

            #--- Get Lidar
            data_lidar_train, data_lidar_test = pre_process_lidar.read_pre_processed_data_rx_like_cube ()

            #--- Get Coord
            _, _, encoding_coord_train = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s008(escala=preprocess_resolution)
            encoding_coord_test = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s009(escala=preprocess_resolution)

            #input_train = np.concatenate ((encoding_coord_train, data_lidar_train), axis=1)
            #input_test = np.concatenate ((encoding_coord_test, data_lidar_test), axis=1)

            input_train = np.concatenate ((data_lidar_train, encoding_coord_train), axis=1)
            input_test = np.concatenate ((data_lidar_test, encoding_coord_test), axis=1)

            size_of_address = 64

    print('\t\tDataset: \t ', data_set)
    print("-------------------------------------------")
    if input_type == 'coord':
        print("\t\t pre-process coordinates")
        print("\tStep \t\tTrain \t\t", flag_test)
        print("\t", preprocess_resolution, "\t", input_train.shape, "\t", input_test.shape)
    if input_type == 'coord_lidar':
        print ("\t\t pre-process coordinates and Lidar")
        print ("\tStep \t\tTrain \t\t", flag_test)
        print ("\t", preprocess_resolution, "\t", input_train.shape, "\t", input_test.shape)
    if input_type == 'lidar':
        print("\t\t pre-process LiDAR")
        print ("\t\tTrain \t\t", flag_test)
        print ("\t", input_train.shape, "\t", input_test.shape)

    print("-------------------------------------------")
    print("\t\t Labels or Beams Index")
    print ("\t\tTrain \t\t", flag_test)
    print("\t\t", len(label_train), "\t\t", len(label_test))
    print ("-------------------------------------------")

    if top_k_accuracy:
        beam_selection_top_k_wisard(x_train=input_train,
                                             x_test=input_test,
                                             y_train=label_train,
                                             y_test=label_test,
                                             data_input=input_type,
                                             data_set=data_set,
                                             address_of_size=size_of_address)
    else:

        name = name_of_figure +str(preprocess_resolution)+'_baseLine'
        select_best_beam(input_train=input_train,
                                               input_validation=input_test,
                                               label_train=label_train,
                                               label_validation=label_test,
                                               figure_name=name,
                                               antenna_config='8x32',
                                               type_of_input=input_type,
                                               titulo_figura=title_of_figure,
                                               user='all')

def beam_selection_wisard_lidar_2D_and_coord():
    lidar_2D_with_rx_2D_thermometer = True
    lidar_rx_2D_like_thermometer = False
    top_k_accuracy = False
    preprocess_resolution = 8

    # --- Get Coord
    _, _, encoding_coord_train = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s008 (
        escala=preprocess_resolution)
    encoding_coord_test = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s009 (escala=preprocess_resolution)


    if lidar_2D_with_rx_2D_thermometer:
        data_lidar_2D_with_rx_train, data_lidar_2D_with_rx_test = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer ()
        input_type = 'lidar_2D_with_rx_2D_therm_and_coord'
        size_of_address = 60

        input_train = np.concatenate((data_lidar_2D_with_rx_train, encoding_coord_train), axis=1)
        input_test = np.concatenate((data_lidar_2D_with_rx_test, encoding_coord_test), axis=1)



    if lidar_rx_2D_like_thermometer:
        position_of_rx_2D_as_thermomether_train, position_of_rx_2D_as_thermomether_test = pre_process_lidar.process_data_rx_2D_like_thermomether ()
        input_type = 'lidar_rx_2D_therm_and_coord'
        size_of_address = 32

        input_train = np.concatenate ((position_of_rx_2D_as_thermomether_train, encoding_coord_train), axis=1)
        input_test = np.concatenate ((position_of_rx_2D_as_thermomether_test, encoding_coord_test), axis=1)

    # ------- Get Beams
    index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline ()
    index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline ()

    label_train = np.concatenate((index_beams_train, index_beam_validation), axis=0)
    label_test = index_beams_test


    name_of_figure = 's008_train_s009_test_' + input_type + '_'
    title_of_figure = "Accuracy [Train: s008 - Test: s009] /n - Input[", input_type, "] -"
    data_set = 's008-s009'


    if top_k_accuracy:
        beam_selection_top_k_wisard (x_train=input_train,
                                     x_test=input_test,
                                     y_train=label_train,
                                     y_test=label_test,
                                     data_input=input_type,
                                     data_set=data_set,
                                     address_of_size=size_of_address)
    else:

        select_best_beam (input_train=input_train,
                          input_validation=input_test,
                          label_train=label_train,
                          label_validation=label_test,
                          figure_name=name_of_figure,
                          antenna_config='8x32',
                          type_of_input=input_type,
                          titulo_figura=title_of_figure,
                          user='all')
def beam_selection_wisard_with_lidar_3D():
    top_k_accuracy = False
    input_type = 'lidar_3D_without_variance' #'lidar_3D_rx_as_therm_cube' #'lidar_3D' #'lidar_3D_without_variance'

    # ------- Get Beams
    index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline ()
    index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline ()

    label_train = np.concatenate((index_beams_train, index_beam_validation), axis=0)
    label_test = index_beams_test
    print('Labels Train = ',len(label_train))
    print('Labels Test =',len(label_test))

    if input_type == 'lidar_3D_without_variance':
        data_lidar_3D_rx_as_therm_cube_train, data_lidar_3D_rx_as_therm_cube_test = pre_process_lidar.data_lidar_3D_with_rx_in_thermometer_as_cube ()
        th_variance = 0
        data_lidar_3D_train, data_lidar_3D_test = pre_process_lidar.data_lidar_3D_binary_without_variance(data_lidar_3D_rx_as_therm_cube_train, data_lidar_3D_rx_as_therm_cube_test, th_variance )
        print ('Input Train = ', data_lidar_3D_train.shape)
        print ('Input Test = ', data_lidar_3D_test.shape)

        input_train = data_lidar_3D_train
        input_test = data_lidar_3D_test

    if input_type == 'lidar_3D':
        data_lidar_3D_as_vector_train, data_lidar_3D_as_vector_test = pre_process_lidar.data_lidar_3D()

        input_train = data_lidar_3D_as_vector_train
        input_test = data_lidar_3D_as_vector_test

    if input_type == 'lidar_3D_rx_as_therm_cube':
        data_lidar_3D_rx_as_therm_cube_train, data_lidar_3D_rx_as_therm_cube_test = pre_process_lidar.data_lidar_3D_with_rx_in_thermometer_as_cube()

        input_train = data_lidar_3D_rx_as_therm_cube_train
        input_test = data_lidar_3D_rx_as_therm_cube_test

    print('Input Train = ', input_train.shape)
    print('Input Test = ', input_test.shape)

    name_of_figure = 's008_train_s009_test_' + input_type + '_'
    title_of_figure = 'Accuracy [Train: s008 - Test: s009] \n - Input ['+ input_type +  '] -'
    print(title_of_figure)
    data_set = 's008-s009'
    flag_test = 'Test'



    if top_k_accuracy:
        size_of_address = 64
        beam_selection_top_k_wisard (x_train=input_train,
                                     x_test=input_test,
                                     y_train=label_train,
                                     y_test=label_test,
                                     data_input=input_type,
                                     data_set=data_set,
                                     address_of_size=size_of_address)
    else:

        select_best_beam (input_train=input_train,
                          input_validation=input_test,
                          label_train=label_train,
                          label_validation=label_test,
                          figure_name=name_of_figure,
                          antenna_config='8x32',
                          type_of_input=input_type,
                          titulo_figura=title_of_figure,
                          user='all')
def beam_selection_wisard_lidar_2D():
    top_k_accuracy = False

    lidar_2D = False
    lidar_2D_without_variance = True
    lidar_rx_2D_like_thermometer = False
    lidar_2D_with_rx_2D_thermometer = False
    lidar_2D_dilated = False

    print ('-------------------------------------------------------------------')
    print ("   SELECIONANDO BEAM USANDO DADOS LIDAR EM 2D")

    if lidar_2D:
        data_lidar_2D_train, data_lidar_2D_test = pre_process_lidar.read_pre_processed_data_lidar_2D()
        input_type = 'lidar_2D'

        input_train = data_lidar_2D_train
        input_test = data_lidar_2D_test

    if lidar_2D_without_variance:
        print ('    E RX EM FORMA DE TERMOMETRO ')
        data_lidar_2D_with_rx_termomether_train, data_lidar_2D_with_rx_termomether_test = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer ()
        th_variance = 0.24
        data_lidar_2D_without_variance_train, data_lidar_2D_without_variance_test = pre_process_lidar.data_lidar_2D_binary_without_variance(data_lidar_2D_with_rx_termomether_train, data_lidar_2D_with_rx_termomether_test, th_variance)
        #input_train = data_lidar_2D_without_variance_train
        #input_test = data_lidar_2D_without_variance_test

        input_type = 'lidar_2D_without_variance'

        input_train = data_lidar_2D_without_variance_train
        input_test = data_lidar_2D_without_variance_test
        size_of_address = 64


    if lidar_rx_2D_like_thermometer:
        position_of_rx_2D_as_thermomether_train, position_of_rx_2D_as_thermomether_test = pre_process_lidar.process_data_rx_2D_like_thermomether()
        input_type = 'lidar_rx_2D_thermometer'
        size_of_address = 56

        input_train = position_of_rx_2D_as_thermomether_train
        input_test = position_of_rx_2D_as_thermomether_test

    if lidar_2D_with_rx_2D_thermometer:
        data_lidar_2D_with_rx_train, data_lidar_2D_with_rx_test = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer()
        input_type = 'lidar_2D_with_rx_2D_thermometer'
        size_of_address = 64

        input_train = data_lidar_2D_with_rx_train
        input_test = data_lidar_2D_with_rx_test

    if lidar_2D_dilated:
        ite = 4
        data_lidar_2D_dilated_train, data_lidar_2D_dilated_test = pre_process_lidar.process_lidar_2D_dilation(ite)
        input_type = 'lidar_2D_dilated'

        input_train = data_lidar_2D_dilated_train
        input_test = data_lidar_2D_dilated_test

    print ('-------------------------------------------------------------------')

    print ('-------------------------')
    print ('--- Input for WiSARD ---')
    print ('   X_train = ', input_train.shape)
    print ('   X_test = ', input_test.shape)
    print ('-------------------------')
    # ------- Get Beams
    index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline ()
    index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline ()

    label_train = np.concatenate((index_beams_train, index_beam_validation), axis=0)
    label_test = index_beams_test

    name_of_figure = 's008_train_s009_test_' + input_type + '_'
    #title_of_figure = 'Accuracy [Train: s008 - Test: s009] \n - Input[', input_type, '] -'
    title_of_figure = 'Accuracy [Train: s008 - Test: s009] \n - Input[', input_type, ' with rx in termomether ] -'
    data_set = 's008-s009'
    flag_test = 'Test'

    if top_k_accuracy:
        beam_selection_top_k_wisard(x_train=input_train,
                                    x_test=input_test,
                                    y_train=label_train,
                                    y_test=label_test,
                                    data_input=input_type,
                                    data_set=data_set,
                                    address_of_size=size_of_address)
    else:

        select_best_beam (input_train=input_train,
                          input_validation=input_test,
                          label_train=label_train,
                          label_validation=label_test,
                          figure_name=name_of_figure,
                          antenna_config='8x32',
                          type_of_input=input_type,
                          titulo_figura=title_of_figure,
                          user='all')
def beam_selection_wisard_lidar_3D_rx_therm_2D():
    top_k_accuracy = False
    input_type = 'lidar_3D_rx_therm_2D'
    size_of_address = 64

    data_lidar_3D_rx_therm_2D_train, data_lidar_3D_rx_therm_2D_test = pre_process_lidar.pre_process_lidar_3D_rx_therm_2D()

    # ------- Get Beams
    index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline ()
    index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline ()

    label_train = np.concatenate((index_beams_train, index_beam_validation), axis=0)
    label_test = index_beams_test

    name_of_figure = 's008_train_s009_test_' + input_type + '_'
    title_of_figure = "Accuracy [Train: s008 - Test: s009] /n - Input[", input_type, "] -"
    data_set = 's008-s009'

    if top_k_accuracy:
        beam_selection_top_k_wisard (x_train=data_lidar_3D_rx_therm_2D_train,
                                     x_test=data_lidar_3D_rx_therm_2D_test,
                                     y_train=label_train,
                                     y_test=label_test,
                                     data_input=input_type,
                                     data_set=data_set,
                                     address_of_size=size_of_address)
    else:

        select_best_beam (input_train=data_lidar_3D_rx_therm_2D_train,
                          input_validation=data_lidar_3D_rx_therm_2D_test,
                          label_train=label_train,
                          label_validation=label_test,
                          figure_name=name_of_figure,
                          antenna_config='8x32',
                          type_of_input=input_type,
                          titulo_figura=title_of_figure,
                          user='all')

def beam_selection_wisard_using_encoder_lidar_2D():
    top_k_accuracy = False
    input_type = 'lidar_2D_autoencoder'
    x_train, x_test = autoencoder.autoencoder()
    #x_train, x_test = autoencoder.binarize_features_by_threshold()
    #x_train, x_test = pre_process_lidar.data_lidar_3D_binary_without_variance()




    # ------- Get Beams
    index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline ()
    index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline ()

    label_train = np.concatenate ((index_beams_train, index_beam_validation), axis=0)
    label_test = index_beams_test

    name_of_figure = 's008_train_s009_test_' + input_type + '_'
    title_of_figure = "Accuracy [Train: s008 - Test: s009] /n - Input[", input_type, "] -"
    data_set = 's008-s009'
    size_of_address = 64

    if top_k_accuracy:
        beam_selection_top_k_wisard (x_train=x_train,
                                     x_test=x_test,
                                     y_train=label_train,
                                     y_test=label_test,
                                     data_input=input_type,
                                     data_set=data_set,
                                     address_of_size=size_of_address)
    else:

        select_best_beam (input_train=x_train,
                          input_validation=x_test,
                          label_train=label_train,
                          label_validation=label_test,
                          figure_name=name_of_figure,
                          antenna_config='8x32',
                          type_of_input=input_type,
                          titulo_figura=title_of_figure,
                          user='all')
def beam_selection_wisard_using_pca():
    input_type = 'lidar_3D_pca'
    x_train, x_test = pca.pca(nro_components=70, type_of_data=input_type)
    top_k_accuracy = False

    # ------- Get Beams
    index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline ()
    index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline ()

    label_train = np.concatenate ((index_beams_train, index_beam_validation), axis=0)
    label_test = index_beams_test

    name_of_figure = 's008_train_s009_test_' + input_type + '_'
    title_of_figure = "Accuracy [Train: s008 - Test: s009] /n - Input[", input_type, "] -"
    data_set = 's008-s009'
    size_of_address = 64

    if top_k_accuracy:
        beam_selection_top_k_wisard (x_train=x_train,
                                     x_test=x_test,
                                     y_train=label_train,
                                     y_test=label_test,
                                     data_input=input_type,
                                     data_set=data_set,
                                     address_of_size=size_of_address)
    else:

        select_best_beam (input_train=x_train,
                          input_validation=x_test,
                          label_train=label_train,
                          label_validation=label_test,
                          figure_name=name_of_figure,
                          antenna_config='8x32',
                          type_of_input=input_type,
                          titulo_figura=title_of_figure,
                          user='all')
def beam_selection_wisard_using_pca_with_lidar_2D():
    print ('-------------------------------------------------------------------')
    print("   SELECIONANDO BEAM USANDO PCA COM DADOS ")

    input_type = 'lidar_2D_pca'
    top_k_accuracy = False

    # ------- Get Lidar
    # there are three types of input:
    #  1. -> lidar 2D with rx like thermometer
    #  2. -> lidar 2D with rx like thermometer and elimination of variance
    #  3. -> lidar 2D with Int data and rx like thermometer
    #  4. -> lidar 2D with Int data and rx like thermometer and elimination of variance

    type_of_lidar_2D =4

    if type_of_lidar_2D == 1:
        print ("LIDAR 2D BINARIO COM RX EM FORMA DE TERMOMETRO")
        data_lidar_2D_with_rx_termomether_train, data_lidar_2D_with_rx_termomether_test = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer()
        lidar_train = data_lidar_2D_with_rx_termomether_train
        lidar_test = data_lidar_2D_with_rx_termomether_test
    if type_of_lidar_2D == 2:
        print ("LIDAR 2D BINARIO COM RX EM FORMA DE TERMOMETRO")
        print ("        E ELIMINANDO VARIANCIA")
        th_variance = 0.1
        data_lidar_2D_with_rx_termomether_train, data_lidar_2D_with_rx_termomether_test = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer()
        data_lidar_2D_without_variance_train, data_lidar_2D_without_variance_test = pre_process_lidar.data_lidar_2D_binary_without_variance (
            data_lidar_2D_with_rx_termomether_train, data_lidar_2D_with_rx_termomether_test, th_variance)
        lidar_train = data_lidar_2D_without_variance_train
        lidar_test = data_lidar_2D_without_variance_test
    if type_of_lidar_2D == 3:
        print("LIDAR 2D INTEIROS COM RX EM FORMA DE TERMOMETRO")
        data_lidar_2D_in_Int_with_rx_therm_train, data_lidar_2D_in_Int_with_rx_therm_test, _,_ = pre_process_lidar.process_data_lidar_into_2D_matrix()
        lidar_train = data_lidar_2D_in_Int_with_rx_therm_train
        lidar_test = data_lidar_2D_in_Int_with_rx_therm_test
    if type_of_lidar_2D == 4:
        print ("LIDAR 2D INTEIROS COM RX EM FORMA DE TERMOMETRO")
        print ("       E ELIMINANDO VARIANCIA")
        th_variance = 0.1
        data_lidar_2D_in_Int_with_rx_therm_train, data_lidar_2D_in_Int_with_rx_therm_test,_,_ = pre_process_lidar.process_data_lidar_into_2D_matrix ()
        data_lidar_2D_Int_without_var_train, data_lidar_2D_Int_without_var_test = pre_process_lidar.data_lidar_2D_binary_without_variance(data_lidar_2D_in_Int_with_rx_therm_train,
                                                                                    data_lidar_2D_in_Int_with_rx_therm_test,
                                                                                    th_variance)
        lidar_train = data_lidar_2D_Int_without_var_train
        lidar_test = data_lidar_2D_Int_without_var_test

    print ('-------------------------------------------------------------------')
    type_of_binarization = 'thermometer_1'  # options: 'simple_threshold', 'thermometer_1', 'thermometer_2'

    print('Input for PCA')
    print(' Train = ', lidar_train.shape)
    print(' Test = ', lidar_test.shape)

    # ------- Get PCA
    components = [10, 20, 30, 40, 50, 60, 70]
    for i in range(len(components)):
        x_train, x_test = pca.pca(data_train=lidar_train,
                                  data_test=lidar_test,
                                  nro_components=components[i],
                                  binarization_type = type_of_binarization #options: 'simple_threshold', 'thermometer_1', 'thermometer_2'
                                  )
        print('-------------------------')
        print('--- Input for WiSARD ---')
        print('    Using a ', type_of_binarization)
        print('   X_train = ', x_train.shape)
        print('   X_test = ', x_test.shape)
        print ('-------------------------')

        # ------- Get Beams
        index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline()
        index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline()

        label_train = np.concatenate((index_beams_train, index_beam_validation), axis=0)
        label_test = index_beams_test
        # -------------------------

        name_of_figure = 's008_train_s009_test_' + input_type + '_'+type_of_binarization+'_'+str(components[i])
        title_of_figure = 'Accuracy [Train: s008 - Test: s009] \n - Input['+ input_type+ ' with '+ str(components[i])+ ' componentes e binarizacao '+type_of_binarization+' ] -'

        print(title_of_figure)
        data_set = 's008-s009'
        size_of_address = 64

        if top_k_accuracy:
            beam_selection_top_k_wisard (x_train=x_train,
                                         x_test=x_test,
                                         y_train=label_train,
                                         y_test=label_test,
                                         data_input=input_type,
                                         data_set=data_set,
                                         address_of_size=size_of_address)
        else:

            select_best_beam (input_train=x_train,
                              input_validation=x_test,
                              label_train=label_train,
                              label_validation=label_test,
                              figure_name=name_of_figure,
                              antenna_config='8x32',
                              type_of_input=input_type,
                              titulo_figura=title_of_figure,
                              user='all')
def beam_selection_wisard_using_pca_with_lidar_3D_into_2D():

    variance_eliminination = False
    type_of_binarization = 'thermometer_1'
    components = 50
    data_lidar_2D_vector_train, data_lidar_2D_vector_test = pre_process_lidar.process_data_lidar_into_2D_matrix()

    print ('-------------------------------------------------------------------')
    print ("   SELECIONANDO BEAM USANDO PCA COM DADOS DE LIDAR 2D INTEIROS")
    print ('         E RX EM FORMA DE TERMOMETRO ')
    print ('-------------------------------------------------------------------')

    if variance_eliminination:
        th_variance = 0.1
        input_type = 'lidar_2D_int_pca'
        top_k_accuracy = False

        data_lidar_2D_without_variance_train, data_lidar_2D_without_variance_test = pre_process_lidar.data_lidar_2D_binary_without_variance (
            data_lidar_2D_vector_train, data_lidar_2D_vector_test, th_variance)
        x_train, x_test = pca.pca(data_train=data_lidar_2D_without_variance_train,
                                  data_test=data_lidar_2D_without_variance_test,
                                  nro_components=components,
                                  binarization_type=type_of_binarization)
    else:
        x_train, x_test = pca.pca(data_train=data_lidar_2D_vector_train,
                                  data_test=data_lidar_2D_vector_test,
                                  nro_components=components,
                                  binarization_type=type_of_binarization)

    print ('-------------------------')
    print ('--- Input for WiSARD ---')
    print ('    Using a ', type_of_binarization)
    print ('   X_train = ', x_train.shape)
    print ('   X_test = ', x_test.shape)
    print ('-------------------------')



    # ------- Get Beams
    index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline ()
    index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline ()

    label_train = np.concatenate ((index_beams_train, index_beam_validation), axis=0)
    label_test = index_beams_test

    name_of_figure = 's008_train_s009_test_' + input_type + '_' + type_of_binarization + '_' + str (components)
    title_of_figure = 'Accuracy [Train: s008 - Test: s009] \n - Input[' + input_type + ' with ' + str (
        components) + ' componentes e binarizacao ' + type_of_binarization + ' ] -'
    # title_of_figure = 'Accuracy [Train: s008 - Test: s009] \n - Input[', input_type, ' with rx in termomether ] -'

    print (title_of_figure)
    data_set = 's008-s009'
    size_of_address = 64

    if top_k_accuracy:
        beam_selection_top_k_wisard (x_train=x_train,
                                     x_test=x_test,
                                     y_train=label_train,
                                     y_test=label_test,
                                     data_input=input_type,
                                     data_set=data_set,
                                     address_of_size=size_of_address)
    else:

        select_best_beam (input_train=x_train,
                          input_validation=x_test,
                          label_train=label_train,
                          label_validation=label_test,
                          figure_name=name_of_figure,
                          antenna_config='8x32',
                          type_of_input=input_type,
                          titulo_figura=title_of_figure,
                          user='all')
def beam_selection_wisard_using_oversampled():
    print ('-------------------------------------------------------------------')
    print ("   SELECIONANDO BEAM USANDO OVERSAMPLE")
    print ('-------------------------------------------------------------------')
    input_type = 'lidar_2D'
    x_train, y_train, x_test, y_test = oversampling.oversampling(input_type)
    print ('-------------------------')
    print ('--- Input for WiSARD ---')
    print ('   X_train = ', x_train.shape)
    print ('   X_test = ', x_test.shape)
    print ('-------------------------')
    top_k_accuracy = False



    name_of_figure = 's008_train_s009_test_' + input_type + '_'
    title_of_figure = "Accuracy [Train: s008 - Test: s009] /n - Input[", input_type, "] -"
    data_set = 's008-s009'
    size_of_address = 64

    if top_k_accuracy:
        beam_selection_top_k_wisard (x_train=x_train,
                                     x_test=x_test,
                                     y_train=y_train,
                                     y_test=y_test,
                                     data_input=input_type,
                                     data_set=data_set,
                                     address_of_size=size_of_address)
    else:

        select_best_beam (input_train=x_train,
                          input_validation=x_test,
                          label_train=y_train,
                          label_validation=y_test,
                          figure_name=name_of_figure,
                          antenna_config='8x32',
                          type_of_input=input_type,
                          titulo_figura=title_of_figure,
                          user='all')

def beam_selection_wisard_with_lidar_and_coord(input_type_comb, top_k):
    print ('-------------------------------------------------------------------')
    print ("SELECIONANDO BEAM COM :")
    #print ("   SELECIONANDO BEAM COM DADOS DE LIDAR  ")
    #print ('         E RX EM FORMA DE TERMOMETRO     ')
    #print ('-------------------------------------------------------------------')
    top_k_accuracy = top_k
    #top_k_accuracy = True

    # Options for combinations of input:
    # 1. only Coordinates
    # 2. only Coordinates using variance elimination
    # 3. Lidar 2D with Rx as Thermometer
    # 4. Lidar 2D with Rx as Thermometer using variance elimination
    # 5. lidar 2D with Rx as Thermometer and coordinates
    # 6. lidar 2D with Rx as Thermometer using variance elimination and coordinates
    # 7. lidar_3D with Rx as cube and coordinates
    # 8. lidar_3D with Rx as cube and coordinates using variance elimination
    # 9. Lidar 3D with Rx as Thermometer
    # 10. Lidar 3D with Rx as Thermometer using variance elimination
    # 11. Lidar 3D with Rx as Thermometer and coordinates
    # 12. Lidar 3D with Rx as Thermometer and coordinates using variance elimination
    input_type = 'lidar_+_coord'
    combinations_of_input = input_type_comb
    #combinations_of_input = 6

    # --- Get Coord
    preprocess_resolution = 16
    _, _, encoding_coord_train = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s008 (
        escala=preprocess_resolution)
    encoding_coord_test = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s009 (
        escala=preprocess_resolution)

    if combinations_of_input == 1: # only Coordinates
        print(" COM COORDENADAS")
        label_input_type = 'coord'
        input_train = encoding_coord_train
        input_test = encoding_coord_test
        size_of_address = 44

    elif combinations_of_input == 2: # only Coordinates using variance elimination
        print(" COM COORDENADAS E ELIMINACAO DE VARIANCIA")
        print(encoding_coord_train.shape)
        print(encoding_coord_test.shape)
        th = 0.1
        coord_sv_train, coord_sv_test = pre_process_lidar.data_lidar_2D_binary_without_variance(encoding_coord_train, encoding_coord_test, th)
        label_input_type = 'coord_sv'
        input_train = coord_sv_train
        input_test = coord_sv_test


    elif combinations_of_input == 3: #lidar 2D with Rx as Thermometer
        print (" COM DADOS DE LIDAR 2D + RX EM FORMA DE TERMOMETRO")
        label_input_type = 'LiDAR_2D_+_Rx_Term'
        data_lidar_2D_with_rx_train, data_lidar_2D_with_rx_test = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer ()

        input_train = data_lidar_2D_with_rx_train
        input_test = data_lidar_2D_with_rx_test
        size_of_address = 48

    elif combinations_of_input == 4: #Lidar 2D with Rx as Thermometer using variance elimination
        print(" DADOS DE LIDAR 2D + RX EM FORMA DE TERMOMETRO")

        th = 0.15
        label_input_type = 'LiDAR_2D_+_Rx_Term_SVar'
        data_lidar_2D_with_rx_train, data_lidar_2D_with_rx_test = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer()
        print ("Datos de Train: ", data_lidar_2D_with_rx_train.shape)
        print ("Datos de Test: ", data_lidar_2D_with_rx_test.shape)

        print (" APLICANDO A ELIMINACAO DE VARIANCIA ")
        data_lidar_2D_with_rx_SV_train, data_lidar_2D_with_rx_SV_test = pre_process_lidar.data_lidar_2D_binary_without_variance (data_lidar_2D_with_rx_train, data_lidar_2D_with_rx_test, th)

        input_train = data_lidar_2D_with_rx_SV_train
        input_test = data_lidar_2D_with_rx_SV_test
        size_of_address = 24

    elif combinations_of_input == 5: #lidar 2D with Rx as Thermometer and coordinates
        print(" COM DADOS DE LIDAR 2D + RX EM FORMA DE TERMOMETRO")
        print(" E COORDENADAS")
        label_input_type = 'LiDAR_2D_+_Rx_Term_+_Coord'

        data_lidar_2D_with_rx_train, data_lidar_2D_with_rx_test = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer ()

        input_train = np.concatenate ((data_lidar_2D_with_rx_train, encoding_coord_train), axis=1)
        input_test = np.concatenate ((data_lidar_2D_with_rx_test, encoding_coord_test), axis=1)
        size_of_address = 64

    elif combinations_of_input == 6: #lidar 2D with Rx as Thermometer using variance elimination and coordinates
        print (" COM DADOS DE LIDAR 2D + RX EM FORMA DE TERMOMETRO E ELIMINACAO DE VARIANCIA")
        print (" E COORDENADAS")
        label_input_type = 'LiDAR_2D_+_Rx_Term_+_Coord_'+str(preprocess_resolution)+'_SVar'
        data_lidar_2D_with_rx_termomether_train, data_lidar_2D_with_rx_termomether_test = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer ()
        th_variance = 0.1
        data_lidar_2D_without_variance_train, data_lidar_2D_without_variance_test = pre_process_lidar.data_lidar_2D_binary_without_variance (
            data_lidar_2D_with_rx_termomether_train, data_lidar_2D_with_rx_termomether_test, th_variance)

        input_train = np.concatenate ((data_lidar_2D_without_variance_train, encoding_coord_train), axis=1)
        input_test = np.concatenate ((data_lidar_2D_without_variance_test, encoding_coord_test), axis=1)
        size_of_address = 44

    elif combinations_of_input == 7: #lidar_3D with Rx as cube and coordinates
        print (" COM DADOS DE LIDAR 3D + RX EM FORMA DE CUBO")
        print (" E COORDENADAS")
        label_input_type = 'LiDAR_3D_+_Rx_Cubo_+_Coord'
        data_lidar_3D_rx_as_therm_cube_train, data_lidar_3D_rx_as_therm_cube_test = pre_process_lidar.data_lidar_3D_with_rx_in_thermometer_as_cube ()

        input_train = np.concatenate ((data_lidar_3D_rx_as_therm_cube_train, encoding_coord_train), axis=1)
        input_test = np.concatenate ((data_lidar_3D_rx_as_therm_cube_test, encoding_coord_test), axis=1)

    elif combinations_of_input == 8: #lidar_3D with Rx as cube and coordinates using variance elimination
        print (" COM DADOS DE LIDAR 3D + RX EM FORMA DE CUBO E ELIMINACAO DE VARIANCIA")
        print (" E COORDENADAS")
        label_input_type = 'LiDAR_3D_+_Rx_Cubo_SVar_+_Coord'
        data_lidar_3D_rx_as_therm_cube_train, data_lidar_3D_rx_as_therm_cube_test = pre_process_lidar.data_lidar_3D_with_rx_in_thermometer_as_cube ()
        th_variance = 0.15
        data_lidar_3D_train, data_lidar_3D_test = pre_process_lidar.data_lidar_3D_binary_without_variance (
            data_lidar_3D_rx_as_therm_cube_train, data_lidar_3D_rx_as_therm_cube_test, th_variance)

        input_train = np.concatenate ((data_lidar_3D_train, encoding_coord_train), axis=1)
        input_test = np.concatenate ((data_lidar_3D_test, encoding_coord_test), axis=1)

    elif combinations_of_input == 9:  # Lidar 3D with Rx as Thermometer
        print (" COM DADOS DE LIDAR 3D + RX EM FORMA DE TERMOMETRO")
        label_input_type = 'LiDAR_3D_+_Rx_Term'
        data_lidar_3D_rx_as_therm_cube_train, data_lidar_3D_rx_as_therm_cube_test = pre_process_lidar.pre_process_lidar_3D_rx_therm_2D ()

        input_train = data_lidar_3D_rx_as_therm_cube_train
        input_test = data_lidar_3D_rx_as_therm_cube_test

    elif combinations_of_input == 10: # Lidar 3D with Rx as Thermometer using variance elimination
        print (" COM DADOS DE LIDAR 3D + RX EM FORMA DE TERMOMETRO E ELIMINACAO DE VARIANCIA")
        label_input_type = 'LiDAR_3D_+_Rx_Term_SVar'
        data_lidar_3D_rx_as_therm_train, data_lidar_3D_rx_as_therm_test = pre_process_lidar.pre_process_lidar_3D_rx_therm_2D()
        th_variance = 0.15
        data_lidar_3D_train, data_lidar_3D_test = pre_process_lidar.data_lidar_3D_binary_without_variance(data_lidar_3D_rx_as_therm_train, data_lidar_3D_rx_as_therm_test, th_variance)

        input_train = data_lidar_3D_train
        input_test = data_lidar_3D_test

    elif combinations_of_input == 11: #Lidar 3D with Rx as Thermometer and coordinates
        print (" COM DADOS DE LIDAR 3D + RX EM FORMA DE TERMOMETRO")
        print (" E COORDENADAS")
        label_input_type = 'LiDAR_3D_+_Rx_Term_+_Coord'
        data_lidar_3D_rx_as_therm_cube_train, data_lidar_3D_rx_as_therm_cube_test = pre_process_lidar.pre_process_lidar_3D_rx_therm_2D()
        input_train = np.concatenate ((data_lidar_3D_rx_as_therm_cube_train, encoding_coord_train), axis=1)
        input_test = np.concatenate ((data_lidar_3D_rx_as_therm_cube_test, encoding_coord_test), axis=1)

    elif combinations_of_input == 12: #Lidar 3D with Rx as Thermometer and coordinates using variance elimination
        print (" COM DADOS DE LIDAR 3D + RX EM FORMA DE TERMOMETRO E ELIMINACAO DE VARIANCIA")
        print (" E COORDENADAS")
        label_input_type = 'LiDAR_3D_+_Rx_Term_SVar_+_Coord'
        data_lidar_3D_rx_as_therm_train, data_lidar_3D_rx_as_therm_test = pre_process_lidar.pre_process_lidar_3D_rx_therm_2D ()
        th_variance = 0.15
        data_lidar_3D_train, data_lidar_3D_test = pre_process_lidar.data_lidar_3D_binary_without_variance (
            data_lidar_3D_rx_as_therm_train, data_lidar_3D_rx_as_therm_test, th_variance)

        all_data_train = np.concatenate ((data_lidar_3D_train, encoding_coord_train), axis=1)
        all_data_test = np.concatenate ((data_lidar_3D_test, encoding_coord_test), axis=1)

        input_train = all_data_train
        input_test = all_data_test


    print ('Input Train = ', input_train.shape)
    print ('Input Test = ', input_test.shape)


    # ------- Get Beams of baseline
    #index_beams_train, index_beam_validation, _, _ = analyse_s008.read_beams_output_from_baseline ()
    #index_beams_test = analyse_s009.read_beam_output_generated_by_raymobtime_baseline ()

    #label_train = np.concatenate ((index_beams_train, index_beam_validation), axis=0)
    #label_test = index_beams_test

    #------ Get Beams of Raymobtime generated by me
    index_beams_train, index_beams_test, _, _ = generate_beams.read_beams_output_generated_by_ray_tracing()

    label_train = index_beams_train
    label_test = index_beams_test

    print ('Labels Train = ', len (label_train))
    print ('Labels Test =', len (label_test))
    print ("------------------------------------------------------------")

    name_of_figure = label_input_type
    title_of_figure = 'Accuracy [Train: s008 - Test: s009] \n - Input[' + label_input_type + '] -'

    #print (title_of_figure)
    data_set = 's008-s009'


    if top_k_accuracy:
        beam_selection_top_k_wisard (x_train=input_train,
                                     x_test=input_test,
                                     y_train=label_train,
                                     y_test=label_test,
                                     data_input=input_type,
                                     data_set=data_set,
                                     address_of_size=size_of_address,
                                     name_of_conf_input = label_input_type)
    else:

        select_best_beam (input_train=input_train,
                          input_validation=input_test,
                          label_train=label_train,
                          label_validation=label_test,
                          figure_name=name_of_figure,
                          antenna_config='8x32',
                          type_of_input=input_type,
                          titulo_figura=title_of_figure,
                          user='all')

def main():
    print ("--------------------------------------------------------------------------------------")
    print("*\t\tAntes de realizar a selecao de Beam com a rede WiSARD verifique: \t\t\t*"    
          "\n* \t\t\t se os dados de treinamento e teste estao disponiveis \t\t\t\t\t*"
          "\n* \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t*"
          "\n* \t COORDENADAS \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t*"
          "\n* \t\t\t data/coord/CoordVehiclesRxPerScene_s008.csv \t\t\t\t\t\t\t*"
          "\n* \t\t\t data/coord/CoordVehiclesRxPerScene_s009.csv \t\t\t\t\t\t\t*"
          "\n* \t LiDAR 2D \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t*"
          "\n* \t\t\t data/lidar/s008/lidar_train_raymobtime.npz \t\t\t\t\t\t\t*"
          "\n* \t\t\t data/lidar/s008/lidar_validation_raymobtime.npz\t\t\t\t\t\t\t*"
          "\n* \t\t\t data/lidar/s009/lidar_test_raymobtime.npz \t\t\t\t*"
          "\n* \t BEAMS \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t*"
          "\n* \t\t data/beams_output/beam_output_baseline_raymobtime_s008/ \t\t\t\t\t*"
          "\n* \t\t\t beams_output_train.npz \t\t beams_output_test.npz \t\t\t\t\t*"
          "\n* \t\t data/beams_output/beam_output_baseline_raymobtime_s009/ \t\t\t\t\t*"
          "\n* \t\t\t beams_output_test_s009.npz \t\t\t\t\t\t\t\t\t\t\t*")
    print ("--------------------------------------------------------------------------------------")
    print("\t\t\tSELECAO DE BEAM USANDO A REDE WISARD COMO CLASSIFICADOR")
    print("--------------------------------------------------------------------------------------")
    print("\tEscolha o tipo de entrada a ser usada: ")
    print("\t\t\t1. Coordenadas")
    print("\t\t\t2. LiDAR em 2D")
    print("\t\t\t3. LiDAR em 2D + Coordenadas")

    input_type = int(input("\tDigite o tipo de entrada: "))
    print("--------------------------------------------------------------------------------------")
    print("\tRealizar analise top-k da acuracia?")
    top_k_accuracy = int(input("\t Digite:\n\t\t 1 para SIM e \n\t\t 0 para NAO: "))
    print("--------------------------------------------------------------------------------------")

    if input_type == 1:
        if top_k_accuracy == 1:
            print("Selecionando Beam TOP-K usando coordenadas")
            top_k = True
            beam_selection_wisard_with_lidar_and_coord(input_type_comb=1, top_k=top_k)
        else:
            print("Analisando o desempenho da WiSARD com diferentes enderecamentos de memoria")
            print("Selecionando beam usando coordenadas")
            top_k = False
            beam_selection_wisard_with_lidar_and_coord(input_type_comb=1, top_k=top_k)

    if input_type == 2:
        if top_k_accuracy == 1:
            print("Selecionando Beam TOP-K usando LiDAR 2D")
            top_k = True
            beam_selection_wisard_with_lidar_and_coord(input_type_comb=4, top_k=top_k)
        else:
            print("Analisando o desempenho da WiSARD com diferentes enderecamentos de memoria")
            print("Selecionando beam usando LiDAR 2D")
            top_k = False
            beam_selection_wisard_with_lidar_and_coord(input_type_comb=4, top_k=top_k)

    if input_type == 3:
        if top_k_accuracy == 1:
            print("Selecionando Beam TOP-K usando LiDAR 2D + Coordenadas")
            top_k = True
            beam_selection_wisard_with_lidar_and_coord(input_type_comb=6, top_k=top_k)
        else:
            print("Analisando o desempenho da WiSARD com diferentes enderecamentos de memoria")
            print("Selecionando beam usando LiDAR 2D + Coordenadas")
            top_k = False
            beam_selection_wisard_with_lidar_and_coord(input_type_comb=6, top_k=top_k)


#beam_selection_wisard_using_pca_with_lidar_2D()
#beam_selection_wisard_with_lidar_and_coord()
main()
#beam_selection_wisard_with_s008_and_s009_from_ray_tracing()

