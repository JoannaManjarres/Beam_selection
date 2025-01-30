import numpy as np
import csv
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
import beam_selection_wisard as bsw
import matplotlib.pyplot as plt
import pre_process_lidar
from matplotlib.gridspec import GridSpec

def read_beams_raymobtime(num_antennas_rx, path_of_data):

    data_path = path_of_data
    beams = np.load(data_path)['output_classification']

    best_beam_index = []
    for sample in range(beams.shape[0]):
        best_beam_index.append(np.argmax(beams[sample, :]))

    beam_index_rx = np.array(best_beam_index)

    tx_index = np.zeros((beams.shape[0]), dtype=int)
    rx_index = np.zeros((beams.shape[0]), dtype=int)

    for sample in range(len(beam_index_rx)):
        index_tx = best_beam_index[sample] // int(num_antennas_rx)
        index_rx = best_beam_index[sample] % int(num_antennas_rx)
        tx_index[sample] = index_tx
        rx_index[sample] = index_rx

    return tx_index, rx_index, best_beam_index
def read_data_s008():
    filename = '../data/coord/CoordVehiclesRxPerScene_s008.csv'
    limit_ep_train = 1564
    data = pd.read_csv (filename)
    valid_data = data [data ['Val'] == 'V']
    #train_data = valid_data [valid_data ['EpisodeID'] <= limit_ep_train]

    path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_train.npz'
    tx_index_train, rx_index_train, best_beam_index_train = read_beams_raymobtime (num_antennas_rx=8, path_of_data=path)

    path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_validation.npz'
    tx_index_val, rx_index_val, best_beam_index_val = read_beams_raymobtime (num_antennas_rx=8, path_of_data=path)

    tx_index = np.concatenate ((tx_index_train, tx_index_val), axis=0)
    rx_index = np.concatenate ((rx_index_train, rx_index_val), axis=0)
    best_beam_index = np.concatenate ((best_beam_index_train, best_beam_index_val), axis=0)

    valid_data.insert (6, 'index_beams', best_beam_index)
    valid_data.insert (7, 'tx_index', tx_index)
    valid_data.insert (8, 'rx_index', rx_index)

    train_data_LOS = valid_data [valid_data ['LOS'] == 'LOS=1']
    train_data_NLOS = valid_data [valid_data ['LOS'] == 'LOS=0']

    return train_data_LOS, train_data_NLOS, valid_data
def read_data_s009():
    filename = '../data/coord/CoordVehiclesRxPerScene_s009.csv'
    limit_ep_train = 1564
    data = pd.read_csv (filename)
    valid_data = data [data ['Val'] == 'V']
    #train_data = valid_data [valid_data ['EpisodeID'] <= limit_ep_train]

    path = '../data/beams_output/beam_output_baseline_raymobtime_s009/beams_output_test.npz'
    tx_index_train, rx_index_train, best_beam_index_train = read_beams_raymobtime (num_antennas_rx=8, path_of_data=path)

    valid_data.insert (6, 'index_beams', best_beam_index_train)
    valid_data.insert (7, 'tx_index', tx_index_train)
    valid_data.insert (8, 'rx_index', rx_index_train)

    train_data_LOS = valid_data [valid_data ['LOS'] == 'LOS=1']
    train_data_NLOS = valid_data [valid_data ['LOS'] == 'LOS=0']

    return train_data_LOS, train_data_NLOS, valid_data

def  Thermomether_coord_x_y_unbalanced(escala, all_info_coord_val, data):
    #all_info_coord_val, coord_train, coord_test = read_valid_coordinates_s008()

    episodios = data['EpisodeID']

    all_x_coord_str = all_info_coord_val['x'].iloc[:]
    all_x_coord = [int(x) for x in all_x_coord_str]
    all_y_coord_str = all_info_coord_val['y'].iloc[:]
    all_y_coord = [int(y) for y in all_y_coord_str]

    min_x_coord = np.min(all_x_coord)
    min_y_coord = np.min(all_y_coord)
    max_x_coord = np.max(all_x_coord)
    max_y_coord = np.max(all_y_coord)

    x_coord = [int(x) for x in data['x']]
    y_coord = [int(x) for x in data['y']]

    diff_all_x_coord = np.array((x_coord - min_x_coord))
    diff_all_y_coord = np.array(y_coord) - min_y_coord

    escala = escala
    size_of_data_x = (max_x_coord - min_x_coord) * escala
    size_of_data_y = (max_y_coord - min_y_coord) * escala
    enconding_x = np.array([len(data), size_of_data_x], dtype=int)
    enconding_y = np.array([len(data), size_of_data_y+escala], dtype=int) #+escala], dtype=int)

    encoding_x_vector = np.zeros(enconding_x, dtype=int)
    encoding_y_vector = np.zeros(enconding_y, dtype=int)

    n_x = 1 * escala

    sample = 0
    for i in diff_all_x_coord:
        for j in range(i * n_x):
            encoding_x_vector[sample, j] = 1
        sample = sample + 1

    sample = 0
    for i in diff_all_y_coord:
        for j in range(i * n_x+escala):#+escala):
            encoding_y_vector[sample, j] = 1
        sample = sample + 1

    encondig_coord = np.concatenate((encoding_x_vector, encoding_y_vector), axis=1)
    encoding_coord_and_episode = np.column_stack([episodios, encondig_coord])

    return encondig_coord

def Thermomether_coord_x_y_unbalanced_for_s009(escala, all_info_coord_val, data):
    episodios = all_info_coord_val['EpisodeID']

    all_x_coord = [int(x) for x in all_info_coord_val['x']]
    all_y_coord = [int(y) for y in all_info_coord_val['y']]

    min_x_coord = np.min(all_x_coord)
    min_y_coord = np.min(all_y_coord)

    max_x_coord = np.max(all_x_coord)
    max_y_coord = np.max(all_y_coord)

    x_coord = [int(x) for x in data['x']]
    y_coord = [int(x) for x in data['y']]

    diff_all_x_coord = np.array((x_coord - min_x_coord))
    diff_all_y_coord = np.array((y_coord - min_y_coord))

    escala = escala
    size_of_data_x = (max_x_coord-min_x_coord) * escala
    size_of_data_y = (max_y_coord-min_y_coord) * escala
    enconding_x = np.array([len(data), size_of_data_x], dtype=int)
    enconding_y = np.array([len(data), size_of_data_y], dtype=int)

    encoding_x_vector = np.zeros(enconding_x, dtype=int)
    encoding_y_vector = np.zeros(enconding_y, dtype=int)

    n_x = 1 * escala

    sample = 0
    for i in diff_all_x_coord:
        for j in range(i * n_x):
            encoding_x_vector[sample, j] = 1
        sample = sample + 1

    sample = 0
    for i in diff_all_y_coord:
        for j in range(i * n_x):
            encoding_y_vector[sample, j] = 1
        sample = sample + 1

    encondig_coord = np.concatenate((encoding_x_vector, encoding_y_vector), axis=1)

    return encondig_coord

def get_coord_preprocess(connection_type, preprocess_resolution):
    train_data_LOS_s009, train_data_NLOS_s009, valid_data_s009 = read_data_s009 ()
    train_data_LOS_s008, train_data_NLOS_s008, valid_data_s008 = read_data_s008 ()

    if connection_type == 'LOS':
        data_train = train_data_LOS_s008
        data_test = train_data_LOS_s009
    elif connection_type == 'NLOS':
        data_train = train_data_NLOS_s008
        data_test = train_data_NLOS_s009
    elif connection_type == 'ALL':
        data_train = valid_data_s008
        data_test = valid_data_s009

    # for i in range (len (preprocess_resolution)):
    encoding_coord_test = Thermomether_coord_x_y_unbalanced_for_s009 (escala=preprocess_resolution,
                                                                      all_info_coord_val=valid_data_s009,
                                                                      data=data_test)
    encoding_coord_train = Thermomether_coord_x_y_unbalanced (escala=preprocess_resolution,
                                                              all_info_coord_val=valid_data_s008,
                                                              data=data_train)

    input_train = encoding_coord_train
    input_test = encoding_coord_test
    label_train = [str (y) for y in data_train ['index_beams']]
    label_test = [str (y) for y in data_test ['index_beams']]

    return input_train, input_test, label_train, label_test

def get_lidar_preprocess(connection_type):
    train_data_LOS_s009, train_data_NLOS_s009, valid_data_s009 = read_data_s009 ()
    train_data_LOS_s008, train_data_NLOS_s008, valid_data_s008 = read_data_s008 ()

    data_lidar_2D_with_rx_termomether_train, data_lidar_2D_with_rx_termomether_test = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer ()
    th_variance = 0.1
    data_lidar_2D_without_variance_train, data_lidar_2D_without_variance_test = pre_process_lidar.data_lidar_2D_binary_without_variance (
        data_lidar_2D_with_rx_termomether_train, data_lidar_2D_with_rx_termomether_test, th_variance)

    valid_data_s008.insert (13, 'lidar', data_lidar_2D_without_variance_train.tolist ())
    valid_data_s009.insert (13, 'lidar', data_lidar_2D_without_variance_test.tolist ())

    if connection_type == 'LOS':
        data_train = valid_data_s008 [valid_data_s008 ['LOS'] == 'LOS=1']
        data_test = valid_data_s009 [valid_data_s009 ['LOS'] == 'LOS=1']
        input_train = data_train ['lidar'].tolist ()
        input_test = data_test ['lidar'].tolist ()
        label_train = [str (y) for y in data_train ['index_beams']]
        label_test = [str (y) for y in data_test ['index_beams']]
    elif connection_type == 'NLOS':
        data_train = valid_data_s008 [valid_data_s008 ['LOS'] == 'LOS=0']
        data_test = valid_data_s009 [valid_data_s009 ['LOS'] == 'LOS=0']
        input_train = data_train ['lidar'].tolist ()
        input_test = data_test ['lidar'].tolist ()
        label_train = [str (y) for y in data_train ['index_beams']]
        label_test = [str (y) for y in data_test ['index_beams']]
    elif connection_type == 'ALL':
        input_train = data_lidar_2D_without_variance_train.tolist ()
        input_test = data_lidar_2D_without_variance_test.tolist ()
        label_train = [str (y) for y in valid_data_s008 ['index_beams']]
        label_test = [str (y) for y in valid_data_s009 ['index_beams']]

    return input_train, input_test, label_train, label_test

def beam_selection_LOS_NLOS(input_type='lidar_coord', connection_type='ALL'):
    train_data_LOS_s009, train_data_NLOS_s009, valid_data_s009 = read_data_s009()
    train_data_LOS_s008, train_data_NLOS_s008, valid_data_s008 = read_data_s008()
    print('Beam selection with: '+input_type+' and '+connection_type)
    if input_type == 'coord':
        preprocess_resolution = 8
        input_train, input_test, label_train, label_test = get_coord_preprocess(connection_type, preprocess_resolution)
        file_name = 'accuracy_' + input_type + '_res_' + str(preprocess_resolution) + '_' + connection_type + '.csv'

    elif input_type == 'lidar':
        input_train, input_test, label_train, label_test = get_lidar_preprocess(connection_type)
        file_name = 'accuracy_' + input_type + '_' + connection_type +'_thr_01' +'.csv'

    elif input_type == 'lidar_coord':
        preprocess_resolution = 8
        print(input_type)
        input_train_coord, input_test_coord, label_train_coord, label_test_coord = get_coord_preprocess (connection_type, preprocess_resolution)
        input_train_lidar, input_test_lidar, label_train_lidar, label_test_lidar = get_lidar_preprocess (connection_type)

        input_train = np.concatenate((input_train_coord, input_train_lidar), axis=1)
        input_test = np.concatenate((input_test_coord, input_test_lidar), axis=1)
        label_train = label_train_coord
        label_test = label_test_coord
        file_name = 'accuracy_' + input_type + '_res_' + str(preprocess_resolution) + '_' + connection_type + '_thr_01' +'.csv'

    print('input_train', len(input_train[0]))
    vector_acuracia, address_size = select_best_beam (input_train,
                                                          input_test,
                                                          label_train,
                                                          label_test,
                                                          figure_name='coord_' + connection_type + '_s008_s009',
                                                          antenna_config='8x8',
                                                          type_of_input='coord',
                                                          titulo_figura='Coordenadas',
                                                          user='s008_s009',
                                                          enableDebug=False,
                                                          plot_confusion_matrix_enable=False)

    path_csv = '../results/score/Wisard/' + input_type + '/' + connection_type + '/'
    df = pd.DataFrame ({"addres_size": address_size, "accuracy": vector_acuracia})
    df.to_csv (path_csv + file_name, index=False)






def select_best_beam(input_train,
                     input_validation,
                     label_train,
                     label_validation,
                     figure_name,
                     antenna_config,
                     type_of_input,
                     titulo_figura,
                     user,
                     enableDebug=False,
                     plot_confusion_matrix_enable=False):

    # config parameters
    if (enableDebug):
        address_size = [28]
        numero_experimentos = 2
    else:
        address_size = [6,12,18,24,28,34,38,44,48,54,58,64]


        numero_experimentos = 1

    vector_time_train_media = []
    vector_time_test_media = []
    vector_acuracia_media = []

    vector_acuracia_desvio_padrao = []
    vector_time_train_desvio_padrao = []
    vector_time_test_desvio_padrao = []

    path_result = "../results"

    vector_acuracia = []
    vector_time_test = []
    vector_time_train = []

    for j in range(len(address_size)):  # For encargado de variar el tamano de la memoria
        for i in range(numero_experimentos):  # For encargado de ejecutar el numero de rodadas (experimentos)


            # -----------------USA LA RED WISARD -------------------
            out_red, time_train, time_test = bsw.redWizard(input_train,
                                                       label_train,
                                                       input_validation,
                                                       address_size[j])

            vector_time_train.append(time_train)
            vector_time_test.append(time_test)

            acuracia = accuracy_score(label_validation, out_red)
            vector_acuracia.append(acuracia)

        # ----------------- CALCULA ESTADISTICAS -----------------------
        [acuracia_media, acuracia_desvio_padrao] = bsw.calculoDesvioPadrao(vector_acuracia)
        [time_train_media, time_train_desvio_padrao] = bsw.calculoDesvioPadrao(vector_time_train)
        [time_test_media, time_test_desvio_padrao] = bsw.calculoDesvioPadrao(vector_time_test)

        # ----------------- GUARDA VECTORES DE ESTADISTICAS -----------------------
        vector_acuracia_media.append(acuracia_media)
        vector_acuracia_desvio_padrao.append(acuracia_desvio_padrao)

        vector_time_train_media.append(time_train_media)
        vector_time_train_desvio_padrao.append(time_train_desvio_padrao)

        vector_time_test_media.append(time_test_media)
        vector_time_test_desvio_padrao.append(time_test_desvio_padrao)

    save_results = False
    if save_results:
        # ----------------- GUARDA EM CSV VECTORES DE ESTADISTICAS  -----------------------
        #print ("-------------------------------------------")
        #print('\n Saving results files ...')
        #print ("-------------------------------------------")

        #with open(path_result + '/accuracy/'+antenna_config+'/'+type_of_input+'/'+user+'/acuracia_' + figure_name + '.csv', 'w') as f:
        with open (path_result + '/score/Wisard/' + figure_name + '/acuracia_' + figure_name + '.csv', 'w') as f:
            writer_acuracy = csv.writer(f, delimiter='\t')
            writer_acuracy.writerows(zip(address_size, vector_acuracia_media, vector_acuracia_desvio_padrao))

        #with open(path_result + '/score/Wisard/'+type_of_input+'/'+user+'/score_' + figure_name + '.csv', 'w') as f:
        #    writer_acuracy = csv.writer(f, delimiter='\t')
        #    writer_acuracy.writerows(zip(address_size, vector_acuracia_media, vector_acuracia_desvio_padrao))

        #with open(path_result + '/processingTime/'+antenna_config+'/'+type_of_input + '/' + user + '/time_train_' + figure_name + '.csv', 'w') as f:
        with open(path_result + '/processingTime/Wisard/' + figure_name + '/time_train_' + figure_name + '.csv', 'w') as f:
            writer_time_train = csv.writer(f, delimiter='\t')
            writer_time_train.writerows(zip(address_size, vector_time_train_media, vector_time_train_desvio_padrao))

        #with open(path_result + '/processingTime/'+antenna_config+'/'+type_of_input + '/' + user +'/time_test_' + figure_name + '.csv', 'w') as f:
        with open(path_result + '/processingTime/Wisard/' + figure_name + '/time_test_' + figure_name + '.csv', 'w') as f:
            writer_time_test = csv.writer(f, delimiter='\t')
            writer_time_test.writerows(zip(address_size, vector_time_test_media, vector_time_test_desvio_padrao))

    plot_results = False
    if plot_results:
        # ----------------- PLOT DE RESULTADOS  ------------------------------
        titulo = titulo_figura
        nombre_curva = "Dado com desvio padrão"

        bsw.plotarResultados(address_size,
                         vector_acuracia_media,
                         vector_acuracia_desvio_padrao,
                         titulo,
                         nombre_curva,
                         "Tamanho da memória",
                         "Acuracia Média",
                         ruta=path_result + '/accuracy/'+antenna_config+'/'+type_of_input + '/' + user +'/acuracia_'+figure_name+'.png')


        bsw.plotarResultados(address_size,
                         vector_time_train_media,
                         vector_time_train_desvio_padrao,
                         titulo,
                         nombre_curva,
                         "Tamanho da memória",
                         "Tempo de treinamento Médio (s)",
                         ruta=path_result + '/processingTime/'+antenna_config+'/'+type_of_input + '/' + user +'/time_train_'+figure_name+'.png''')

        bsw.plotarResultados(address_size,
                         vector_time_test_media,
                         vector_time_test_desvio_padrao,
                         titulo,
                         nombre_curva,
                         "Tamanho da memória",
                         "Tempo de Teste Médio (s)",
                         ruta=path_result + '/processingTime/'+antenna_config+'/'+type_of_input + '/' + user +'/time_test_'+figure_name+'.png')


    return vector_acuracia, address_size


def plot_performance_WiSARD_LOS_NLOS_connection(input_type):

    path = '../results/score/Wisard/'+input_type+'/'
    connection_type = ['LOS', 'NLOS', 'ALL']
    if input_type == 'coord':
        file = 'accuracy_'+input_type+'_res_8_'
    if input_type == 'lidar':
        file = 'accuracy_'+input_type+'_'
    if input_type == 'lidar_coord':
        file = 'accuracy_'+input_type+'_res_8_'

    for i in range(len(connection_type)):
        file_name = file + connection_type[i]+'.csv'
        data = pd.read_csv(path+connection_type[i]+'/'+file_name, delimiter=',')
        plt.plot(data['addres_size'],
                 data['accuracy'],
                 label=connection_type[i],
                 marker='o')
    plt.legend()
    plt.xticks(data['addres_size'])
    plt.grid()
    plt.xlabel('Tamanho da memoria', font='Times New Roman', fontsize=14)
    plt.ylabel('Acuracia',  font='Times New Roman', fontsize=14)
    path_to_save = '../results/score/Wisard/'+input_type+'/'
    file_name = 'performance_accuracy_'+input_type+'LOS_NLOS_ALL.png'
    plt.savefig(path_to_save+file_name, dpi=300, bbox_inches='tight')

def plot_all_performance_WiSARD():

    connection_type = ['LOS', 'NLOS', 'ALL']

    input_type = 'coord'
    path = '../results/score/Wisard/' + input_type + '/'
    file_name = 'accuracy_' + input_type + '_res_8_'+connection_type[0]+'.csv'
    data_LOS_coord = pd.read_csv (path + connection_type[0] + '/' + file_name, delimiter=',')
    file_name = 'accuracy_' + input_type + '_res_8_' + connection_type [1] + '.csv'
    data_NLOS_coord = pd.read_csv (path + connection_type[1] + '/' + file_name, delimiter=',')
    file_name = 'accuracy_' + input_type + '_res_8_' + connection_type [2] + '.csv'
    data_ALL_coord = pd.read_csv (path + connection_type[2] + '/' + file_name, delimiter=',')

    input_type = 'lidar'
    path = '../results/score/Wisard/' + input_type + '/'
    file = 'accuracy_' + input_type + '_'+connection_type[0]+'_thr_01'+'.csv'
    data_LOS_lidar = pd.read_csv (path + connection_type[0] + '/' + file, delimiter=',')
    file = 'accuracy_' + input_type + '_'+connection_type[1]+'_thr_01'+'.csv'
    data_NLOS_lidar = pd.read_csv (path + connection_type[1] + '/' + file, delimiter=',')
    file = 'accuracy_' + input_type + '_'+connection_type[2]+'_thr_01'+'.csv'
    data_ALL_lidar = pd.read_csv (path + connection_type[2] + '/' + file, delimiter=',')

    input_type = 'lidar_coord'
    path = '../results/score/Wisard/' + input_type + '/'
    file = 'accuracy_' + input_type + '_res_8_'+connection_type[0]+'_thr_01'+'.csv'
    data_LOS_lidar_coord = pd.read_csv (path + connection_type[0] + '/' + file, delimiter=',')
    file = 'accuracy_' + input_type + '_res_8_'+connection_type[1]+'_thr_01'+'.csv'
    data_NLOS_lidar_coord = pd.read_csv (path + connection_type[1] + '/' + file, delimiter=',')
    file = 'accuracy_' + input_type + '_res_8_'+connection_type[2]+'_thr_01'+'.csv'
    data_ALL_lidar_coord = pd.read_csv (path + connection_type[2] + '/' + file, delimiter=',')

    fig, ax = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    plt.subplots_adjust (left=0.08, right=0.98, bottom=0.1, top=0.9, hspace=0.12, wspace=0.05)
    size_of_font = 16
    ax[0].plot(data_LOS_coord['addres_size'], data_LOS_coord['accuracy'], label='Coord LOS', marker='o')
    ax[0].plot(data_NLOS_coord['addres_size'], data_NLOS_coord['accuracy'], label='Coord NLOS', marker='o')
    ax[0].plot(data_ALL_coord['addres_size'], data_ALL_coord['accuracy'], label='Coord ALL', marker='o')
    ax[0].grid ()
    ax[0].set_xticks (data_LOS_coord ['addres_size'])
    ax[0].set_xlabel ('Tamanho da memória \n Coordenadas', font='Times New Roman', fontsize=size_of_font)


    ax[1].plot (data_LOS_lidar['addres_size'], data_LOS_lidar['accuracy'], label='Lidar LOS', marker='o')
    ax[1].plot (data_NLOS_lidar['addres_size'], data_NLOS_lidar['accuracy'], label='Lidar NLOS', marker='o')
    ax[1].plot (data_ALL_lidar['addres_size'], data_ALL_lidar['accuracy'], label='Lidar ALL', marker='o')
    ax[1].grid()
    ax[1].set_xticks (data_LOS_coord ['addres_size'])
    ax[1].set_xlabel ('Tamanho da memória \n Lidar', font='Times New Roman', fontsize=size_of_font)

    ax[2].plot (data_LOS_lidar_coord['addres_size'], data_LOS_lidar_coord['accuracy'], label='Lidar Coord LOS', marker='o')
    ax[2].plot (data_NLOS_lidar_coord['addres_size'], data_NLOS_lidar_coord['accuracy'], label='Lidar Coord NLOS', marker='o')
    ax[2].plot (data_ALL_lidar_coord['addres_size'], data_ALL_lidar_coord['accuracy'], label='Lidar Coord ALL', marker='o')
    ax[2].grid ()
    ax[2].set_xticks (data_LOS_coord ['addres_size'])
    ax[2].set_xlabel ('Tamanho da memória \n Lidar e Coordenadas', font='Times New Roman', fontsize=size_of_font)

    ax[0].set_ylabel ('Acurácia', font='Times New Roman', fontsize=size_of_font)
    ax[1].legend ()


    #plt.show()
    plt.savefig('../results/score/Wisard/performance_accuracy_all_LOS_NLOS'+'_thr_01'+'.png', dpi=300, bbox_inches='tight')
    a = 0


#plot_performance_WiSARD_LOS_NLOS_connection('lidar_coord')
#beam_selection_LOS_NLOS(input_type='lidar_coord', connection_type='ALL')
#beam_selection_LOS_NLOS(input_type='lidar_coord', connection_type='LOS')
#beam_selection_LOS_NLOS(input_type='lidar_coord', connection_type='NLOS')

plot_all_performance_WiSARD()