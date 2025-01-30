import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc


def read_results_conventional_evaluation(input_type):
    path = '../../results/score/'
    path_batool =path + 'Batool/top_k/'+input_type +'/'
    file_name ='score_'+input_type+'_top_k.csv'
    data_batool = pd.read_csv(path_batool+file_name)
    batool = data_batool[data_batool['Top-K']<=30]

    path_ruseckas = path + 'ruseckas/top_k/'+input_type + '/'
    file_name ='score_'+input_type+'_top_k.csv'
    data_ruseckas = pd.read_csv(path_ruseckas+file_name)
    ruseckas = data_ruseckas[data_ruseckas['Top-K']<=30]

    path_wisard =path + 'Wisard/top_k/'+input_type+'/'
    file_name ='score_'+input_type+'_top_k.csv'
    data_wisard = pd.read_csv(path_wisard+file_name)
    wisard = data_wisard[data_wisard['Top-K']<=30]

    return data_batool, data_ruseckas, data_wisard
    #return batool, ruseckas, wisard
def plot_results_with_conventional_evaluation():
    lidar_batool, lidar_ruseckas, lidar_wisard = read_results_conventional_evaluation('lidar')
    coord_batool, coord_ruseckas, coord_wisard = read_results_conventional_evaluation('coord')
    coord_lidar_batool, coord_lidar_ruseckas, coord_lidar_wisard = read_results_conventional_evaluation('lidar_coord')

    fig, ax = plt.subplots (1, 3, figsize=(14, 6), sharey=True)
    plt.subplots_adjust (left=0.08, right=0.98, bottom=0.1, top=0.9, hspace=0.12, wspace=0.05)
    size_of_font = 16
    ax [0].plot (coord_wisard['Top-K'], coord_wisard ['Acuracia'], label='WiSARD',  color='red', marker='o')
    ax [0].plot (coord_ruseckas['Top-K'], coord_ruseckas['Acuracia'], label='Ruseckas',  color='teal', marker='o')
    ax [0].plot (coord_batool['Top-K'], coord_batool['Acuracia'], label='Batool',  color='purple', marker='o')
    ax [0].grid ()
    #ax [0].set_xticks (coord_wisard['Top-K'])
    ax [0].set_xlabel ('Top-K \n Coordenadas', font='Times New Roman', fontsize=size_of_font)

    ax [1].plot (lidar_wisard ['Top-K'], lidar_wisard['Acuracia'], label='WiSARD', color='red', marker='o')
    ax [1].plot (lidar_ruseckas['Top-K'], lidar_ruseckas['Acuracia'], label='Ruseckas', color='teal', marker='o')
    ax [1].plot (lidar_batool['Top-K'], lidar_batool['Acuracia'], label='Batool', color='purple', marker='o')
    ax [1].grid ()
    #ax [1].set_xticks(coord_wisard['Top-K'])
    ax [1].set_xlabel ('Top-K \n LiDAR', font='Times New Roman', fontsize=size_of_font)

    ax [2].plot (coord_lidar_wisard['Top-K'], coord_lidar_wisard['Acuracia'], label='WiSARD',color='red', marker='o')
    ax [2].plot (coord_lidar_ruseckas ['Top-K'], coord_lidar_ruseckas['Acuracia'], label='Ruseckas', color='teal', marker='o')
    ax [2].plot (coord_lidar_batool['Top-K'], coord_lidar_batool['Acuracia'], label='Batool', color='purple', marker='o')

    ax [2].grid ()
    #ax [2].set_xticks(coord_wisard['Top-K'])
    ax [2].set_xlabel ('Top-K \n Coordenadas + LiDAR', font='Times New Roman', fontsize=size_of_font)

    ax [0].set_ylabel ('Acurácia', font='Times New Roman', fontsize=size_of_font)
    ax [1].legend ()

    #plt.show()
    path_to_save = '../../results/score/plot_for_jornal/'
    plt.savefig (path_to_save+'comparacao_convencional_WiSARD_Ruseckas_batool_top_30.png', dpi=300, bbox_inches='tight')
    a = 0

def read_througput_radio(input_type, tech, true_all_power_norm, all_possible_power_norm):
    import sys
    sys.path.insert (1, '../')
    import throughput as throughput

    #true_all_power_norm, all_possible_power_norm, true_beam_index = power_of_sinal_rx ()

    technique = tech
    path_index_beams_estimated = '../../results/index_beams_predict/' + technique + '/top_k/' + input_type + '/'
    filename = 'index_beams_predict_top_k.npz'

    index_estimated_wisard = throughput.read_index_beams_estimated_novo (path_index_beams_estimated, filename)
    througput_ratio_wisard = {}
    for i in range (len (index_estimated_wisard)):
        best_power_top_k, all_power_order_top_k = throughput.calculate_top_k_all_power (index_estimated_wisard [i + 1],
                                                                                        all_possible_power_norm)
        througput_ratio_wisard [i + 1] = throughput.througput_ratio (true_all_power_norm, best_power_top_k)

    ratio_thr_wisard = [througput_ratio_wisard [key] for key in througput_ratio_wisard.keys ()]
    ratio_thr_wisard = np.array (ratio_thr_wisard)

    return ratio_thr_wisard

def plot_througput_of_all_techniques():
    true_all_power_norm, all_possible_power_norm, true_beam_index = power_of_sinal_rx ()

    input_type = 'coord'
    rt_coord_wisard = read_througput_radio(input_type, 'WiSARD', true_all_power_norm, all_possible_power_norm)
    rt_coord_ruseckas = read_througput_radio(input_type, 'Ruseckas', true_all_power_norm, all_possible_power_norm)
    rt_coord_batool = read_througput_radio(input_type, 'Batool', true_all_power_norm, all_possible_power_norm)

    input_type = 'lidar'
    rt_lidar_wisard = read_througput_radio(input_type, 'WiSARD', true_all_power_norm, all_possible_power_norm)
    rt_lidar_ruseckas = read_througput_radio(input_type, 'Ruseckas', true_all_power_norm, all_possible_power_norm)
    rt_lidar_batool = read_througput_radio(input_type, 'Batool', true_all_power_norm, all_possible_power_norm)

    input_type = 'lidar_coord'
    rt_lidar_coord_wisard = read_througput_radio(input_type, 'WiSARD', true_all_power_norm, all_possible_power_norm)
    rt_lidar_coord_ruseckas = read_througput_radio(input_type, 'Ruseckas', true_all_power_norm, all_possible_power_norm)
    rt_lidar_coord_batool = read_througput_radio(input_type, 'Batool', true_all_power_norm, all_possible_power_norm)

    top_k = np.arange (1, 31, 1)
    fig, ax = plt.subplots (1, 3, figsize=(14, 6), sharey=True)
    plt.subplots_adjust (left=0.08, right=0.98, bottom=0.1, top=0.9, hspace=0.12, wspace=0.05)
    size_of_font = 16
    ax [0].plot (top_k, rt_coord_wisard[0:30], label='WiSARD', color='red', linestyle='dashed')
    ax [0].plot (top_k, rt_coord_ruseckas[0:30], label='Ruseckas', color='teal', linestyle='dashed')
    ax [0].plot (top_k, rt_coord_batool[0:30], label='Batool', color='purple', linestyle='dashed')
    ax [0].grid ()
    # ax [0].set_xticks (coord_wisard['Top-K'])
    ax [0].set_xlabel ('Top-K \n Coordenadas', font='Times New Roman', fontsize=size_of_font)

    ax [1].plot (top_k, rt_lidar_wisard[0:30], label='WiSARD', color='red', linestyle='dashed')
    ax [1].plot (top_k, rt_lidar_ruseckas[0:30], label='Ruseckas', color='teal', linestyle='dashed')
    ax [1].plot (top_k, rt_lidar_batool[0:30], label='Batool', color='purple', linestyle='dashed')
    ax [1].grid ()
    # ax [1].set_xticks(coord_wisard['Top-K'])
    ax [1].set_xlabel ('Top-K \n LiDAR', font='Times New Roman', fontsize=size_of_font)

    ax [2].plot (top_k, rt_lidar_coord_wisard[0:30], label='WiSARD', color='red', linestyle='dashed')
    ax [2].plot (top_k, rt_lidar_coord_ruseckas[0:30], label='Ruseckas', color='teal', linestyle='dashed')
    ax [2].plot (top_k, rt_lidar_coord_batool[0:30], label='Batool', color='purple', linestyle='dashed')

    ax [2].grid ()
    # ax [2].set_xticks(coord_wisard['Top-K'])
    ax [2].set_xlabel ('Top-K \n Coordenadas + LiDAR', font='Times New Roman', fontsize=size_of_font)

    ax [0].set_ylabel ('Througput Ratio', font='Times New Roman', fontsize=size_of_font)
    ax [1].legend ()

    # plt.show()
    path_to_save = '../../results/score/plot_for_jornal/'
    plt.savefig (path_to_save + 'RT_comparacao_convencional_WiSARD_Ruseckas_batool_top_30.png', dpi=300,
                 bbox_inches='tight')

    a=0



def read_beams_output_generated_by_ray_tracing():
    print("\t\tRead Beams output generated from Ray-tracing ")
    path = '../../data/beams_output/beam_output_baseline_raymobtime_s008/'
    beam_output_train = np.load(path + "beams_output_train.npz", allow_pickle=True)['output_classification']

    path = '../../data/beams_output/beam_output_baseline_raymobtime_s009/'
    beam_output_test = np.load(path + "beams_output_test.npz", allow_pickle=True)['output_classification']

    return beam_output_train, beam_output_test

def calculate_index_beams(beam_output):
    # calculate the index of the best beam
    tx_size = beam_output.shape[2]

    # Reshape beam pair index
    num_classes = beam_output.shape[1] * beam_output.shape[2]
    beams = beam_output.reshape(beam_output.shape[0], num_classes)

    # Beams distribution
    best_beam_index = []
    for sample in range(beams.shape[0]):
        best_beam_index.append(np.argmax(beams[sample, :]))

    return(best_beam_index)

def power_of_sinal_rx():
    beam_output_train, beam_output_test = read_beams_output_generated_by_ray_tracing()

    true_all_power_norm = np.zeros ((beam_output_test.shape[0], 1))
    estimated_all_power_norm = np.zeros ((beam_output_test.shape[0], 1))

    # Beam index true
    true_beam_index = calculate_index_beams(beam_output_test)

    # Calculate the power of the signal received TRUE
    for i in range(len(beam_output_test)):
        a = beam_output_test[i].flatten()
        power_norm = [np.linalg.norm(i)**2 for i in a]
        true_power_norm = power_norm[true_beam_index[i]]
        true_all_power_norm[i] = true_power_norm
        #true_all_power_norm.append(true_power_norm)

    #calculate ALL possible power of the signal received
    all_possible_power_norm = np.zeros ((beam_output_test.shape[0], 256))
    for i in range (len (beam_output_test)):
        a = beam_output_test [i].flatten ()
        power_norm = [np.linalg.norm(i) ** 2 for i in a]
        all_possible_power_norm[i] = power_norm

    return true_all_power_norm, all_possible_power_norm, true_beam_index

def plot_time_train():
    data_batool = [19.3647, 1310.276, 2301.6973]
    data_ruseckas= [ 19.744, 2187.7453, 2180.0649]
    data_wisard =[1.8348, 0.8296, 2.8824]
    name = ['Coordenadas', 'LiDAR', 'Coordenadas + LiDAR']

    barWidth = 0.3
    position = [1, 3, 5]
    position2 = [x + barWidth for x in position]
    position3 = [x + barWidth for x in position2]
    batool = np.array(data_batool)
    ruseckas = np.array(data_ruseckas)
    wisard = np.array(data_wisard)

    plt.rcParams["font.family"] = "Times"
    plt.rcParams["font.size"] = "16"
    fig, ax1 = plt.subplots (figsize=(13, 7))

    plt.bar(position, data_batool, width=0.3, color='purple', label='Batool')
    plt.bar(position2, data_ruseckas, width=0.3, color='teal', label='Ruseckas')
    for i in range(len(position)):
        plt.text(position[i], data_batool[i], str(round(data_batool[i],1)), ha='center', va='bottom')
        plt.text(position2[i], data_ruseckas[i], str(round(data_ruseckas[i],1)), ha='center', va='bottom')

    ax1.set_ylabel ('Tempo de Treinamento [seg]', fontsize=16, font='Times New Roman')
    #fig.legend (loc='outside upper center', ncol=2, frameon=False)#, columnspacing=0.5, numcolumns=2)
    plt.grid()

    ax2 = ax1.twinx ()
    plt.bar(position3, data_wisard, width=0.3, color='red', label='WiSARD')
    for i in range(len(position)):
        plt.text(position3[i], data_wisard[i], str(round(data_wisard[i],1)), ha='center', va='bottom')
    ax2.set_ylabel ('Tempo de Treinamento [seg]', fontsize=16, font='Times New Roman')
    fig.legend( loc='outside upper center', frameon=False, ncol=3, )
    plt.xticks (position2,
                ['Coordenadas', 'LiDAR', 'Coord+LiDAR'], fontsize=16, font='Times New Roman')
    plt.grid()
    plt.savefig('../../results/score/plot_for_jornal/comparacao_tempo_treinamento.png', dpi=300, bbox_inches='tight')

    a=0


#plot_results_with_conventional_evaluation()
#plot_througput_of_all_techniques()
plot_time_train()
