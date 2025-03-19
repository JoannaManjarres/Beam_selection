import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_results_conventional_evaluation_inverter_dataset(input_type, connection_type):
    path = '../../results/inverter_dataset/score/'
    path_batool =path + 'Batool/'+input_type +'/'+connection_type+'/'
    file_name ='accuracy_'+input_type+'.csv'
    data_batool = pd.read_csv(path_batool+file_name)
    batool = data_batool[data_batool['top-k']<=30]

    path_wisard = path + 'Wisard/top-k/'+input_type + '/' +connection_type+'/'
    if input_type =='coord':
        file_name = 'accuracy_coord_res_8_ALL.csv'
    if input_type == 'lidar':
        file_name = 'accuracy_lidar_ALL_thr_01.csv'
    if input_type == 'lidar_coord':
        file_name = 'accuracy_lidar_coord_res_8_ALL_thr_01.csv'
    data_wisard = pd.read_csv(path_wisard + file_name)
    #wisard = data_wisard [data_wisard ['top-k']]
    wisard = data_wisard[data_wisard['top-k']<=30]
    top_k = [1, 5, 10, 15, 20, 25, 30]
    wisard_top_k = []
    for i in range(len(top_k)):
        w = wisard [wisard ['top-k'] == top_k [i]] ['score'].values
        wisard_top_k.append(w[0])

    df_score_wisard_top_k = pd.DataFrame ({"Top-K": top_k, "Acuracia": wisard_top_k})
    df_score_batool_top_k = pd.DataFrame ({"Top-K": top_k, "Acuracia": batool ['score']})

    return df_score_wisard_top_k, df_score_batool_top_k


def plot_results_inverter_dataset():
    coord_wisard, coord_batool = read_results_conventional_evaluation_inverter_dataset (input_type='coord', connection_type='ALL')
    lidar_wisard, lidar_batool = read_results_conventional_evaluation_inverter_dataset (input_type='lidar', connection_type='ALL')
    coord_lidar_wisard, coord_lidar_batool = read_results_conventional_evaluation_inverter_dataset (input_type='lidar_coord', connection_type='ALL')

    fig, ax = plt.subplots (1, 3, figsize=(14, 6), sharey=True)
    plt.subplots_adjust (left=0.08, right=0.98, bottom=0.1, top=0.9, hspace=0.12, wspace=0.05)

    size_of_font = 18
    ax [0].plot (coord_wisard ['Top-K'], coord_wisard ['Acuracia'], label='WiSARD', color='red', marker='o')
    #ax [0].plot (coord_ruseckas ['Top-K'], coord_ruseckas ['Acuracia'], label='Ruseckas', color='teal', marker='o')
    ax [0].plot (coord_batool ['Top-K'], coord_batool ['Acuracia'], label='Batool', color='purple', marker='o')
    ax [0].grid ()
    # ax [0].set_xticks (coord_wisard['Top-K'])
    ax [0].set_xlabel ('Coordenadas \n K  ', font='Times New Roman', fontsize=size_of_font)

    ax [1].plot (lidar_wisard ['Top-K'], lidar_wisard ['Acuracia'], label='WiSARD', color='red', marker='o')
    #ax [1].plot (lidar_ruseckas ['Top-K'], lidar_ruseckas ['Acuracia'], label='Ruseckas', color='teal', marker='o')
    ax [1].plot (lidar_batool ['Top-K'], lidar_batool ['Acuracia'], label='Batool', color='purple', marker='o')
    ax [1].grid ()
    # ax [1].set_xticks(coord_wisard['Top-K'])
    ax [1].set_xlabel ('LiDAR \n K  ', font='Times New Roman', fontsize=size_of_font)

    ax [2].plot (coord_lidar_wisard ['Top-K'], coord_lidar_wisard ['Acuracia'], label='WiSARD', color='red', marker='o')
    #ax [2].plot (coord_lidar_ruseckas ['Top-K'], coord_lidar_ruseckas ['Acuracia'], label='Ruseckas', color='teal',marker='o')
    ax [2].plot (coord_lidar_batool ['Top-K'], coord_lidar_batool ['Acuracia'], label='Batool', color='purple',
                 marker='o')

    ax [2].grid ()
    # ax [2].set_xticks(coord_wisard['Top-K'])
    ax [2].set_xlabel ('Coordenadas + LiDAR \n K', font='Times New Roman', fontsize=size_of_font)

    ax [0].set_ylabel ('Acurácia top-k', font='Times New Roman', fontsize=size_of_font)
    ax [1].legend ()
    fig.suptitle ('Selecao de feixes top-k com dataset invertido', fontsize=size_of_font, font='Times New Roman')


    a=0

plot_results_inverter_dataset()