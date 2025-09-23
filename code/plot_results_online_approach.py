import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def read_csv_file(input_type, ref_name, window):
    path = '../results/score/' + ref_name
    if ref_name == 'ruseckas':
        specific_path = '/online/top_k/'+ input_type +'/'+ window + '_window/'
    else:
        specific_path = '/servidor_land/online/' + input_type + '/' + window + '_window/'

    filename = 'all_results_'+window+'_window_top_k.csv'
    if ref_name == 'wisard' and window == 'fixed' or window == 'incremental':
        filename = '0_'+filename
    if ref_name == 'wisard' and window == 'sliding':
        filename = '0_all_results_'+window+'_window_1000_top_k.csv'
    if ref_name == 'Batool' and window == 'sliding':
        filename = 'all_results_'+window+'_window_1000_top_k.csv'
    if ref_name == 'Batool' and window == 'incremental':
        filename =  'all_results_'+window+'_window_top_k.csv'
    usecols = ["top-k", "score", "test_time", "samples_tested",	"episode", "trainning_process_time", "samples_trainning"]
    print(path+specific_path + filename)
    data = pd.read_csv(path +specific_path+ filename)#, header=None, usecols=usecols, names=usecols)
    '''
    print(path + filename)
    accuracy = data['Acuracia'].tolist()
    accuracy = [float (i) for i in accuracy [1:]]
    accuracy = [round (i, 2) for i in accuracy]

    top_k = data ['Top-k'].tolist()
    top_k = [float(i) for i in top_k[1:]]
    '''

    return data

def read_top_k_accuracy_results_ref(data):
    top_k = [1,5,10,20,30]
    all_score = []
    for k in top_k:
        score_top_k = data[data['top-k'] == k]['score'].mean()
        print ('Top-'+str(k)+': '+ str(round(score_top_k, 4)))
        all_score.append(round(score_top_k, 4))

    score_results = pd.DataFrame({'top-k': top_k, 'score': all_score})

    return score_results

def read_coord_results_ref(window, input):
    #input = 'coord'
    #window = 'fixed'

    if window == 'incremental':

        fixed_window_wisard = read_csv_file (input_type=input,
                                             ref_name='wisard',
                                             window=window)
        wisard_accuracy_top_k = read_top_k_accuracy_results_ref (fixed_window_wisard)

        return wisard_accuracy_top_k

    else:
        fixed_window_wisard = read_csv_file (input_type=input,
                                         ref_name='wisard',
                                         window=window)
        wisard_accuracy_top_k = read_top_k_accuracy_results_ref(fixed_window_wisard)

        fixed_window_ruseckas = read_csv_file (input_type=input,
                                           ref_name='ruseckas',
                                           window=window)
        ruseckas_accuracy_top_k = read_top_k_accuracy_results_ref(fixed_window_ruseckas)

        fixed_window_batool = read_csv_file (input_type=input,
                                         ref_name='Batool',
                                         window=window)
        batool_accuracy_top_k = read_top_k_accuracy_results_ref(fixed_window_batool)

        return wisard_accuracy_top_k, ruseckas_accuracy_top_k, batool_accuracy_top_k

def plot_compare_bars_accuracy_models():
    coord_incremental_wisard = read_coord_results_ref (window='incremental',
                                                       input='coord')
    lidar_incremental_wisard = read_coord_results_ref (window='incremental',
                                                       input='lidar')
    lidar_coord_incremental_wisard = read_coord_results_ref (window='incremental',
                                                             input='lidar_coord')
    coord_fixed_wisard, coord_fixed_ruseckas, coord_fixed_batool = read_coord_results_ref (window='fixed',
                                                                                           input='coord')
    lidar_fixed_wisard, lidar_fixed_ruseckas, lidar_fixed_batool = read_coord_results_ref (window='fixed',
                                                                                           input='lidar')
    lidar_coord_fixed_wisard, lidar_coord_fixed_ruseckas, lidar_coord_fixed_batool = read_coord_results_ref (
        window='fixed', input='lidar_coord')

    coord_sliding_wisard, coord_sliding_ruseckas, coord_sliding_batool = read_coord_results_ref (window='sliding',
                                                                                                 input='coord')
    lidar_sliding_wisard, lidar_sliding_ruseckas, lidar_sliding_batool = read_coord_results_ref (window='sliding',
                                                                                                 input='lidar')
    lidar_coord_sliding_wisard, lidar_coord_sliding_ruseckas, lidar_coord_sliding_batool = read_coord_results_ref (
        window='sliding', input='lidar_coord')

    fig, axes = plt.subplots (3, 3, figsize=(12, 6), sharex=True, sharey=True)

    # Deixa os eixos em 1D para facilitar iteração
    axes = axes.ravel ()
    # Remove espaços entre os subplots
    plt.subplots_adjust (wspace=0, hspace=0.03)
    plt.rcParams.update ({"font.size": 14,  # tamanho da fonte
                          "font.family": "Times New Roman"  # tipo de fonte (ex: 'serif', 'sans-serif', 'monospace')
    })


    #x_axis = np.array([2,6,10,14,18])
    x_axis = np.array([2,4,6,8,10])
    width_bar = 0.4
    alpha_grid = 0.4
    size_font = 14

    #1.5 2 2.5    3    3.5 4 4.5    5    5.5 6 6.5  7    7.5 8 8.5    9    9.5 10 10.5   11   11.5 12 12.5   13   13.5 14 14.5   15   15.5 16 16.5   17   17.5 18 18.5

    axes [0].bar (x_axis - 0.5, coord_fixed_wisard['score'], width=width_bar, label='Wisard')
    axes [0].bar (x_axis, coord_fixed_ruseckas['score'], width=width_bar, label='Ruseckas')
    axes [0].bar (x_axis + 0.5, coord_fixed_batool['score'], width=width_bar, label='Batool')
    #axes [0].set_title ('GPS Input - Fixed Window')
    axes [0].grid (True, linestyle="--", alpha=alpha_grid)
    axes [0].set_ylabel('Fixed', fontfamily="serif", fontsize=size_font)
    axes[0].spines['top'].set_visible(False)

    axes [1].bar (x_axis - 0.5, lidar_fixed_wisard['score'], width=width_bar)#, label='Wisard')
    axes [1].bar (x_axis, lidar_fixed_ruseckas['score'], width=width_bar)#, label='Ruseckas')
    axes [1].bar (x_axis + 0.5, lidar_fixed_batool['score'], width=width_bar)#, label='Batool')
    axes [1].grid (True, linestyle="--", alpha=alpha_grid)
    axes[1].spines['top'].set_visible(False)

    axes [2].bar (x_axis - 0.5, lidar_coord_fixed_wisard['score'], width=width_bar)#, label='Wisard')
    axes [2].bar (x_axis, lidar_coord_fixed_ruseckas['score'], width=width_bar)#, label='Ruseckas')
    axes [2].bar (x_axis + 0.5, lidar_coord_fixed_batool['score'], width=width_bar)#, label='Batool')
    axes [2].grid (True, linestyle="--", alpha=alpha_grid)
    axes [2].spines['top'].set_visible(False)
    axes [2].spines['right'].set_visible(False)

    axes [3].bar (x_axis - 0.5, coord_sliding_wisard['score'], width=width_bar)#, label='Wisard')
    axes [3].bar (x_axis, coord_sliding_ruseckas['score'], width=width_bar)#, label='Ruseckas')
    axes [3].bar (x_axis + 0.5, coord_sliding_batool['score'], width=width_bar)#, label='Batool')
    axes [3].grid (True, linestyle="--", alpha=alpha_grid)
    axes[3].set_ylabel('Sliding', fontfamily="serif", fontsize=size_font)
    axes[3].spines['top'].set_visible(False)

    axes [4].bar (x_axis - 0.5, lidar_sliding_wisard['score'], width=width_bar)
    axes [4].bar (x_axis, lidar_sliding_ruseckas['score'], width=width_bar)
    axes [4].bar (x_axis + 0.5, lidar_sliding_batool['score'], width=width_bar)
    axes [4].grid (True, linestyle="--", alpha=alpha_grid)
    axes [4].spines['top'].set_visible(False)

    axes [5].bar (x_axis - 0.5, lidar_coord_sliding_wisard['score'], width=width_bar)
    axes [5].bar (x_axis, lidar_coord_sliding_ruseckas['score'], width=width_bar)
    axes [5].bar (x_axis + 0.5, lidar_coord_sliding_batool['score'], width=width_bar)
    axes [5].grid (True, linestyle="--", alpha=alpha_grid)
    axes [5].spines['top'].set_visible(False)
    axes [5].spines['right'].set_visible(False)

    axes [6].bar (x_axis - 0.5, coord_incremental_wisard['score'], width=width_bar)#, label='Wisard')
    #axes [6].bar (x_axis, coord_incremental_ruseckas['score'], width=width_bar, label='Ruseckas')
    #axes [6].bar (x_axis + 0.5, coord_incremental_batool['score'], width=width_bar, label='Batool')
    axes[6].grid (True, linestyle="--", alpha=alpha_grid)
    axes[6].set_xlabel('GPS data \n Top-k', fontfamily="serif", labelpad=10, fontsize=size_font)
    axes[6].set_ylabel('Incremental', fontfamily="serif", fontsize=size_font)
    axes[6].spines['top'].set_visible(False)
    axes [6].set_xticklabels ([1, 5, 10, 20, 30], fontfamily="serif", fontsize=size_font)

    axes [7].bar (x_axis - 0.5, lidar_incremental_wisard['score'], width=width_bar)#, label='Wisard')
    #axes [7].bar (x_axis, lidar_incremental_ruseckas['score'], width=width_bar, label='Ruseckas')
    #axes [7].bar (x_axis + 0.5, lidar_incremental_batool['score'], width=width_bar, label='Batool')
    #axes [7].set_title ('LiDAR Input - Incremental Window')
    axes [7].grid (True, linestyle="--", alpha=alpha_grid)
    axes [7].set_xlabel('LiDAR data \n Top-k', fontfamily="serif", labelpad=10, fontsize=size_font)
    axes [7].spines['top'].set_visible(False)
    axes [7].set_xticklabels ([1, 5, 10, 20, 30], fontfamily="serif", fontsize=size_font)


    axes [8].bar (x_axis - 0.5, lidar_coord_incremental_wisard['score'], width=width_bar)
    #axes [8].bar (x_axis, lidar_coord_incremental_ruseckas['score'], width=width_bar, label='Ruseckas')
    #axes [8].bar (x_axis + 0.5, lidar_coord_incremental_batool['score'], width=width_bar, label='Batool')
    axes [8].grid (True, linestyle="--", alpha=alpha_grid)
    axes [8].set_xlabel ('LiDAR + GPS data \nTop-k', fontfamily="serif", labelpad=10, fontsize=size_font)
    axes [8].spines['top'].set_visible(False)
    axes [8].spines['right'].set_visible(False)
    axes[8].set_xticks(x_axis)
    axes[8].set_xticklabels([1,5,10,20,30], fontfamily="serif", fontsize=size_font)
    axes[8].set_yticks([0.2,0.4,0.6,0.8,1.0])

    fig.supylabel ("Accuracy")
    fig.legend (loc="upper right", ncol=3, frameon=False)#, bbox_to_anchor=(0.9, 0.95))
    plt.tight_layout (rect=[0, 0, 1, 0.95])
    plt.savefig ('../results/score/compare_online_all_ref_all_windows.png', dpi=300, bbox_inches='tight')




    a=0




def plot_compare_accuracy_models():
    coord_incremental_wisard = read_coord_results_ref (window='incremental', input='coord')
    lidar_incremental_wisard = read_coord_results_ref (window='incremental', input='lidar')
    lidar_coord_incremental_wisard = read_coord_results_ref (window='incremental',
                                                                                             input='lidar_coord')
    coord_fixed_wisard, coord_fixed_ruseckas, coord_fixed_batool = read_coord_results_ref(window='fixed', input='coord')
    lidar_fixed_wisard, lidar_fixed_ruseckas, lidar_fixed_batool = read_coord_results_ref(window='fixed', input='lidar')
    lidar_coord_fixed_wisard, lidar_coord_fixed_ruseckas, lidar_coord_fixed_batool = read_coord_results_ref(window='fixed', input='lidar_coord')

    coord_sliding_wisard, coord_sliding_ruseckas, coord_sliding_batool = read_coord_results_ref(window='sliding', input='coord')
    lidar_sliding_wisard, lidar_sliding_ruseckas, lidar_sliding_batool = read_coord_results_ref(window='sliding', input='lidar')
    lidar_coord_sliding_wisard, lidar_coord_sliding_ruseckas, lidar_coord_sliding_batool = read_coord_results_ref(window='sliding', input='lidar_coord')


    #plot in 3 subplot with line graphs the accuracy of the models


    # Criação da figura com 6 subplots (2 linhas, 3 colunas), compartilhando eixos
    fig, axes = plt.subplots (3, 3, figsize=(12, 6), sharex=True, sharey=True)

    # Deixa os eixos em 1D para facilitar iteração
    axes = axes.ravel ()

    axes[0].plot(coord_fixed_wisard['top-k'], coord_fixed_wisard['score'], label='Wisard')
    axes[0].plot(coord_fixed_ruseckas['top-k'], coord_fixed_ruseckas['score'], label='Ruseckas')
    axes[0].plot(coord_fixed_batool['top-k'], coord_fixed_batool['score'], label='Batool')
    axes[0].set_title('GPS Input - Fixed Window')
    axes[0].grid (True, linestyle="--", alpha=0.5)

    axes[1].plot(lidar_fixed_wisard['top-k'], lidar_fixed_wisard['score'], label='Wisard')
    axes[1].plot(lidar_fixed_ruseckas['top-k'], lidar_fixed_ruseckas['score'], label='Ruseckas')
    axes[1].plot(lidar_fixed_batool['top-k'], lidar_fixed_batool['score'], label='Batool')
    axes[1].set_title('LiDAR Input - Fixed Window')
    axes[1].grid (True, linestyle="--", alpha=0.5)

    axes[2].plot(lidar_coord_fixed_wisard['top-k'], lidar_coord_fixed_wisard['score'], label='Wisard')
    axes[2].plot(lidar_coord_fixed_ruseckas['top-k'], lidar_coord_fixed_ruseckas['score'], label='Ruseckas')
    axes[2].plot(lidar_coord_fixed_batool['top-k'], lidar_coord_fixed_batool['score'], label='Batool')
    axes[2].set_title('LiDAR + GPS Input - Fixed Window')
    axes[2].grid (True, linestyle="--", alpha=0.5)

    axes[3].plot(coord_sliding_wisard['top-k'], coord_sliding_wisard['score'], label='Wisard')
    axes[3].plot(coord_sliding_ruseckas['top-k'], coord_sliding_ruseckas['score'], label='Ruseckas')
    axes[3].plot(coord_sliding_batool['top-k'], coord_sliding_batool['score'], label='Batool')
    #axes[3].set_title('GPS Input - Sliding Window')
    axes[3].grid (True, linestyle="--", alpha=0.5)

    axes[4].plot(lidar_sliding_wisard['top-k'], lidar_sliding_wisard['score'], label='Wisard')
    axes[4].plot(lidar_sliding_ruseckas['top-k'], lidar_sliding_ruseckas['score'], label='Ruseckas')
    axes[4].plot(lidar_sliding_batool['top-k'], lidar_sliding_batool['score'], label='Batool')
    #axes[4].set_title('LiDAR Input - Sliding Window')
    axes[4].grid (True, linestyle="--", alpha=0.5)

    axes[5].plot(lidar_coord_sliding_wisard['top-k'], lidar_coord_sliding_wisard['score'], label='Wisard')
    axes[5].plot(lidar_coord_sliding_ruseckas['top-k'], lidar_coord_sliding_ruseckas['score'], label='Ruseckas')
    axes[5].plot(lidar_coord_sliding_batool['top-k'], lidar_coord_sliding_batool['score'], label='Batool')
    #axes[5].set_title('LiDAR + GPS Input - Sliding Window')
    axes[5].grid (True, linestyle="--", alpha=0.5)
    axes[5].legend (loc="upper right", fontsize=8)

    axes[6].plot(coord_incremental_wisard['top-k'], coord_incremental_wisard['score'], label='Wisard')
    #axes[6].plot(coord_incremental_ruseckas['top-k'], coord_incremental_ruseckas['score'], label='Ruseckas')
    #axes[6].plot(coord_incremental_batool['top-k'], coord_incremental_batool['score'], label='Batool')
    axes[6].grid (True, linestyle="--", alpha=0.5)
    axes[6].legend (loc="upper right", fontsize=8)

    axes[7].plot(lidar_incremental_wisard['top-k'], lidar_incremental_wisard['score'], label='Wisard')
    #axes[7].plot(lidar_incremental_ruseckas['top-k'], lidar_incremental_ruseckas['score'], label='Ruseckas')
    #axes[7].plot(lidar_incremental_batool['top-k'], lidar_incremental_batool['score'], label='Batool')
    axes[7].grid (True, linestyle="--", alpha=0.5)
    axes[7].legend (loc="upper right", fontsize=8)

    axes[8].plot(lidar_coord_incremental_wisard['top-k'], lidar_coord_incremental_wisard['score'], label='Wisard')
    #axes[8].plot(lidar_coord_incremental_ruseckas['top-k'], lidar_coord_incremental_ruseckas['score'], label='Ruseckas')
    #axes[8].plot(lidar_coord_incremental_batool['top-k'], lidar_coord_incremental_batool['score'], label='Batool')
    axes[8].grid (True, linestyle="--", alpha=0.5)
    axes[8].legend (loc="upper right", fontsize=8)

    # Ajuste de layout
    #fig.suptitle ("Comparação de Modelos de Classificação", fontsize=14)

    plt.tight_layout (rect=[0, 0, 1, 0.95])

def merge_file_ruseckas_files():
    path = '../results/score/ruseckas/online/top_k/lidar/fixed_window/'
    filename1 = 'all_results_fixed_window_top_k_parte_1.csv'
    filename2 = 'all_results_fixed_window_top_k_parte_2.csv'
    filename3 = 'all_results_fixed_window_top_k_parte_3.csv'

    usecols = ["top-k", "score", "test_time", "samples_tested",	"episode", "trainning_process_time", "samples_trainning"]
    data1 = pd.read_csv(path + filename1)#, header=None, usecols=usecols, names=usecols)
    data2 = pd.read_csv(path + filename2)#, header=None, usecols=usecols, names=usecols)
    data3 = pd.read_csv(path + filename3)#, header=None, usecols=usecols, names=usecols)

    frames = [data1, data2, data3]
    result = pd.concat(frames)

    result.to_csv(path + 'all_results_fixed_window_top_k.csv', index=False, header=True)


plot_compare_bars_accuracy_models()
plot_compare_accuracy_models()







a=0