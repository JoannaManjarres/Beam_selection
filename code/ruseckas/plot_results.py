import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
sys.path.append ("../")

import plot_results as p

def read_results_for_plot(type_of_input, type_of_window, window_size=0):

    window_size = str(window_size)
    flag_servidor = 3
    filename = 'all_results_' + type_of_window + '_top_k.csv'
    metric = 'time_trainning'  # 'score'
    servidor = 'vm_land'
    title = 'Beam Selection using with ' + type_of_input + ' and ' + type_of_window + '\n Reference: Batool -' + servidor


    path = '../../results/score/ruseckas/online/top_k/' + type_of_input + '/' + type_of_window + '/'

    pos_x = [10, 250, 500, 750, 1000, 1250, 1500]
    pos_y = 0.8

    all_results_fixed_window = pd.read_csv(path + filename)

    return path, servidor, filename, pos_x, pos_y, title, all_results_fixed_window

def plot_accum_score_top_k(pos_x, pos_y, path, title, filename, window_size=0):
    import tools as tls
    all_csv_data = pd.read_csv (path + filename)
    top_k = [1, 5, 10, 15, 20, 25, 30]
    color = ['blue', 'red', 'green', 'purple', 'orange', 'maroon',
             'teal']  # 'maroon', 'teal', 'black', 'gray', 'brown', 'cyan', 'magenta', 'yellow', 'olive', 'navy', 'lime', 'aqua', 'fuchsia', 'silver', 'white']

    plt.clf ()
    for i in range (len (top_k)):
        top_1 = all_csv_data [all_csv_data ['top-k'] == top_k [i]]
        all_score_top_1 = top_1 ['score']
        mean_accum_top_1 = tls.calculate_mean_score (all_score_top_1)

        plt.plot (top_1 ['episode'], mean_accum_top_1, '.', color=color [i],
                  label='Top-' + str (top_k [i]))
        # plt.text (pos_x[i], pos_y+0.02, 'mean:', color=color [i],fontsize=7)
        plt.text (pos_x [i], pos_y,
                  str (np.round (mean_accum_top_1 [-1], 3)),
                  fontsize=8, color=color [i])
    plt.xlabel ('Episode', fontsize=10)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.ylabel ('Accumulative score', fontsize=10)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.title (title, fontsize=14)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.legend (ncol=4, loc='lower right')
    if window_size == 0:
        plt.savefig (path + 'accum_score_top_k.png', dpi=300)
    else:
        plt.savefig (path + str (window_size) + '_accum_score_top_k.png', dpi=300)
    plt.close ()
def plot_score_top_k(path, filename, title, window_size=0):
    all_csv_data = pd.read_csv (path + filename)
    top_k = [1, 5, 10, 15, 20, 25, 30]

    all_mean_score_top_k = []
    for i in range(len(top_k)):
        score_top_k = all_csv_data[all_csv_data['top-k'] == top_k[i]]
        mean_score_top_k = np.mean(score_top_k['score'])
        all_mean_score_top_k.append(mean_score_top_k)

    plt.clf()
    plt.plot(top_k, all_mean_score_top_k, 'o-', color='blue')
    for i in range(len(top_k)):
        plt.text(top_k[i], all_mean_score_top_k[i] - 0.08, str(np.round(all_mean_score_top_k[i], 3)))
    plt.xlabel('Top-K', fontsize=16, font='Times New Roman')  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.ylabel('Acurácia', fontsize=16, font='Times New Roman')  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.xticks(top_k)
    plt.ylim([0, 1.1])
    plt.xlim([0, 35])
    plt.grid()
    #plt.title(title, fontsize=12)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    if window_size == 0:
        plt.savefig(path + 'score_top_k.png', dpi=300)
    else:
        plt.savefig(path + str(window_size)+'_score_top_k_without_title.png', dpi=300)
    plt.close()
    plt.clf()
def plot_score_top_1(path, filename, title, window_size=0):
    all_csv_data = pd.read_csv (path + filename)
    top_k = [1, 5, 10, 15, 20, 25, 30]
    color = ['blue', 'red', 'green', 'purple', 'orange', 'maroon',
             'teal']  # 'maroon', 'teal', 'black', 'gray', 'brown', 'cyan', 'magenta', 'yellow', 'olive', 'navy', 'lime', 'aqua', 'fuchsia', 'silver', 'white']

    top_1 = all_csv_data [all_csv_data ['top-k'] == 1]
    all_score_top_1 = top_1 ['score']

    plt.clf ()
    plt.plot (top_1 ['episode'], all_score_top_1, '.', color=color [0],
              label='Top-' + str (top_k [0]))
    plt.text(1750, 0.8, str(np.round(np.mean(all_score_top_1), 3)),
                  fontsize=10, color=color[1],
                bbox=dict(facecolor='yellow', edgecolor='none'))
    plt.xlabel ('Episode', fontsize=10)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.ylabel ('Score', fontsize=10)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.title (title, fontsize=12)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.legend (ncol=4, loc='lower right')
    if window_size == 0:
        plt.savefig (path + 'score_top_1.png', dpi=300)
    else:
        plt.savefig (path + str(window_size)+'_score_top_1.png', dpi=300)
    plt.close()
def plot_histogram_of_trainning_time(path, filename, title, graph_type, window_size=0):
    all_csv_data = pd.read_csv (path + filename)
    if window_size == 0:
        path = path
    else:
        path = path + str(window_size) + '_'


    top_k = [1]#[1, 5, 10, 15, 20, 25, 30]


    for i in range(len(top_k)):
        top_1 = all_csv_data [all_csv_data ['top-k'] == top_k [i]]
        trainning_time = top_1 ['trainning_process_time'] * 1e-9

    if graph_type == 'ecdf':
        #sns.set_style ('whitegrid', {'axes.edgecolor': '.6', 'grid.color': '.6'})
        sns.ecdfplot(trainning_time, label='Top-' + str (top_k [i]))

        plt.title(title + ' \n Trainning Time with ECDF')
        plt.grid(True)
        plt.savefig(path + 'ecdf_of_trainning_time.png', dpi=300)
        plt.close()

    if graph_type == 'hist':
        fig, ax1 = plt.subplots ()
        sns.kdeplot(trainning_time, label='kde density')
        ax2 = ax1.twinx()
        plt.grid (True)
        plt.hist(trainning_time, bins=60, alpha=0.3, label='Top-' + str (top_k [i]))
        ax2.set_ylabel('Counts')
        ax1.set_xlabel('Trainning Time [s]')
        ax1.legend(loc='upper right')
        plt.title(title)
        plt.grid(True)
        #ax2.legend (loc='upper right')
        plt.savefig (path + 'histogram_of_trainning_time.png', dpi=300)
        plt.close()

    if graph_type == 'kde':
        sns.kdeplot(trainning_time, label='kde density', shade=True, alpha=0.3)
        plt.title(title)
        plt.show()
        plt.savefig(path + 'kde_of_trainning_time.png', dpi=300)
        plt.close()

    sns.reset_orig ()



input = 'coord'
window = 'fixed_window'#'fixed_window' #'jumpy_sliding_window' # 'sliding_window' or 'jumpy_sliding_window'
window_size = 0

path, servidor, filename, pos_x, pos_y, title, data = read_results_for_plot (input,
                                                                           window)

plot_score_top_k(path=path, filename=filename, title=title, window_size=0)

plot_accum_score_top_k(pos_x, pos_y, path, title, filename, window_size=0)
plot_score_top_k(path, filename, title, window_size)
plot_score_top_1(path, filename, title, window_size)
plot_histogram_of_trainning_time(path=path, filename=filename, title=title, graph_type='hist', window_size=window_size)
plot_histogram_of_trainning_time(path=path, filename=filename, title=title, graph_type='ecdf', window_size=window_size)



'''
plot.plot_compare_types_of_windows (input_name=input, ref='Batool')
plot.plot_compare_windows_size_in_window_sliding(input_name=input, ref='Batool')

plot.plot_time_process_vs_samples_online_learning(path=path, filename=filename, title=title, ref='batool', window_size=window_size, flag_fast_experiment=True)
plot.plot_time_process_online_learning(path=path, filename=filename,  title=title, window_size=window_size, window_type=type_of_window)
'''

