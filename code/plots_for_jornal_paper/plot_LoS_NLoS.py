import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def read_file_csv_LoS_NLoS(input_type, ref_name, type_connection, inverter_dataset):

    if inverter_dataset:
        data_path = '../../results/results_article_LoS_NLoS/data_for_article/LoS_NLoS_tests/'
        path = data_path + ref_name + '/inverter_dataset/' + input_type + '/' + type_connection + '/'
        filename = 'accuracy_'+input_type + '_'+type_connection + '.csv'

    else:
        data_path = '../../results/results_article_LoS_NLoS/data_for_article/LoS_NLoS_tests/'
        path = data_path + ref_name + '/' + input_type + '/' + type_connection + '/'
        filename = input_type + '_results_top_k_' + ref_name + '_' + type_connection + '.csv'

    data = pd.read_csv(path + filename)

    if ref_name == 'wisard':
      data.rename(columns={'top_k': 'top-k', 'Acurácia': 'score', 'Tamanho Treino':'train_size', 'Tamanho Teste':'test_size'}, inplace=True)
      df = data.iloc[:, 0:2].apply(pd.to_numeric, errors='coerce')

    else:
      df = data.iloc[:, 0:2].apply(pd.to_numeric, errors='coerce')

    return df

def plot_baseline_top_1_results():
    results_coord = {"ref_names":['Batool', 'Ruseckas','WiSARD'],
                           "score":[12.3, 60.3, 58.4],
                           "throughput":[0.19, 0.74, 0.72],
                           "trainning_time":[19.36,19.74,1.83],
                           "parameters":[2.55,  326.94, 4256],
                           "parameters_in_KB": [2550, 326.9, 0.53],
                           "parameters_unit":['MB', 'kB', 'bits']}

    df_coord_baseline = pd.DataFrame(results_coord)

    results_lidar = {"ref_names":['Batool', 'Ruseckas','WiSARD'],
                     "score":[46.2, 54.8, 58.2],
                     "throughput":[0.59, 0.70, 0.71],
                     "trainning_time":[1310.2, 2187.7, 0.8],
                     "parameters":[8.86, 3.93, 1888],
                     "parameters_in_KB":[8860, 3930, 0.24],
                     "parameters_unit":['MB', 'MB', 'bits']}

    df_lidar_baseline = pd.DataFrame(results_lidar)
    results_lidar_coord = {"ref_names":['Batool', 'Ruseckas','WiSARD'],
                           "score":[42.6, 58.8, 60],
                           "throughput":[0.56, 0.73, 0.74],
                           "trainning_time":[2301.6, 2180, 2.9],
                           "parameters":[18.4, 4.25, 6551],
                           "parameters_in_KB":[18400, 4250, 0.82],
                           "parameters_unit":['MB', 'MB', 'bits']}
    df_lidar_coord_baseline = pd.DataFrame(results_lidar_coord)

    plt.rcParams.update ({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 14,

        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,

        'figure.titlesize': 14
    })

    colors =['slategrey','lightblue','steelblue']
    fig, axs = plt.subplots (2, 2, figsize=(10, 8), gridspec_kw=dict(hspace=0.3, wspace=0.3) )
    x = np.arange(len(df_coord_baseline['ref_names']))
    axs[0, 0].bar(x - 0.2, df_coord_baseline['score'], width=0.2, label='Coord', color=colors[0])
    axs[0, 0].bar(x, df_lidar_baseline['score'], width=0.2, label='LiDAR', color=colors[1])
    axs[0, 0].bar(x + 0.2, df_lidar_coord_baseline['score'], width=0.2, label='LiDAR+Coord', color=colors[2])
    axs[0, 0].set_title('Score Top-1 ')
    axs[0, 0].set_xticks(x, df_lidar_baseline['ref_names'])
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    axs[0, 0].set_ylabel('Accuracy (%) ')
    axs[0 ,0].yaxis.grid (True)
    axs[0 ,0].spines[['right', 'top']].set_visible (False)

    axs[0, 1].bar(x - 0.2, df_coord_baseline['throughput'], width=0.2, color=colors[0])
    axs[0, 1].bar(x, df_lidar_baseline['throughput'], width=0.2,  color=colors[1])
    axs[0, 1].bar(x + 0.2, df_lidar_coord_baseline['throughput'], width=0.2,  color=colors[2])
    axs[0, 1].set_title('Throughput Top-1')
    axs[0, 1].set_xticks(x, df_lidar_baseline['ref_names'])
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    axs[0, 1].spines [['right', 'top']].set_visible (False)

    axs[1, 0].bar(x - 0.2, df_coord_baseline['trainning_time'], width=0.2,  color=colors[0])
    axs[1, 0].bar(x, df_lidar_baseline['trainning_time'], width=0.2,  color=colors[1])
    axs[1, 0].bar(x + 0.2, df_lidar_coord_baseline['trainning_time'], width=0.2,  color=colors[2])
    axs[1, 0].set_title('Training Time')
    axs[1, 0].set_xticks(x, df_lidar_baseline['ref_names'])
    #axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    axs[1, 0].set_ylabel('Time (s)')
    axs[1, 0].set_yscale('log')
    axs[1, 0].spines [['right', 'top']].set_visible (False)

    axs[1, 1].barh(x, df_coord_baseline['parameters_in_KB'], height=0.2,  color=colors[0])
    axs[1, 1].barh(x + 0.2, df_lidar_baseline['parameters_in_KB'], height=0.2,  color=colors[1])
    axs[1, 1].barh(x + 0.4, df_lidar_coord_baseline['parameters_in_KB'], height=0.2,  color=colors[2])
    axs[1, 1].set_yticks(x + 0.2, df_lidar_baseline['ref_names'])
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_xlim(0.01, 10000)
    axs[1, 1].set_title('Number of Parameters')
    axs[1, 1].spines[['left', 'right', 'top']].set_visible (False)
    for i in range(len(df_coord_baseline['ref_names'])):

        if i == 2:
            text_pos = 0.0001
        else:
            text_pos = 3
        axs[1, 1].text (df_coord_baseline['parameters_in_KB'][i]+text_pos, i - 0.05,
                        f"{df_coord_baseline['parameters'][i]} {df_coord_baseline['parameters_unit'][i]}",
                        fontsize=8)

        axs[1, 1].text (df_lidar_baseline['parameters_in_KB'][i]+text_pos, i + 0.15,
                        f"{df_lidar_baseline['parameters'][i]} {df_lidar_baseline['parameters_unit'][i]}",
                        fontsize=8)
        axs[1, 1].text (df_lidar_coord_baseline['parameters_in_KB'][i]+text_pos, i + 0.35,
                        f"{df_lidar_coord_baseline['parameters'][i]} {df_lidar_coord_baseline['parameters_unit'][i]}",
                        fontsize=8)

    fig.legend (loc='lower center', ncol=3, frameon=False)#, bbox_to_anchor=(0.5, 0.95))
    plt.tight_layout (rect=[0, 0.05, 1, 1])
    data_path = '../../results/results_article_LoS_NLoS/data_for_article/LoS_NLoS_tests/'
    plt.savefig(data_path+'baseline_top_1_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def read_ALL_LoS_NLoS_data_top_1(inverter_dataset):
  input_types = ['coord', 'lidar', 'lidar_coord']
  ref_names = ['Batool', 'ruseckas', 'wisard']
  type_connections = ['LOS', 'NLOS']
  all_dataframes = []


  for ref in ref_names:
    for i_type in input_types:
      for conn_type in type_connections:
        df_temp = read_file_csv_LoS_NLoS(i_type, ref, conn_type, inverter_dataset)
        df_temp['ref_name'] = ref
        df_temp['type_connection'] = conn_type
        df_temp['input_type'] = i_type
        all_dataframes.append(df_temp)

  combined_data = pd.concat (all_dataframes, ignore_index=True)
  #combined_data = combined_data.rename (columns={'':'', 'score': 'score'})

  return combined_data


def plot_LoS_NLoS_data_top_1():
    df_LoS_NLoS = read_ALL_LoS_NLoS_data_top_1 ()
    df_top_1 = df_LoS_NLoS [df_LoS_NLoS ['top-k'] == 1].copy ()

    input_types = ['coord', 'lidar', 'lidar_coord']
    type_connections = ['LOS', 'NLOS']
    ref_names = ['Batool', 'ruseckas', 'wisard']

    plt.rcParams.update ({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 14,

        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,

        'figure.titlesize': 14
    })


    fig, axes = plt.subplots (len (input_types),
                              len (type_connections),
                              figsize=(10, 8),
                              sharey=True)

    x_pos = [0,0.5,1]#np.arange(len(ref_names))-0.5
    width = 0.3


    for i, input_type in enumerate (input_types):
        for j, type_connection in enumerate (type_connections):
            ax = axes[i, j]
            for k, ref_name in enumerate (ref_names):
                df_plot = df_top_1.loc[(df_top_1['input_type'] == input_type) &
                                       (df_top_1['type_connection'] == type_connection) &
                                       (df_top_1['ref_name'] == ref_name)]

                #df_plot.plot(kind='bar', x=x_pos[k], y='score', ax=ax, label=ref_name, alpha=0.7)
                ax.bar (x_pos [k], df_plot ['score'].values[0], width=width,  alpha=0.7 )


            ax.set_xticks (x_pos)
            ax.set_xticklabels (ref_names, rotation=0)
            if j == 0:
                ax.set_ylabel (f'Input: {input_type}')
            if i ==0:
                ax.set_title (f'Connection: {type_connection}')

            ax.grid(True, linestyle='--', alpha=0.5)

    fig.text (0.055, 0.5, 'Accuracy Top-1', va='center', rotation='vertical')
    plt.tight_layout (rect=[0.08, 0.05, 1, 0.95])
    #plt.tight_layout (rect=[0.08, 0.15, 0.995, 0.95])
    plt.show()

def plot_():
    inverter_dataset = True
    df_LoS_NLoS = read_ALL_LoS_NLoS_data_top_1 (inverter_dataset)
    df_top_1 = df_LoS_NLoS [df_LoS_NLoS ['top-k'] == 1].copy ()

    input_types = ['coord', 'lidar', 'lidar_coord']
    type_connections = ['LOS', 'NLOS']
    ref_names = ['Batool', 'ruseckas', 'wisard']


    data_LOS = df_top_1 [df_top_1 ['type_connection'] == 'LOS']
    LOS_coord = data_LOS[data_LOS['input_type']=='coord']
    LOS_LiDAR = data_LOS[data_LOS['input_type']=='lidar']
    LOS_LiDAR_coord = data_LOS[data_LOS['input_type']=='lidar_coord']

    data_NLOS = df_top_1 [df_top_1 ['type_connection'] == 'NLOS']
    NLOS_coord = data_NLOS[data_NLOS['input_type']=='coord']
    NLOS_LiDAR = data_NLOS[data_NLOS['input_type']=='lidar']
    NLOS_LiDAR_coord = data_NLOS[data_NLOS['input_type']=='lidar_coord']



    plt.rcParams.update ({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 14,

        'axes.titlesize': 14,
        #'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,

        'figure.titlesize': 14
    })

    fig, axs = plt.subplots (1, 3, figsize=(8, 3),
                             gridspec_kw=dict (hspace=0.3, wspace=0.2), sharey=True)

    axs[0].set_title('Input: Coord')
    axs[1].set_title('Input: LiDAR')
    axs[2].set_title('Input: LiDAR + Coord')

    x_label = ['LoS', 'NLoS']
    colors = ['steelblue', 'green', 'darkmagenta']

    for i in range(len(ref_names)):
        axs[0].bar (i, LOS_coord [LOS_coord ['ref_name'] == ref_names[i]] ['score'],
                     alpha=0.3, color=colors[i], edgecolor=colors[i])
        axs[0].bar(i+4, NLOS_coord [NLOS_coord ['ref_name'] == ref_names[i]] ['score'], color=colors[i], label=ref_names[i], )
        axs[0].set_xticks ( [1,5], x_label)
        axs[0].grid (True, linestyle='--', alpha=0.5)
        axs[0].spines [['right', 'top']].set_visible (False)
        axs[0].set_ylabel ('Accuracy (%) ')

        axs[1].bar (i, LOS_LiDAR [LOS_LiDAR ['ref_name'] == ref_names[i]] ['score'],
                     alpha=0.3, color=colors[i], edgecolor=colors[i])
        axs[1].bar(i+4, NLOS_LiDAR [NLOS_LiDAR ['ref_name'] == ref_names[i]] ['score'],  color=colors[i])
        axs[1].set_xticks ( [1,5], x_label)
        axs[1].grid (True, linestyle='--', alpha=0.5)
        axs[1].spines [['right', 'top']].set_visible (False)

        axs[2].bar (i, LOS_LiDAR_coord [LOS_LiDAR_coord ['ref_name'] == ref_names[i]] ['score'],
                    alpha=0.3, color=colors[i], edgecolor=colors[i])
        axs[2].bar(i+4, NLOS_LiDAR_coord [NLOS_LiDAR_coord ['ref_name'] == ref_names[i]] ['score'],  color=colors[i])
        axs[2].set_xticks ( [1,5], x_label)
        axs[2].spines[['right', 'top']].set_visible (False)
        axs[2].grid (True, linestyle='--', alpha=0.5)

    #plt.legend (ref_names, loc='lower center', ncol=3, frame bbox_to_anchor=(1, 0.5))# bbox_to_anchor=(0.5, 0.95))
    #fig.legend (loc='center left', bbox_to_anchor=(1, 0), ncol=3, frameon=False)
    #plt.plot()

    names_for_legend = ['Batool', 'Ruseckas', 'WiSARD']
    legend_handles = [Patch (facecolor=colors [i], edgecolor=colors [i], label=names_for_legend [i])
        for i in range (len (names_for_legend))
    ]

    fig.subplots_adjust (bottom=0.32)

    fig.legend (
        handles=legend_handles,
        loc='lower center',
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02)
    )

    if inverter_dataset:
        data_path = '../../results/results_article_LoS_NLoS/data_for_article/LoS_NLoS_tests/'
        plt.savefig (data_path + 'inverter_dataset_LoS_NLoS_top_1_compartion.png', dpi=300, bbox_inches='tight')
        plt.show ()


    else:
        data_path = '../../results/results_article_LoS_NLoS/data_for_article/LoS_NLoS_tests/'
        plt.savefig (data_path + 'LoS_NLoS_top_1_compartion.png', dpi=300, bbox_inches='tight')
        plt.show ()


    a=0

#plot_()
plot_baseline_top_1_results()
#plot_LoS_NLoS_data_top_1()
v=0