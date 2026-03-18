import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def read_results_conventional_evaluation_inverter_dataset(input_type, connection_type):
    path = '../../results/inverter_dataset/score/'
    path_batool =path + 'Batool/'+input_type +'/'+connection_type+'/'
    file_name ='accuracy_'+input_type+'.csv'
    data_batool = pd.read_csv(path_batool+file_name)
    batool = data_batool[data_batool['top-k']<=30]

    path_wisard = path + 'Wisard/top-k/'+input_type + '/' +connection_type+'/'
    if input_type =='coord':
        file_name = 'accuracy_coord.csv'#_res_8_ALL.csv'
    if input_type == 'lidar':
        file_name = 'accuracy_lidar.csv'#_ALL_thr_01.csv'
    if input_type == 'lidar_coord':
        file_name = 'accuracy_lidar_coord.csv'#_res_8_ALL_thr_01.csv'
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

def read_data_train_with_s009():
    input_type = ['coord', 'lidar', 'lidar_coord']
    ref_name = ['Batool', 'ruseckas', 'Wisard']
    all_dataframes = []

    for ref in ref_name:
        for i_type in input_type:
            try:
                df_temp = read_file_csv (i_type, ref)
                df_temp ['input_type'] = i_type
                df_temp ['ref_name'] = ref
                all_dataframes.append (df_temp)
            except FileNotFoundError:
                print (f"Warning: File not found for input_type='{i_type}', ref_name='{ref}'. Skipping.")
            except Exception as e:
                print (f"An error occurred for input_type='{i_type}', ref_name='{ref}': {e}")

    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat (all_dataframes, ignore_index=True)

    # Create a copy to avoid modifying the original combined_df directly
    df_unified = combined_df.copy ()

    return df_unified
def read_data_train_with_s008():
    input_type = ['coord', 'lidar', 'lidar_coord']
    ref_name = ['Batool', 'ruseckas', 'Wisard']
    all_dataframes = []

    for ref in ref_name:
        for i_type in input_type:
            try:
                df_temp = read_csv_file (i_type, ref)
                df_temp ['input_type'] = i_type
                df_temp ['ref_name'] = ref
                all_dataframes.append (df_temp)
            except FileNotFoundError:
                print (f"Warning: File not found for input_type='{i_type}', ref_name='{ref}'. Skipping.")
            except Exception as e:
                print (f"An error occurred for input_type='{i_type}', ref_name='{ref}': {e}")

    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat (all_dataframes, ignore_index=True)

    # Create a copy to avoid modifying the original combined_df directly
    df_unified = combined_df.copy ()

    df_unified.loc[df_unified['ref_name']== 'Wisard', 'score'] = df_unified.loc[df_unified['ref_name']== 'Wisard', 'Acurácia']
    # Initialize new columns with pandas missing value type (pd.NA) for better type inference
    df_unified ['Acurácia'] = pd.NA

    df_unified = df_unified.drop(columns =['Acurácia'])


    return df_unified
def read_csv_file(input_type, ref_name):
    connection_type = 'ALL'
    general_path = '../../results/score/'
    specific_path = ref_name+'/split_dataset_LOS_NLOS/'+input_type +'/'+connection_type+'/'
    top_k = 'top-k'
    score = 'score'
    if ref_name == 'Batool':
        ref_name = 'batool'

    elif ref_name == 'Wisard':

        specific_path = ref_name + '/split_dataset/' + connection_type +'/' + input_type + '/'
        ref_name = 'wisard'
        top_k = 'top_k'
        score = 'Acurácia'


    filename = input_type + '_results_top_k_' + ref_name + '_' + connection_type + '.csv'
    path = general_path + specific_path
    all_data = pd.read_csv (path + filename)
    data = all_data [all_data [top_k] == 1] [score].to_frame ()

    return data
def read_file_csv(input_type='coord', ref_name='Ruseckas'):
    connection_type = 'ALL'
    general_path ='../../results/'
    specific_path = 'inverter_dataset/score/'+ref_name+'/'+input_type +'/'+connection_type +'/'

    filename = 'accuracy_' + input_type + '.csv'

    if ref_name == 'Wisard':
        specific_path = 'inverter_dataset/score/'+ref_name+'/top-k/'+input_type +'/'+connection_type +'/'

    path = general_path + specific_path
    all_data = pd.read_csv (path + filename)  # , header=None, usecols=usecols, names=usecols)
    data = all_data[all_data['top-k']==1]['score'].to_frame()

    return data

def plot_generalization_test_with_all_data_top_1():

  input_types_to_plot = ['coord', 'lidar', 'lidar_coord']
  all_data_train_with_s008 = read_data_train_with_s008()
  all_data_train_with_s009 = read_data_train_with_s009()

  unique_ref_names = pd.concat([all_data_train_with_s008['ref_name'], all_data_train_with_s009['ref_name']]).unique()
  plt.rcParams.update ({'font.size': 18})
  plt.rcParams.update ({'font.weight': 'normal'})  # Explicitly set to normal
  plt.rcParams.update ({'axes.labelweight': 'normal'})  # Also set axis labels to normal
  plt.rcParams.update ({'axes.titleweight': 'normal'})  # And title to normal
  plt.rcParams.update ({'legend.title_fontsize': 14})
  plt.rcParams.update ({'legend.fontsize': 14})
  plt.rcParams.update ({'xtick.labelsize': 18})
  plt.rcParams.update ({'ytick.labelsize': 18})

  # Configure serif fonts, with Times New Roman as preferred, and fallbacks
  plt.rcParams ['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
  plt.rcParams ['font.family'] = 'serif'

  fig, axes = plt.subplots (1, 3, figsize=(20, 7), sharey=True)
  # fig.suptitle('Accuracy at Top-1 Generalization Test (s008 vs. s009)', fontsize=16)

  bar_width = 0.35  # Width of each bar
  r_indices = np.arange (len (unique_ref_names))  # X-axis positions for groups of bars

  # Define colors for s008 and s009 datasets
  s008_color = sns.color_palette ("PuBu") [3]  # A darker green
  s009_color = sns.color_palette ("YlOrRd") [0]  # A darker blue

  for i, input_type in enumerate (input_types_to_plot):
      ax = axes [i]

      # Filter data for the current input_type and specifically for top-k == 1
      filtered_s008_data = all_data_train_with_s008 [
          (all_data_train_with_s008 ['input_type'] == input_type) &
          (all_data_train_with_s008 ['score'] )
          ]
      filtered_s009_data = all_data_train_with_s009 [
          (all_data_train_with_s009 ['input_type'] == input_type) &
          (all_data_train_with_s009  ['score'] )
          ]

      # Prepare data for bar plotting for each ref_name
      s008_accuracies = []
      s009_accuracies = []
      # Ensure the order of accuracies matches the unique_ref_names for consistent plotting
      for ref in unique_ref_names:
          # Get accuracy for s008
          s008_acc = filtered_s008_data [filtered_s008_data ['ref_name'] == ref] ['score'].values
          s008_accuracies.append (s008_acc [0] if s008_acc.size > 0 else 0)

          # Get accuracy for s009
          s009_acc = filtered_s009_data [filtered_s009_data ['ref_name'] == ref] ['score'].values
          s009_accuracies.append (s009_acc [0] if s009_acc.size > 0 else 0)

      # Plotting the bars
      ax.bar (r_indices - bar_width / 2, s008_accuracies, bar_width, label='s008', color=s008_color)
      ax.bar (r_indices + bar_width / 2, s009_accuracies, bar_width, label='s009', color=s009_color)

      for i in range(len(s008_accuracies)):
          ax.text (r_indices [i] - bar_width / 2, s008_accuracies [i] + 0.001, f'{s008_accuracies [i]:.2f}', ha='center', va='bottom', fontsize=10)
          ax.text (r_indices [i] + bar_width / 2, s009_accuracies [i] + 0.001, f'{s009_accuracies [i]:.2f}', ha='center', va='bottom', fontsize=10)
      ax.set_title (f'Input Type: {input_type}')

      # ax.set_xlabel('Reference Name')
      if i == 0:  # Only set ylabel for the first subplot due to sharey=True
          ax.set_ylabel ('Accuracy')
          ax.legend (title='Dataset', loc='upper left', fontsize=10, title_fontsize=10)  # Adjust legend loc
      ax.set_xticks (r_indices + 0.2)
      ax.set_xticklabels (unique_ref_names, rotation=0, ha="right")  # Use unique_ref_names for labels
      ax.grid (axis='y', linestyle='--', alpha=0.6)  # Grid only on y-axis for bar plots
      #ax.legend (title='Dataset', loc='upper left', fontsize=10, title_fontsize=10)  # Adjust legend loc

      # Adjust layout to prevent suptitle overlap and provide enough space for rotated labels
  plt.subplots_adjust (left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.02, hspace=1)


  path_to_save = '../../results/results_article_LoS_NLoS/generalization_test/'
  plt.savefig (path_to_save+'accuracy_generalization_test_s008_vs_s009.png', dpi=300, bbox_inches='tight')
  plt.show ()


def read_LoS_NLoS_ALL_results():
    conn_type = ['ALL', 'LOS', 'NLOS']
    ref_name = ['Batool', 'ruseckas', 'wisard']
    train_s008 = [True, False]
    all_dataframes = []


    for ref in ref_name:
        for connection in conn_type:
            for train in train_s008:
                try:
                    df_temp = read_data_LoS_NLoS (connection_type=connection, ref_name=ref, train_s008=train)
                    df_temp ['connection_type'] = connection
                    df_temp ['ref_name'] = ref
                    if train:
                        df_temp['training_set'] = 's008'
                    else:
                        df_temp['training_set'] = 's009'
                    all_dataframes.append (df_temp)
                except FileNotFoundError:
                    print (f"Warning: File not found for input_type='{connection}', ref_name='{ref}'. Skipping.")
                except Exception as e:
                    print (f"An error occurred for input_type='{connection}', ref_name='{ref}': {e}")

    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat (all_dataframes, ignore_index=True)

    # Create a copy to avoid modifying the original combined_df directly
    df_unified = combined_df.copy ()

    cond = (
            (df_unified ['ref_name'] == 'wisard') &
            (df_unified ['training_set'] == 's008') &
            (df_unified ['Acurácia'].notna())
    )
    # move os valores de Acuracia para score
    df_unified.loc [cond, 'score'] = df_unified.loc [cond, 'Acurácia']
    # limpar a coluna auxiliar
    df_unified.loc [cond, 'Acurácia'] = np.nan

    # remover a coluna definitivamente
    df_unified = df_unified.drop (columns=['Acurácia'])

    return df_unified
def read_data_LoS_NLoS(ref_name, connection_type, train_s008):
    general_path = '../../results/results_article_LoS_NLoS/data_for_article/LoS_NLoS_tests/'

    top_k = 'top-k'
    score = 'score'
    if train_s008:
        specific_path = ref_name + '/lidar_coord/' + connection_type + '/'
        file_name = 'lidar_coord_results_top_k_' + ref_name + '_' + connection_type + '.csv'
        if ref_name == 'wisard':
            top_k = 'top_k'
            score = 'Acurácia'
    else:
        specific_path = ref_name + '/inverter_dataset/lidar_coord/' + connection_type + '/'
        file_name = 'accuracy_lidar_coord_'+connection_type+'.csv'

    path = general_path + specific_path
    all_data = pd.read_csv (path + file_name)
    data = all_data [all_data [top_k] == 1] [score].to_frame ()

    return data

def plot_LoS_NLoS_top_1_type_dataset_trainning(Ds):
    df = read_LoS_NLoS_ALL_results ()
    # estilo (opcional, ajuste se necessário)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # minhas_cores = ["#003f5c", "#008585", "#58508d"]
    minhas_cores = ["lightseagreen", "steelblue", "yellowgreen"]  # Exemplo de cores personalizadas

    sns.set_theme (style="white", palette=minhas_cores, rc=custom_params)
    plt.rcParams.update ({'font.size': 18})
    plt.rcParams.update ({'font.weight': 'normal'})
    plt.rcParams.update ({'axes.labelweight': 'normal'})
    plt.rcParams.update ({'axes.titleweight': 'normal'})
    plt.rcParams.update ({'legend.title_fontsize': 14})
    plt.rcParams.update ({'legend.fontsize': 14})
    plt.rcParams.update ({'xtick.labelsize': 12})
    plt.rcParams.update ({'ytick.labelsize': 14})

    plt.rcParams ['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams ['font.family'] = 'serif'
    plt.rcParams ['mathtext.fontset'] = 'custom'
    plt.rcParams ['mathtext.rm'] = 'Times New Roman'

    plt.figure (figsize=(8, 6))
    df_s008 = df [df ['training_set'] == Ds]
    name_map = {'Batool': 'B. Saleh', 'ruseckas': 'J. Ruseckas', 'wisard': 'J. Manjarres'}
    df_s008 ['ref_name'] = df_s008 ['ref_name'].map (name_map)

    sns.barplot (
        data=df_s008,
        x='connection_type',
        y='score',
        hue='ref_name',
        #ax=axes [0], width=0.4, linewidth=1,  # alpha=0.6,
        errorbar='sd'  # ou None se não quiser barras de erro
    )
    plt.legend ( title='Models',
                 title_fontsize=14,
                 loc='upper center',
                 frameon=False,
                 shadow=True,
                 ncol=3,
                 bbox_to_anchor=(0.5, 1.15),
                 borderaxespad=0
                 )
    plt.text (2, 0.6, r'$\mathcal{D}_s$ = '+ Ds,
           verticalalignment='center', horizontalalignment='center',
           color='black', alpha=0.8, fontsize=14,
           bbox=dict (boxstyle="round", fc="papayawhip",  pad=0.5))

    #sns.move_legend(a, "upper right", title='Reference', frameon=False, shadow=True, ncol=1, labels=labels)
    plt.xlabel ('Connection type')
    plt.ylabel ('Top-1 Accuracy ')
    plt.yticks(np.arange (0, 0.8, 0.1))
    plt.tick_params (labelbottom=True)
    plt.grid (False)
    plt.tight_layout (rect=[0, 0.05, 1, 1])
    plt.savefig ('../../results/results_article_LoS_NLoS/data_for_article/All_results_LoS_NLoS_top_1_Ds_'+Ds+'.png', dpi=300,
                 bbox_inches='tight')
    plt.show ()

def plot_LoS_NLoS_top_1_results():
        df = read_LoS_NLoS_ALL_results()
        # Aqui você pode criar um gráfico usando seaborn ou matplotlib para comparar os resultados de LOS e NLOS

        #s008_results = df_LoS_NLoS_results [df_LoS_NLoS_results ['training_set'] == 's008']
        #s009_results = df_LoS_NLoS_results [df_LoS_NLoS_results ['training_set'] == 's009']

        # estilo (opcional, ajuste se necessário)
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        #minhas_cores = ["#003f5c", "#008585", "#58508d"]
        minhas_cores = ["lightseagreen", "steelblue", "yellowgreen"]  # Exemplo de cores personalizadas

        sns.set_theme ( style="darkgrid", palette=minhas_cores, rc=custom_params)
        plt.rcParams.update ({'font.size': 18})
        plt.rcParams.update ({'font.weight': 'normal'})
        plt.rcParams.update ({'axes.labelweight': 'normal'})
        plt.rcParams.update ({'axes.titleweight': 'normal'})
        plt.rcParams.update ({'legend.title_fontsize': 14})
        plt.rcParams.update ({'legend.fontsize': 14})
        plt.rcParams.update ({'xtick.labelsize': 12})
        plt.rcParams.update ({'ytick.labelsize': 14})

        plt.rcParams ['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
        plt.rcParams ['font.family'] = 'serif'

        # cria a figura e os subplots
        fig, axes = plt.subplots (nrows=2, ncols=1, figsize=(8, 10), sharex=True, sharey=True)

        # -----------------------------
        # Subplot 1 — training_set s008
        # -----------------------------
        df_s008 = df [df ['training_set'] == 's008']

        sns.barplot (
            data=df_s008,
            x='connection_type',
            y='score',
            hue='ref_name',
            ax=axes [0], width=0.4,linewidth = 1, #alpha=0.6,
            errorbar='sd'  # ou None se não quiser barras de erro
        )

        #axes [0].set_title ('Training set: s008')
        axes [0].set_xlabel ('')
        axes [0].set_ylabel ('Accuracy Top-1', fontsize=14)
        axes [0].text (2, 0.63,
                       'Training set: s008',
                       verticalalignment='center',
                       horizontalalignment='center',
                       color='white', alpha=0.8, fontsize=14,
                       bbox=dict (boxstyle="round", fc="gray", pad=0.5)
                       )
        axes[0].tick_params (labelbottom=True)
        axes[0].grid (axis='y', linestyle='--', alpha=0.6)  # Grid only on y-axis for bar plots

        # -----------------------------
        # Subplot 2 — training_set s009
        # -----------------------------
        df_s009 = df[df['training_set'] == 's009']

        sns.barplot (
            data=df_s009,
            x='connection_type',
            y='score',
            hue='ref_name',
            ax=axes[1], width=0.4, linewidth = 1, #, alpha=0.7,
            errorbar='sd'

        )

        #axes [1].set_title ('Training set: s009')
        axes [1].set_xlabel ('Connection type', fontsize=14)
        axes [1].set_ylabel ('Accuracy Top-1', fontsize=14)
        axes [1].text (2, 0.55, 'Training set: s009',
                       verticalalignment='center', horizontalalignment='center',
                       color='white', alpha=0.8, fontsize=14,
                       bbox=dict(boxstyle="round", fc="gray", pad=0.5))

        # -----------------------------
        # Ajustes finais
        # -----------------------------
        # legenda única (remove a de baixo)
        axes[0].legend_.remove()
        labels = ['B. Saleh', 'J. Ruseckas', 'J. Manjarres']
        handles, _ = axes[1].get_legend_handles_labels ()
        axes[1].legend(handles=handles,
                        loc="upper right",
                        ncol=3,
                        labels=labels,
                        frameon=False,
                        shadow=True
        )
        axes [1].grid (axis='y', linestyle='--', alpha=0.6)  # Grid only on y-axis for bar plots

        plt.tight_layout (rect=[0, 0.05, 1, 1])
        plt.savefig ('../../results/results_article_LoS_NLoS/data_for_article/All_results_LoS_NLoS_top_1.png', dpi=300, bbox_inches='tight')
        plt.show ()
        a=0

def plot_CDE_metric():
    # Configuração de estilo acadêmico
    #plt.style.use ('seaborn-v0_8-whitegrid')
    #plt.rcParams.update ({'font.size': 12, 'font.family': 'serif'})

    plt.rcParams.update ({'font.size': 18})
    plt.rcParams.update ({'font.weight': 'normal'})
    plt.rcParams.update ({'axes.labelweight': 'normal'})
    plt.rcParams.update ({'axes.titleweight': 'normal'})
    plt.rcParams.update ({'legend.title_fontsize': 14})
    plt.rcParams.update ({'legend.fontsize': 14})
    plt.rcParams.update ({'xtick.labelsize': 12})
    plt.rcParams.update ({'ytick.labelsize': 14})

    plt.rcParams ['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams ['font.family'] = 'serif'

    # 1. Preparação dos Dados
    # Delta H calculado (valor absoluto da diferença entre as entropias fornecidas)
    # ALL: |0.625 - 0.553| = 0.072
    # LOS: |0.556 - 0.575| = 0.019
    # NLOS: |0.685 - 0.514| = 0.171

    data = {
    'Experiment':['D_S', 'D_S', 'D_S', 'D_T', 'D_T', 'D_T'],
    'CDE_ratio': [36.07, 67.21, 15.03, 34.05, 4.00, 72.78],
    'Accuracy': [60.25, 68.90, 51.47, 50.66, 28.60, 20.00],
    'Delta_H': [0.072, 0.019, 0.171, 0.072, 0.019, 0.171],
    'Link_Type':['ALL', 'LOS', 'NLOS', 'ALL', 'LOS', 'NLOS']
    }

    info_D_S = {'Accuracy': [60.25, 68.90, 51.47],
                'CDE_ratio': [36.07, 67.21, 15.03],
                'Delta_H': [0.072, 0.019, 0.171],
                'Link_Type': ['ALL', 'LOS', 'NLOS']}

    info_D_T = {'Accuracy': [50.66, 28.60, 20.00],
                'CDE_ratio': [34.05, 4.00, 72.78],
                'Delta_H': [0.072, 0.019, 0.171],
                'Link_Type': ['ALL', 'LOS', 'NLOS']}

    info_df_D_Sdf = pd.DataFrame (data)
    info_df_D_S = pd.DataFrame (info_D_S)
    info_df_D_T = pd.DataFrame (info_D_T)


    # 2. Criação da Figura
    fig, ax = plt.subplots (figsize=(10, 7))

    # Definindo cores para cada tipo de link
    colors = {'ALL': '#3498db', 'LOS': '#2ecc71', 'NLOS': '#e74c3c'}
    col = ['#3498db','#2ecc71','#e74c3c']

    ax.scatter (x=info_df_D_S ['CDE_ratio'], y=info_df_D_S ['Accuracy'], s=info_df_D_S['Delta_H']*5000,
                color=col,  label=info_df_D_S['Link_Type'])  # Pontos de dados básicos
    ax.scatter (x=info_df_D_T ['CDE_ratio'], y=info_df_D_T ['Accuracy'], s=info_df_D_T['Delta_H']*5000,
                color=col, alpha=0.5, label=info_df_D_T['Link_Type'])  # Pontos de dados básicos

    for i, txt in enumerate (info_df_D_S['Link_Type']):
        ax.annotate (txt, (info_df_D_S['CDE_ratio'][i]+2, info_df_D_S['Accuracy'][i]),
                     xytext=(10, 5), textcoords='offset points')#, fontsize=9, fontweight='bold')
        ax.annotate ('$\mathcal{D}_S = s008$', (info_df_D_S ['CDE_ratio'][i]+2, info_df_D_S ['Accuracy'] [i]-3),
                     xytext=(10, 5), textcoords='offset points', fontsize=12 )#, fontweight='bold')

    for i, txt in enumerate (info_df_D_T['Link_Type']):
        ax.annotate (txt, (info_df_D_T['CDE_ratio'][i]+2, info_df_D_T['Accuracy'][i]),
                     xytext=(10, 5), textcoords='offset points')#, fontsize=9, fontweight='bold')
        ax.annotate ('$\mathcal{D}_S = s009$', (info_df_D_T ['CDE_ratio'] [i] + 2, info_df_D_T ['Accuracy'] [i] - 3),
                     xytext=(10, 5), textcoords='offset points', fontsize=12)

    ax.set_xlabel ('Class Data Excess Metric ($CDE_{\mathbb{N}} / CDE_{\mathbb{B}}$)')
    ax.set_ylabel ('Top-1 Accuracy (%)')
    #ax.set_title ('Impact of Domain Mismatch on Beam Selection Performance', fontsize=15, pad=20)
    ax.set_ylim (0, 100)
    ax.set_xlim (0, 85)
    ax.grid (True, linestyle='--', alpha=0.5)
    #ax.yticks(np.arange (0, 100, 10))
    #ax.yticks(np.arange (0, 0.85, 10))

    # Adicionando uma linha de tendência (opcional para mostrar a correlação inversa)
    z = np.polyfit (info_df_D_T['CDE_ratio'], info_df_D_T ['Accuracy'], 1)
    p = np.poly1d (z)
    x_sorted = np.sort (info_df_D_T['CDE_ratio'])
    ax.plot (x_sorted, p (x_sorted),  color='orange',
             linestyle='dotted',  label='Trendline')

    z_1 = np.polyfit (info_df_D_S['CDE_ratio'], info_df_D_S ['Accuracy'], 1)
    p_1 = np.poly1d (z_1)
    x_sorted_1 = np.sort (info_df_D_S['CDE_ratio'])
    ax.plot (x_sorted_1, p_1 (x_sorted_1), color='orange',
             linestyle='dotted',  label='Trendline')

    # Adicionando texto explicativo sobre o tamanho das bolhas
    ax.text (65, 5, 'Bubble Size $\propto |\Delta H|$', fontsize=12,
             style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})


    plt.tight_layout ()
    path='../../results/results_article_LoS_NLoS/data_for_article/'
    plt.savefig (path+'accuracy_vs_divergence.png', dpi=300)
    plt.show ()


plot_CDE_metric()
#plot_results_inverter_dataset()
#read_data_train_with_s009()
#plot_generalization_test_with_all_data_top_1()

#df_temp = read_data_LoS_NLoS (connection_type='ALL', ref_name='ruseckas', train_s008=False)
#plot_LoS_NLoS_top_1_type_dataset_trainning(Ds='s008')
#plot_LoS_NLoS_top_1_results()