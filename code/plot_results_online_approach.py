import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        filename = 'all_results_'+window+'_window_top_k.csv'
    usecols = ["top-k", "score", "test_time", "samples_tested",	"episode", "trainning_process_time", "samples_trainning"]
    #print(path+specific_path + filename)
    data = pd.read_csv(path +specific_path+ filename)#, header=None, usecols=usecols, names=usecols)

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


window_type = 'fixed'




def read_trainning_times_models(input_type, window_type):

    wisard_data = read_csv_file(input_type, 'wisard', window_type)
    time_wisard = wisard_data[wisard_data['top-k']==1]['trainning_process_time']*1e-9

    ruseckas_data = read_csv_file(input_type, 'ruseckas', window_type)
    time_ruseckas = ruseckas_data[ruseckas_data['top-k']==1]['trainning_process_time']

    batool_data = read_csv_file(input_type, 'Batool', window_type)
    time_batool = batool_data[batool_data['top-k']==1]['trainning_process_time']*1e-9

    return time_wisard, time_ruseckas, time_batool

def plot_train_time_Douglas():
    input_types = ['coord', 'lidar', 'lidar_coord']
    models = ['Wisard', 'Ruseckas', 'Batool']

    time_wisard, time_ruseckas, time_batool = read_trainning_times_models(input_type='lidar_coord', window_type='sliding')

    dfs = []

    for it in input_types:
        for m, df in zip (models, read_trainning_times_models (it, 'sliding')):
            df = df.to_frame ()
            df ['model'] = m
            df ['input_type'] = it

            dfs.append (df)

    all_data = pd.concat (dfs)


    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.family': 'Times New Roman'})
    #plt.rcParams.update({'figure.autolayout': True})
    #plt.rcParams.update({'figure.figsize': (8, 6)})
    #plt.rcParams.update({'figure.dpi': 300})
    plt.rcParams.update({'axes.spines.top': False,
                         'axes.spines.right': False})
    colors =['lightskyblue', 'forestgreen', 'coral']

    z = sns.barplot (data=all_data, errorbar=('pi', 99), hue='model', y='trainning_process_time', x='input_type',
                     hue_order=sorted (all_data.model.unique ()))#, palette='Paired')#, )#'Paired')#PuBuGn')
    z.set_yscale ('log')
    z.set_ylabel ('Training Time (s)')
    z.set_xlabel ('Input Type')
    z.legend (title='Model', loc='upper center',
              frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.18))

    plt.savefig ('../results/score/compare_online_all_ref_sliding_window_trainning_time_Douglas.png', dpi=300, bbox_inches='tight')
    a=0



def read_accuracy_models(input_type, window_type):

    top_k = [1, 5, 10]
    wisard_data = read_csv_file(input_type, 'wisard', window_type)
    wisard_score_ = wisard_data[['score','top-k']]
    wisard_score = wisard_score_ [wisard_score_ ['top-k'].isin (top_k)]

    ruseckas_data = read_csv_file(input_type, 'ruseckas', window_type)
    ruseckas_score_ = ruseckas_data[['score','top-k']]
    ruseckas_score = ruseckas_score_ [ruseckas_score_ ['top-k'].isin (top_k)]

    batool_data = read_csv_file(input_type, 'Batool', window_type)
    batool_score_ = batool_data[['score','top-k']]
    batool_score = batool_score_ [batool_score_ ['top-k'].isin (top_k)]

    return  wisard_score, ruseckas_score, batool_score,



def plot_accuracys():
    window_types = ['fixed', 'sliding']#, 'incremental']
    input_types = ['coord', 'lidar', 'lidar_coord']
    #modelos = ['Wisard', 'Ruseckas', 'Batool']

    lista_de_dfs = []  # Lista para guardar todos os dataframes

    print ("Carregando dados para todas as 9 combinações...")

    # 💡 PASSO CHAVE 2: Loop aninhado para carregar TUDO
    for w_type in window_types:
        for i_type in input_types:
            print (f"  - Carregando: Window={w_type}, Input={i_type}")

            # Chamar sua função
            wisard_df, ruseckas_df, batool_df = read_accuracy_models (
                window_type=w_type,
                input_type=i_type
            )

            # Juntar os 3 modelos
            dfs = [wisard_df, ruseckas_df, batool_df]
            nomes_modelos = ['Wisard', 'Ruseckas', 'Batool']

            df_modelos_juntos = pd.concat (
                df.assign (modelo=nome) for df, nome in zip (dfs, nomes_modelos)
            )

            # 💡 PASSO CHAVE 3: Adicionar as colunas de contexto
            df_modelos_juntos ['window_type'] = w_type
            df_modelos_juntos ['input_type'] = i_type

            lista_de_dfs.append (df_modelos_juntos)

    # --- 3. Criar o DataFrame Final ---
    df_total_combinado = pd.concat (lista_de_dfs)

    valores_k_ordenados = sorted (df_total_combinado ['top-k'].unique ())

    #sns.set_palette ("pastel")

    # 💡 PASSO CHAVE 4: Usar 'row' E 'col'
    g = sns.catplot (
        data=df_total_combinado,
        x='top-k',
        y='score',
        hue='modelo',  # Comparação lado a lado (Modelos)
        col='input_type',  # Colunas da grade (Inputs)
        row='window_type',  # Linhas da grade (Windows)
        kind='box',
        order=valores_k_ordenados,
        height=4,  # Altura de CADA subplot
        aspect=1.2,  # Proporção (largura/altura) de CADA subplot
        # Define a ordem das linhas/colunas para ficar como pedimos
        row_order=window_types,
        col_order=input_types
        #color='lightblue'
    )

    #g.fig.suptitle ('Comparação de Modelos, Entradas e Janelas',
    #                fontsize=20, y=1.03)

    # Ajustar os rótulos dos eixos
    #g.set_axis_labels ('Top-k', 'Score')
    g.fig.supylabel ('', fontsize=18, x=0.035)
    g.set_axis_labels ('Top-k', '')
    g.set_titles (col_template="Input: {col_name}", row_template=" ")  # row_template=" " remove o título da linha

    # Ajustar os títulos de cada subplot (ex: "Window: fixed | Input: coord")
    #g.set_titles (row_template="Window: {row_name}", col_template="Input: {col_name}")

    # (Opcional) Definir o mesmo limite Y para todos os 9 gráficos
    # g.set(ylim=(0.4, 1.0))

    # Ajustar layout para evitar sobreposição
    plt.tight_layout ()
    plt.subplots_adjust (top=0.9)  # Deixa espaço para o título geral


    for i, row_name in enumerate (g.row_names):
        # Seleciona o primeiro subplot de cada linha (coluna 0)
        ax = g.axes [i, 0]
        # Define o y-label desse subplot para o nome da janela
        ax.set_ylabel (row_name, fontsize=12, labelpad=10, fontfamily="serif")

    # Ocultar o texto padrão das linhas (que o Seaborn coloca à direita)
    for ax in g.axes.flat:
        if ax.texts:
            ax.texts [0].set_visible (False)


    plt.savefig ('../results/score/compare_online_all_ref_all_windows_accuracys.png', dpi=300, bbox_inches='tight')
    plt.show ()

    a=0



def plot_compare_time_train_by_model():
    coord_fixed_wisard_time, coord_fixed_ruseckas_time, coord_fixed_batool_Time = read_trainning_times_models (
        input_type='coord', window_type='fixed')
    lidar_fixed_wisard, lidar_fixed_ruseckas, lidar_fixed_batool = read_trainning_times_models (input_type='lidar',
                                                                                                window_type='fixed')

    lidar_coord_wisard_time = read_csv_file(input_type='lidar_coord', ref_name='wisard', window='fixed')
    time_wisard_lidar_coord_fixed = lidar_coord_wisard_time [lidar_coord_wisard_time ['top-k'] == 1] ['trainning_process_time'] * 1e-9

    # lidar_coord_fixed_wisard, lidar_coord_fixed_ruseckas, lidar_coord_fixed_batool = read_trainning_times_models (
    #    input_type='lidar_coord', window_type='fixed')

    dict_1={'gps':coord_fixed_wisard_time.tolist(),
           'lidar': lidar_fixed_wisard.tolist(),
            'lidar_coord': time_wisard_lidar_coord_fixed.tolist()}

    dict_2={'gps':coord_fixed_batool_Time.tolist(),
            'lidar': lidar_fixed_batool.tolist()}

    dict_3={'gps':coord_fixed_ruseckas_time.tolist(),
            'lidar': lidar_fixed_ruseckas.tolist()}

    wisard_results_time = pd.DataFrame (dict_1)
    batool_results_time = pd.DataFrame (dict_2)
    ruseckas_results_time = pd.DataFrame (dict_3)

    # Sliding window

    lidar_coord_wisard_sliding = read_csv_file (input_type='lidar_coord', ref_name='wisard', window='sliding')
    time_wisard_lidar_coord_sliding = lidar_coord_wisard_sliding [lidar_coord_wisard_sliding ['top-k'] == 1] [
                                        'trainning_process_time'] * 1e-9
    coord_wisard_sliding = read_csv_file (input_type='coord', ref_name='wisard', window='sliding')
    time_wisard_coord_sliding = coord_wisard_sliding [coord_wisard_sliding ['top-k'] == 1] [
        'trainning_process_time'] * 1e-9
    lidar_wisard_sliding = read_csv_file (input_type='lidar', ref_name='wisard', window='sliding')
    time_wisard_lidar_sliding = lidar_wisard_sliding [lidar_wisard_sliding ['top-k'] == 1] [
        'trainning_process_time'] * 1e-9
    dict_sliding = {'gps': time_wisard_coord_sliding.tolist(),
                    'lidar': time_wisard_lidar_sliding.tolist(),
                    'lidar_coord': time_wisard_lidar_coord_sliding.tolist()}

    wisard_results_time_sliding = pd.DataFrame (dict_sliding)
    dict_means_wisard = {'gps': [time_wisard_coord_sliding.mean()],
                            'lidar': [time_wisard_lidar_sliding.mean()],
                            'lidar_coord': [time_wisard_lidar_coord_sliding.mean()]}

    coord_batool_sliding = read_csv_file (input_type='coord', ref_name='Batool', window='sliding')
    time_batool_coord_sliding = coord_batool_sliding [coord_batool_sliding ['top-k'] == 1] [
        'trainning_process_time'] * 1e-9
    lidar_batool_sliding = read_csv_file (input_type='lidar', ref_name='Batool', window='sliding')
    time_batool_lidar_sliding = lidar_batool_sliding [lidar_batool_sliding ['top-k'] == 1] [
        'trainning_process_time'] * 1e-9
    dict_sliding_batool = {'gps': time_batool_coord_sliding.tolist(),
                    'lidar': time_batool_lidar_sliding.tolist()}
    batool_results_time_sliding = pd.DataFrame (dict_sliding_batool)
    dict_means_batool = {'gps': [time_batool_coord_sliding.mean()],
                            'lidar': [time_batool_lidar_sliding.mean()]}

    coord_ruseckas_sliding = read_csv_file (input_type='coord', ref_name='ruseckas', window='sliding')
    time_ruseckas_coord_sliding = coord_ruseckas_sliding [coord_ruseckas_sliding ['top-k'] == 1] [
        'trainning_process_time']
    lidar_ruseckas_sliding = read_csv_file (input_type='lidar', ref_name='ruseckas', window='sliding')
    time_ruseckas_lidar_sliding = lidar_ruseckas_sliding [lidar_ruseckas_sliding ['top-k'] == 1] [
        'trainning_process_time']
    dict_sliding_ruseckas = {'gps': time_ruseckas_coord_sliding.tolist(),
                    'lidar': time_ruseckas_lidar_sliding.tolist()}
    ruseckas_results_time_sliding = pd.DataFrame (dict_sliding_ruseckas)

    dict_means_ruseckas = {'gps': [time_ruseckas_coord_sliding.mean()],
                            'lidar': [time_ruseckas_lidar_sliding.mean()]}
    ##########################################################################################





    # Supondo que você tenha 3 DataFrames diferentes
    #datasets = [wisard_results_time, batool_results_time, ruseckas_results_time, wisard_results_time_sliding]
    titles = ["WiSARD model", "Batool model", "Ruseckas model"]
    datasets = [ wisard_results_time_sliding, batool_results_time_sliding, ruseckas_results_time_sliding]
    means =[dict_means_wisard, dict_means_batool, dict_means_ruseckas]

    fig, axes = plt.subplots (1, 3, figsize=(12, 6), sharey=False)
    axes = axes.ravel ()
    plt.subplots_adjust (wspace=0.25, hspace=0.03)
    plt.rcParams.update ({"font.size": 14,  # tamanho da fonte
                          "font.family": "Times New Roman"  # tipo de fonte (ex: 'serif', 'sans-serif', 'monospace')
                          })

    size_font = 14

    for i, (df, title) in enumerate (zip (datasets, titles)):
        sns.boxplot (data=df, linewidth=.3, color='lightblue', fliersize=3,
            showmeans=True, meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "red", "markersize": "5", "linestyle": "--", "color": 'red'},
            meanline=True, showcaps=True,
            ax=axes[i]  # coloca no subplot correto
                    # notch=True

        )

        #plot in text the mean values
        axes[0].text(0.65, dict_means_wisard['gps'][0], f"{dict_means_wisard['gps'][0]:.2f}", horizontalalignment='center', color='red', fontweight='bold', fontsize=size_font)
        axes[0].text(1.65, dict_means_wisard['lidar'][0], f"{dict_means_wisard['lidar'][0]:.2f}", horizontalalignment='center', color='red', fontweight='bold', fontsize=size_font)
        axes[0].text(2.65, dict_means_wisard['lidar_coord'][0], f"{dict_means_wisard['lidar_coord'][0]:.2f}", horizontalalignment='center', color='red', fontweight='bold', fontsize=size_font)

        axes[1].text(0.65, dict_means_batool['gps'][0], f"{dict_means_batool['gps'][0]:.2f}", horizontalalignment='center', color='red', fontweight='bold', fontsize=size_font)
        axes[1].text(1.65, dict_means_batool['lidar'][0], f"{dict_means_batool['lidar'][0]:.2f}", horizontalalignment='center', color='red', fontweight='bold', fontsize=size_font)

        axes[2].text(0.65, dict_means_ruseckas['gps'][0], f"{dict_means_ruseckas['gps'][0]:.2f}", horizontalalignment='center', color='red', fontweight='bold', fontsize=size_font)
        axes[2].text(1.65, dict_means_ruseckas['lidar'][0], f"{dict_means_ruseckas['lidar'][0]:.2f}", horizontalalignment='center', color='red', fontweight='bold', fontsize=size_font)

        axes [i].set_title (title, fontfamily="serif", fontsize=16, fontweight='bold')
        axes [i].spines ['top'].set_visible (False)
        axes [i].spines ['right'].set_visible (False)
        #axes [i].set_yscale ('log')


        #axes [i].set_yscale ('log')
    axes[0].set_ylabel('Training Time (s)', fontfamily="serif", fontsize=size_font)
    axes[0].set_xticklabels (['GPS', 'LiDAR', 'LiDAR + GPS'], fontfamily="serif", fontsize=10)
    axes[1].set_xticklabels (['GPS', 'LiDAR'], fontfamily="serif", fontsize=10)
    axes[2].set_xticklabels (['GPS', 'LiDAR'], fontfamily="serif", fontsize=10)
    plt.tight_layout (rect=[0, 0, 1, 0.95])
    a=1
    plt.savefig ('../results/score/compare_online_all_ref_all_windows_trainning_time_by_model.png', dpi=300, bbox_inches='tight')
    a=1


def plot_compare_trainning_times_models_by_input():


    coord_fixed_wisard_time, coord_fixed_ruseckas_time, coord_fixed_batool_Time = read_trainning_times_models(input_type='coord', window_type='fixed')
    lidar_fixed_wisard, lidar_fixed_ruseckas, lidar_fixed_batool = read_trainning_times_models(input_type='lidar', window_type='fixed')
    #lidar_coord_fixed_wisard, lidar_coord_fixed_ruseckas, lidar_coord_fixed_batool = read_trainning_times_models (
    #    input_type='lidar_coord', window_type='fixed')
    dict = {'wisard': coord_fixed_wisard_time.tolist(), 'ruseckas': coord_fixed_ruseckas_time.tolist(), 'batool': coord_fixed_batool_Time.tolist()}
    coord_fixed_time = pd.DataFrame(dict)

    dict2 = {'wisard': lidar_fixed_wisard.tolist(), 'ruseckas': lidar_fixed_ruseckas.tolist(), 'batool': lidar_fixed_batool.tolist()}
    lidar_fixed_time = pd.DataFrame(dict2)



    # Supondo que você tenha 3 DataFrames diferentes
    datasets = [coord_fixed_time, lidar_fixed_time]
    titles = ["GPS data", "LiDAR data", "LiDAR + GPS data"]

    fig, axes = plt.subplots (1, 3, figsize=(12, 4), sharey=False)
    axes = axes.ravel ()
    plt.subplots_adjust (wspace=0.25, hspace=0.03)
    plt.rcParams.update ({"font.size": 14,  # tamanho da fonte
                          "font.family": "Times New Roman"  # tipo de fonte (ex: 'serif', 'sans-serif', 'monospace')
                          })

    for i, (df, title) in enumerate (zip (datasets, titles)):
        sns.boxplot (
            data=df,
            linewidth=.3,
            color='lightblue',
            fliersize=3,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": "5"
            },
            meanline=True,
            showcaps=True,
            ax=axes [i]  # coloca no subplot correto

        )
        axes [i].set_title (title)
        size_font=14
    axes[0].set_ylabel('Training Time (s)', fontfamily="serif", fontsize=size_font)
    axes[0].spines ['top'].set_visible (False)
    axes[0].spines ['right'].set_visible (False)
    axes [1].spines ['top'].set_visible (False)
    axes [1].spines ['right'].set_visible (False)
    axes [2].spines ['top'].set_visible (False)
    axes [2].spines ['right'].set_visible (False)
    plt.tight_layout (rect=[0, 0, 1, 0.95])
    plt.savefig ('../results/score/compare_online_all_ref_all_windows_trainning_time.png', dpi=300, bbox_inches='tight')
    a=0


def calculate_results(input_type, ref_name, window):

    results = read_csv_file (input_type=input_type,
                             ref_name=ref_name,
                             window=window)
    top_k = [1, 5, 10, 20, 30]
    all_score = []
    all_dp = []
    for k in top_k:
        score_top_k = results[results['top-k'] == k]['score'].mean()
        dp_top_k = results[results['top-k'] == k]['score'].std()
        all_score.append(round(score_top_k, 3))
        all_dp.append(round(dp_top_k, 4))


    training_time = results[results['top-k'] == 1]['trainning_process_time'].mean()*1e-9
    dp_time = results[results['top-k'] == 1]['trainning_process_time'].std()*1e-9
    dict_results = {'top-k': top_k, 'score': all_score, 'dp': all_dp, 'training_time (s)': training_time, 'dp_time (s)': dp_time}
    results_df = pd.DataFrame(dict_results)
    return results_df


def show_results_the_all_models():
    input_type = 'lidar'
    window_type = 'sliding'
    model = ['ruseckas']#, 'Batool'] #'ruseckas'

    for model in model:
        print ("Results of evaluation models with ")
        print (input_type + " data and " + window_type + " window")
        print (model + " model")
        results_coord_fixed_wisard_df = calculate_results (input_type=input_type,
                                                       ref_name=model,
                                                       window= window_type)
        print (results_coord_fixed_wisard_df)
        print('--------------------------------------------------')


    a=0


def plot_compare_accuracy_models():
    coord_incremental_wisard = read_coord_results_ref (window='incremental', input='coord')
    lidar_incremental_wisard = read_coord_results_ref (window='incremental', input='lidar')
    lidar_coord_incremental_wisard = read_coord_results_ref (window='incremental', input='lidar_coord')
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
    window_type = 'incremental'
    data_type = 'coord'
    path = '../results/score/ruseckas/online/top_k/'+data_type+'/'+window_type+'_window/'
    filename1 = 'all_results_'+window_type+'_window_top_k_1.csv'
    filename2 = 'all_results_'+window_type+'_window_top_k_2.csv'
    #filename3 = 'all_results_fixed_window_top_k_parte_3.csv'

    usecols = ["top-k", "score", "test_time", "samples_tested",	"episode", "trainning_process_time", "samples_trainning"]
    data1 = pd.read_csv(path + filename1)#, header=None, usecols=usecols, names=usecols)
    data2 = pd.read_csv(path + filename2)#, header=None, usecols=usecols, names=usecols)
    #data3 = pd.read_csv(path + filename3)#, header=None, usecols=usecols, names=usecols)

    frames = [data1, data2]#, data3]
    result = pd.concat(frames)

    result.to_csv(path + 'all_results_fixed_window_top_k.csv', index=False, header=True)


def read_file_csv(input_type, ref_name, window):

    data_path ='/content/drive/MyDrive/Doutorado/results_online_learning/'
    data_path = '../results/score/results_vm_LAND/'+ ref_name +'/'
    path = data_path #+ ref_name

    specific_path = input_type +'/'+ window + '_window/'
    filename = 'all_results_'+window+'_window_top_k.csv'

    if window == 'sliding':
      filename = 'all_results_'+window+'_window_1000_top_k.csv'

    #if ref_name == 'Wisard':
    #    data_path = '../results/score/' + ref_name + '/online/servidor_vm_land/Wisard/'
    #    specific_path = input_type+'/'+window+'_window/'
    #    if window == 'sliding':
    #        filename = 'all_results_' + window + '_window_1000_top_k.csv'
    #    path = data_path #+ specific_path
    #    data = pd.read_csv (path + specific_path + filename)
    #else:
    usecols = ["top-k", "score", "test_time", "samples_tested",	"episode", "trainning_process_time", "samples_trainning"]
    data = pd.read_csv(path +specific_path+ filename)#, header=None, usecols=usecols, names=usecols)

    return data

def read_accuracy_models(input_type, window_type):
        top_k = [1, 5, 10]
        wisard_data = read_file_csv (input_type, 'Wisard', window_type)
        wisard_score_ = wisard_data [['score', 'top-k']]
        wisard_score = wisard_score_ [wisard_score_ ['top-k'].isin (top_k)]

        ruseckas_data = read_file_csv (input_type, 'ruseckas', window_type)
        ruseckas_score_ = ruseckas_data [['score', 'top-k']]
        ruseckas_score = ruseckas_score_ [ruseckas_score_ ['top-k'].isin (top_k)]

        batool_data = read_file_csv (input_type, 'Batool', window_type)
        batool_score_ = batool_data [['score', 'top-k']]
        batool_score = batool_score_ [batool_score_ ['top-k'].isin (top_k)]

        return wisard_score, ruseckas_score, batool_score,

def plot_accuracy_comparison_windows_inputs_ref():
  window_types = ['fixed', 'sliding']#, 'incremental']
  input_types = ['coord', 'lidar', 'lidar_coord']
  nomes_modelos = ['Wisard', 'Ruseckas', 'Batool']
  #modelos = ['Wisard', 'Ruseckas', 'Batool']

  lista_de_dfs = []

  data_incre_coord = read_accuracy_models(input_type='coord', window_type='incremental')
  data_incre_lidar = read_accuracy_models(input_type='lidar', window_type='incremental')

  df_wisard_incr_coord = data_incre_coord[0]
  df_ruseckas_incr_coord = data_incre_coord[1]
  df_batool_incr_coord = data_incre_coord[2]

  df_wisard_incre_lidar = data_incre_lidar[0]
  df_ruseckas_incr_lidar = data_incre_lidar[1]
  df_batool_incr_lidar = data_incre_lidar[2]

  df_wisard_incr_coord ['window_type'] = 'incremental'
  df_wisard_incr_coord ['input_type'] = 'coord'

  df_ruseckas_incr_coord ['window_type'] = 'incremental'
  df_ruseckas_incr_lidar ['input_type'] = 'lidar'

  df_batool_incr_coord ['window_type'] = 'incremental'
  df_batool_incr_coord ['input_type'] = 'coord'

  df_wisard_incre_lidar ['window_type'] = 'incremental'
  df_wisard_incre_lidar ['input_type'] = 'lidar'

  df_ruseckas_incr_lidar ['window_type'] = 'incremental'
  df_ruseckas_incr_lidar ['input_type'] = 'lidar'

  df_batool_incr_lidar ['window_type'] = 'incremental'
  df_batool_incr_lidar ['input_type'] = 'lidar'


  for w_type in window_types:
    for i_type in input_types:
      dfs = read_accuracy_models (window_type=w_type, input_type=i_type)

      df_modelos_juntos = pd.concat (df.assign (modelo=nome) for df, nome in zip (dfs, nomes_modelos))
      # Adicionar as colunas de contexto
      df_modelos_juntos ['window_type'] = w_type
      df_modelos_juntos ['input_type'] = i_type

      lista_de_dfs.append (df_modelos_juntos)


  # Cria o DataFrame Final
  df_total_combinado_1 = pd.concat (lista_de_dfs)

  df_total_combinado = pd.concat([df_total_combinado_1,
                    df_batool_incr_lidar, df_ruseckas_incr_lidar, df_wisard_incre_lidar,
                    df_batool_incr_coord, df_ruseckas_incr_coord, df_wisard_incr_coord])

  valores_k_ordenados = sorted (df_total_combinado ['top-k'].unique ())

  #sns.set_palette ("pastel")

  window_types = ['fixed', 'sliding', 'incremental']
  df_total_combinado.rename(
      inplace=True,
      columns={
          'score': 'top-k accuracy',
          'top-k': 'k',
          'modelo': 'model',
          'window_type': 'window type',
          'input_type': 'input type'
      }
  )

  g = sns.catplot (
      data=df_total_combinado,
      x='k',
      y='top-k accuracy',
      hue='model',  # Comparação lado a lado (Modelos)
      col='input type',  # Colunas da grade (Inputs)
      row='window type',  # Linhas da grade (Windows)
      kind='bar',
      order=valores_k_ordenados,
      height=4,  # Altura de CADA subplot
      aspect=1.2,  # Proporção (largura/altura) de CADA subplot
      # Define a ordem das linhas/colunas
      row_order=window_types,
      col_order=input_types,
      margin_titles=True,
      legend_out=False,
      errorbar=('ci', 99),
      #orientation='vertical'
      #color='lightblue'
  )

  sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
  )
  data_path = '../results/score/results_vm_LAND/'

  g.savefig (data_path +'compare_online_all_ref_all_windows_accuracys.png',
             bbox_inches='tight', dpi=300)

  #g.savefig('/content/drive/MyDrive/Doutorado/results_online_learning/accuracy_comparison_windows_ref_inputs.png', bbox_inches='tight', dpi=300)


plot_accuracy_comparison_windows_inputs_ref()
#merge_file_ruseckas_files()
#read_accuracy_models(input_type='lidar_coord', window_type='sliding')
#plot_accuracys()
#plot_train_time_Douglas()
#show_results_the_all_models()
#plot_compare_time_train_by_model()
#plot_compare_bars_accuracy_models()
#plot_compare_accuracy_models()







a=0