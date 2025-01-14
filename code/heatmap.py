import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import csv



def read_data_for_plot_head_map(input='coord'):

    path = '../results/score/Wisard/'+input+'/'
    resolution = [1,2,4,8,16,32,64,128,256,512]
    all_accuracy =[]
    for i in resolution:
        filename = 'accuracy_'+input+'_res_'+str(i)+'.csv'
        df = pd.read_csv (path + filename)

        data = df['accuracy'].to_numpy()
        all_accuracy.append(data)

    from matplotlib.font_manager import get_font_names

    print (get_font_names ())

    plot_headmap(all_data=all_accuracy,
                 x_axis_labels_vector=df['addres_size'].to_numpy(),
                 y_axis_labels_vector= resolution,
                 x_label='Tamanho da memoria',
                 y_label='Resolução',
                 path=path,
                 folder=input)
    a=0

def plot_headmap(all_data,
                 x_axis_labels_vector,
                 y_axis_labels_vector,
                 x_label,
                 y_label,
                 path,
                 folder):
    y_axis_labels = y_axis_labels_vector
    x_axis_labels = x_axis_labels_vector

    # plotting the heatmap
    plt.figure (figsize=(6, 4), dpi=200)

    annot_kws = {'fontsize': 10,
                 #'fontstyle': 'italic',
                 'color': "w",
                 "family":"Times New Roman"}

    hm = sns.heatmap (data=all_data,
                      annot=True,
                      annot_kws=annot_kws,
                      cmap="YlGnBu",
                      xticklabels=x_axis_labels,
                      yticklabels=y_axis_labels,
                      fmt='.3f',
                      )
    hm.set_xlabel (x_label, color='steelblue', size=14, font='Times New Roman')
    hm.set_ylabel (y_label, color='steelblue', size=14, font='Times New Roman')

    # displaying the plotted heatmap
    plt.subplots_adjust (right=1, left=0.09)
    plt.savefig (path + 'headmap_' + folder, dpi=300, bbox_inches='tight')
    plt.show ()


def read_data_for_coord_in_termometro():

    rota = '../results/accuracy/8x32/coord/all/'

    j = 1
    # all_info = [6, 12, 18,24,28,34,38,44,48,54,58,64]
    all_info = []
    resolucao = []
    for i in range (1, 8):
        file_name = 'acuracia_s008_train_s008_test_' + str (j) + '.csv'

        print (file_name)
        filename = rota + file_name
        name_1 = 'Tamanho_memoria_' + str (j)
        name_2 = 'Acuracia_' + str (j)
        name_3 = 'intervalo_conf_' + str (j)
        df = pd.read_csv (filename,
                          sep='\t',
                          names=[name_1, name_2, name_3])

        data = df [name_2].to_numpy ()
        all_info.append(data)

        resolucao.append(j)
        j = j * 2
        # all_info = pd.concat([all_info, pd], axis=1)

    # print(df_1)
    # print(df_2)

    print (all_info)
    size_memory = [6, 12, 18, 24, 28, 34, 38, 44, 48, 54, 58, 64]
    # resolucao = [12, 24, 3, 4]

    rota = '../results/accuracy/'


    plot_headmap (all_data=all_info,
                  x_axis_labels_vector = size_memory,
                  y_axis_labels_vector = resolucao,
                  x_label='Address Size',
                  y_label='Resolution',
                  path=rota,
                  folder='coord_in_termometro_s008_s008')
    # return all_accuracy, size_memory, resolucao, rota


def read_data_for_coord_in_termometro_iguais_com_decimal():
    rota = '../results/accuracy/8X32/coord_in_termometro_iguais_com_decimal/Combined/'
    file_name = 'acuracia_beam_selection_Combined_[8X32]_coord_in_termometro_iguais_com_decimal_1_ALL.csv'
    filename = rota + file_name

    '''
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)

        all_info = np.zeros([12, 2], dtype=object)

        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            cont = 0
            for row in reader:
                all_info[cont] = int(row['tamanho_memoria']), float(row['Acuracia'])
                cont += 1
    '''

    df_1 = pd.read_csv (filename,
                        sep='\t',
                        names=['Tamanho_memoria', 'Acuracia', 'intervalo_conf'])

    rota = '../results/accuracy/8X32/coord_in_termometro_iguais_com_decimal/Combined/'
    j = 1
    # all_info = [6, 12, 18,24,28,34,38,44,48,54,58,64]
    all_info = []
    resolucao = []
    for i in range (1, 6):
        file_name = 'acuracia_beam_selection_Combined_[8X32]_coord_in_termometro_iguais_com_decimal_' + str (
            j) + '_ALL.csv'

        print (file_name)
        filename = rota + file_name
        name_1 = 'Tamanho_memoria_' + str (j)
        name_2 = 'Acuracia_' + str (j)
        name_3 = 'intervalo_conf_' + str (j)
        df = pd.read_csv (filename,
                          sep='\t',
                          names=[name_1, name_2, name_3])

        data = df [name_2].to_numpy ()
        all_info.append (data)

        resolucao.append (j)
        j = j * 2
        # all_info = pd.concat([all_info, pd], axis=1)

    # print(df_1)
    # print(df_2)

    print (all_info)
    size_memory = [6, 12, 18, 24, 28, 34, 38, 44, 48, 54, 58, 64]
    # resolucao = [12, 24, 3, 4]

    plot_headmap (all_info, size_memory, resolucao, 'Tamanho da memoria', 'Resolução', rota,
                  'coord_in_termometro_iguais')
    # return all_accuracy, size_memory, resolucao, rota


def read_coord_in_Thermomether_x_y_unbalanced():
    rota = '../results/accuracy/8X32/coord_in_Thermomether_x_y_unbalanced/Combined/'
    j = 1
    # all_info = [6, 12, 18,24,28,34,38,44,48,54,58,64]
    all_info = []
    resolucao = []
    for i in range (1, 11):
        file_name = 'acuracia_beam_selection_Combined_[8X32]_coord_in_Thermomether_x_y_unbalanced_' + str (
            j) + '_ALL.csv'

        print (file_name)
        filename = rota + file_name
        name_1 = 'Tamanho_memoria_' + str (j)
        name_2 = 'Acuracia_' + str (j)
        name_3 = 'intervalo_conf_' + str (j)
        df = pd.read_csv (filename,
                          sep='\t',
                          names=[name_1, name_2, name_3])

        data = df [name_2].to_numpy ()
        all_info.append (data)

        resolucao.append (j)
        j = j * 2
        # all_info = pd.concat([all_info, pd], axis=1)

    # print(df_1)
    # print(df_2)

    print (all_info)
    size_memory = [6, 12, 18, 24, 28, 34, 38, 44, 48, 54, 58, 64]
    # resolucao = [12, 24, 3, 4]

    rota = '../results/accuracy/article/'
    plot_headmap (all_info, size_memory, resolucao, 'Address size', 'Resolution', rota,
                  'read_coord_in_Thermomether_x_y_unbalanced')


def read_coord_in_Thermomether_x_y_unbalanced_with_decimal_part():
    rota = '../results/accuracy/8X32/coord_in_Thermomether_x_y_unbalanced_with_decimal_part/Combined/'
    j = 1
    # all_info = [6, 12, 18,24,28,34,38,44,48,54,58,64]
    all_info = []
    resolucao = []
    for i in range (1, 7):
        file_name = 'acuracia_beam_selection_Combined_[8X32]_coord_in_Thermomether_x_y_unbalanced_with_decimal_part_' + str (
            j) + '_ALL.csv'

        print (file_name)
        filename = rota + file_name
        name_1 = 'Tamanho_memoria_' + str (j)
        name_2 = 'Acuracia_' + str (j)
        name_3 = 'intervalo_conf_' + str (j)
        df = pd.read_csv (filename,
                          sep='\t',
                          names=[name_1, name_2, name_3])

        data = df [name_2].to_numpy ()
        all_info.append (data)

        resolucao.append (j)
        j = j * 2
        # all_info = pd.concat([all_info, pd], axis=1)

    # print(df_1)
    # print(df_2)

    print (all_info)
    size_memory = [6, 12, 18, 24, 28, 34, 38, 44, 48, 54, 58, 64]
    # resolucao = [12, 24, 3, 4]

    plot_headmap (all_info, size_memory, resolucao, 'Tamanho da memoria', 'Resolução', rota,
                  'coord_in_Thermomether_x_y_unbalanced_with_decimal_part')

def read_data_lidar_2D_dilated():
    path = '../results/accuracy/8X32/lidar_2D_dilated/all/'
    name_1 = 'Tamanho_memoria'
    name_2 = 'Acuracia'
    name_3 = 'intervalo_conf'
    all_info = []
    for i in range(1,11):
        file = 'acuracia_s008_train_s009_test_lidar_2D_dilated_it_'+str(i)+'.csv'
        filename = path + file
        df = pd.read_csv (filename,
                          sep='\t',
                          names=[name_1, name_2, name_3])

        data = df[name_2].to_numpy ()
        all_info.append(data)

    size_memory = [24, 28,32,36,40,44,48,52,56,60,64]
    resolucao = [1,2,3,4,5,6,7,8,9,10]
    # resolucao = [12, 24, 3, 4]

    rota = '../results/accuracy/article/'
    plot_headmap ( all_data=all_info,
                   x_axis_labels_vector=size_memory,
                   y_axis_labels_vector=resolucao,
                   x_label='Address size',
                   y_label= 'iterations',
                   path=path,
                  folder='acuracy_wisard_with_lidar_2D_dilated')

    a=0


read_data_for_plot_head_map()