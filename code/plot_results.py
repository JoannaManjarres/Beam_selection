import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import PercentFormatter


def results_lidar_2D_binary_without_variance():
    path = 'lidar_2D_without_variance/all/lidar_2D_with_rx_therm/'
    dataset_size = [6429, 2295, 1888, 1381, 697]
    #title = 'Evaluation of variance elimination of data \n Lidar 2D binary with Rx in thermometer without variance'
    title = 'Avaliacao da eliminacao de variancia dos dados \n Lidar 2D binario com Rx em termometro'
    name_result = 'lidar_2D_with_rx_therm_without_variance.png'
    name_of_file = 'acuracia_s008_train_s009_test_lidar_2D_without_variance_'
    read_and_plot_results_lidar_without_variance(path, title, dataset_size, name_result, name_of_file)
def results_lidar_3D_binary_without_variance():
    path= 'lidar_3D_without_variance/all/lidar_3D_with_rx_therm/'
    dataset_size = [28200, 7014, 5627, 3328, 1296]
    title = 'Avaliacao da eliminacao de variancia dos dados \n Lidar 3D binario com Rx em termometro'
    name_result = 'lidar_3D_with_rx_therm_without_variance.png'
    name_of_file = 'acuracia_s008_train_s009_test_lidar_3D_without_variance_'
    read_and_plot_results_lidar_without_variance(path, title, dataset_size, name_result,name_of_file)

def read_and_plot_results_lidar_without_variance(path, title, dataset_size, name_result,name_of_file):
    principal_path = '../results/accuracy/8x32/'
    path = principal_path + path
    usecols = ["Memory_size", "Accuracy", "Standar_deviation"]
    threshold = ['0','0,1', '0,15', '0,2', '0,24']
    scores_all = []
    for i in range(len(threshold)):
        file_name = name_of_file+str(threshold[i])+'.csv'
        data = pd.read_csv (path + file_name, header=None, sep="\t", names=usecols) #, names=["Memory_size", "Acuracy", "Standar_deviation"])
        scores_all.append([threshold[i],data["Accuracy"].max()])

    scores = pd.DataFrame(scores_all, columns=["Threshold", "Accuracy"])

    Thresh = scores['Threshold'].values
    Acc = scores['Accuracy'].values
    plot_results_without_variance(Thresh, Acc, dataset_size, title, principal_path, name_result)


    return Thresh, Acc
def plot_results_without_variance(threshold, acc, dataset_size, title, path, name_result):

    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    size_of_font = 16

    y_pos = np.arange(len(threshold))
    plot_accuracy = False
    accuracy = acc*100 # --> Percentual

    ax.barh (y_pos, accuracy,  align='center')
    for i, v in enumerate (accuracy):
        ax.text(v-10, i , str(dataset_size[i])+' *',
                color='white',
                fontsize=size_of_font,
                font='Times New Roman')
    ax.text(40,y_pos.min()-0.8, '* Tamanho da entrada')
    if plot_accuracy:
        for i, v in enumerate (accuracy):
            ax.text(v + 0.5, i - 0.05, str(round(v,2)),
                    color='red', fontsize=size_of_font,
                    fontweight='bold',
                    font='Times New Roman')

    ax.set_yticks(y_pos, labels=threshold)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Acurácia (%)', fontsize=size_of_font, font='Times New Roman')
    ax.set_ylabel('Limiar', fontsize=size_of_font, font='Times New Roman')
    #ax.set_title(title, fontsize=14, fontweight='bold')

    plt.show()

    fig.savefig(path+name_result, transparent=True, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)


def plot_results_lidar_with_coord():
    path = '../results/accuracy/8x32/lidar_+_coord/all/'

    usecols = ["Memory_size", "Accuracy", "Standar_deviation"]

    file_name = 'acuracia_coord.csv'
    score_coord = pd.read_csv(path+file_name, header=None, sep="\t", names=usecols)

    file_name = 'acuracia_LiDAR_2D_+_Rx_Term.csv'
    lidar_2D_rx_therm = pd.read_csv(path+file_name, header=None, sep="\t", names=usecols)

    file_name = 'acuracia_LiDAR_2D_+_Rx_Term_SVar.csv'
    lidar_2D_rx_therm_svar = pd.read_csv (path + file_name, header=None, sep="\t", names=usecols)

    file_name = 'acuracia_LiDAR_2D_+_Rx_Term_+_Coord_16.csv'
    lidar_2D_rx_therm_coord = pd.read_csv(path+file_name, header=None, sep="\t", names=usecols)

    file_name = 'acuracia_LiDAR_2D_+_Rx_Term_+_Coord_16_SVar.csv'
    lidar_2D_rx_therm_coord_svar = pd.read_csv(path+file_name, header=None, sep="\t", names=usecols)


    #-------------------------------------------
    #falta file_name = 'acuracia_LiDAR_3D_+_Rx_Cubo.csv'
    file_name = 'acuracia_LiDAR_3D_+_Rx_Cubo_+_Coord_16.csv'
    lidar_3D_rx_cubo_coord = pd.read_csv(path+file_name, header=None, sep="\t", names=usecols)

    file_name = 'acuracia_LiDAR_3D_+_Rx_Cubo_SVar_+_Coord_16.csv'
    lidar_3D_rx_cubo_svar_coord = pd.read_csv (path + file_name, header=None, sep="\t", names=usecols)

    # FAlta 'acuracia_LiDAR_3D_+_Rx_Term.csv'

    file_name = 'acuracia_LiDAR_3D_+_Rx_Term_SVar.csv'
    lidar_3D_rx_therm_svar = pd.read_csv (path + file_name, header=None, sep="\t", names=usecols)

    file_name = 'acuracia_LiDAR_3D_+_Rx_Term_+_Coord_16.csv'
    lidar_3D_rx_therm_coord_16 = pd.read_csv (path + file_name, header=None, sep="\t", names=usecols)

    file_name = 'acuracia_LiDAR_3D_+_Rx_Term_SVar_+_Coord.csv'
    lidar_3D_rx_therm_svar_coord = pd.read_csv(path+file_name, header=None, sep="\t", names=usecols)

    var = 1

    data = 'lidar_2D' #'lidar_3D'
    path = '../results/accuracy/8x32/lidar_+_coord/'
    if data == 'lidar_2D':
        plt.figure()
        plt.errorbar(score_coord['Memory_size'], score_coord['Accuracy'], yerr=score_coord['Standar_deviation'],
                    fmt='o-', label='coord', capsize=var, color='salmon') #linestyle= '',
        plt.errorbar(lidar_2D_rx_therm['Memory_size'], lidar_2D_rx_therm['Accuracy'], yerr=lidar_2D_rx_therm['Standar_deviation'],
                     linestyle='-', label='lidar 2D with rx therm', capsize=var, color='teal') #linestyle='--'
        plt.errorbar (lidar_2D_rx_therm_svar ['Memory_size'], lidar_2D_rx_therm_svar ['Accuracy'], yerr=lidar_2D_rx_therm_svar ['Standar_deviation'],
                      linestyle='--', label='lidar 2D with rx therm svar', capsize=var, color='teal')
        plt.errorbar(lidar_2D_rx_therm_coord['Memory_size'], lidar_2D_rx_therm_coord['Accuracy'], yerr=lidar_2D_rx_therm_coord['Standar_deviation'],
                     fmt='*-', label='lidar 2D with rx therm and coord', capsize=var, color='darkred') #
        plt.errorbar(lidar_2D_rx_therm_coord_svar['Memory_size'], lidar_2D_rx_therm_coord_svar['Accuracy'], yerr=lidar_2D_rx_therm_coord_svar['Standar_deviation'],
                     fmt='*--', label='lidar 2D with rx therm and coord svar', capsize=var, color='darkred') #



        plt.grid()
        plt.legend(loc="best")
        plt.title('Desemepenho da WiSARD usando \n LiDAR 2D, coordenadas e LiDAR 2D + Coordenadas')
        plt.xlabel("Tamanho da memoria")
        plt.ylabel("Acuracia")
        plt.legend(loc='lower right', fontsize='small')#, bbox_to_anchor=(0.5, -0.1)) # bbox_to_anchor=(0.5, -0.05),
                    #fancybox=True, shadow=True, ncol=1)
        plt.tight_layout()
        plt.xticks(ticks=score_coord['Memory_size'], labels=[str(i) for i in score_coord['Memory_size']])
        #plt.xticks ((lidar_2D_rx_therm_coord_svar['Memory_size']), labels=[str(i) for i in lidar_2D_rx_therm_coord_svar['Memory_size'])
        plt.savefig (path + data + '.png', dpi=300, bbox_inches='tight')
        plt.show()


    if data == 'lidar_3D':
        plt.figure()
        plt.errorbar(lidar_3D_rx_cubo_coord['Memory_size'], lidar_3D_rx_cubo_coord['Accuracy'], yerr=lidar_3D_rx_cubo_coord['Standar_deviation'],
                     fmt='go-', label='lidar 3D rx cubo coord', capsize=var, color='darkcyan') #
        plt.errorbar(lidar_3D_rx_cubo_svar_coord['Memory_size'], lidar_3D_rx_cubo_svar_coord['Accuracy'], yerr=lidar_3D_rx_cubo_svar_coord['Standar_deviation'],
                     fmt='go-', label='lidar 3D rx cubo svar coord', capsize=var, color='darkgreen') #
        plt.errorbar (lidar_3D_rx_therm_svar ['Memory_size'], lidar_3D_rx_therm_svar ['Accuracy'], yerr=lidar_3D_rx_therm_svar ['Standar_deviation'],
                      label='lidar 3D rx therm svar', linestyle='dashed', capsize=var, color='lightsalmon')
        plt.errorbar(lidar_3D_rx_therm_coord_16['Memory_size'], lidar_3D_rx_therm_coord_16['Accuracy'], yerr=lidar_3D_rx_therm_coord_16['Standar_deviation'],
                     label='lidar 3D rx therm coord', linestyle='dashed', capsize=var, color='red') #
        plt.errorbar(lidar_3D_rx_therm_svar_coord['Memory_size'], lidar_3D_rx_therm_svar_coord['Accuracy'], yerr=lidar_3D_rx_therm_svar_coord['Standar_deviation'],
                     label='lidar 3D rx therm svar coord', linestyle='dashed', capsize=var, color='darkred') #
        #
        plt.grid()
        plt.legend(loc="best")
        plt.title('Lidar 3D')
        plt.xlabel("Tamanho da memoria")

        plt.ylabel("Acuracia")
        plt.legend(loc='lower center', #bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=False, ncol=2, fontsize='small')
        #plt.yticks(np.arange(0, score_coord['Accuracy'].max(), step=0.1))
        plt.savefig (path + data + '.png', dpi=300, bbox_inches='tight')
        plt.show()





    a = 0

    #for i in range (len (threshold)):
    #    file_name = name_of_file + str (threshold [i]) + '.csv'
    #    data = pd.read_csv (path + file_name, header=None, sep="\t",
    #                        names=usecols)  # , names=["Memory_size", "Acuracy", "Standar_deviation"])
    #    scores_all.append ([threshold [i], data ["Accuracy"].max ()])

    #scores = pd.DataFrame (scores_all, columns=["Threshold", "Accuracy"])

def plot_results_lidar_with_coord_top_k():
    path = '../results/accuracy/8x32/lidar_+_coord/top_k/'

    usecols = ["top_k", "Accuracy"]

    name = 'acuracia_wisard_'
    name_1 ='_top_k.csv'

    file_name = 'coord'
    score_coord = pd.read_csv (path + name + file_name+name_1, names=usecols ,skiprows=1)

    file_name = 'LiDAR_2D_+_Rx_Term'
    lidar_2D_Rx_term = pd.read_csv (path + name + file_name + name_1, header=None, names=usecols, skiprows=1)

    file_name = 'LiDAR_2D_+_Rx_Term_Svar'
    lidar_2D_Rx_term_sv = pd.read_csv (path + name + file_name + name_1, header=None, names=usecols, skiprows=1)

    file_name = 'LiDAR_2D_+_Rx_Term_+_Coord'
    lidar_2D_Rx_term_coord = pd.read_csv (path + name + file_name + name_1, header=None, names=usecols, skiprows=1)

    file_name = 'LiDAR_2D_+_Rx_Term_+_Coord_16_Svar'
    lidar_2D_Rx_term_coord_sv = pd.read_csv (path + name + file_name + name_1, header=None, names=usecols, skiprows=1)

    plot_of_bars(score_coord, lidar_2D_Rx_term, lidar_2D_Rx_term_sv, lidar_2D_Rx_term_coord, lidar_2D_Rx_term_coord_sv, path)

    forma = 1

    space = np.arange (1, 49, 7)
    if forma ==1:

        barWidth = 1

        plt.figure(figsize=(5, 8))
        #plt.figure ()

        #fig = plt.figure()
        #ax = plt.subplots(111)

        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        #mpl.rcParams['figure.subplot.[bottom'] = 0.19

        plt.barh([i-(2*barWidth) for i in space], score_coord['Accuracy'], width=2,
                color='salmon', label='coord')
        plt.barh([i-barWidth for i in space], lidar_2D_Rx_term['Accuracy'],
                color='teal', label='lidar 2D with rx term')
        plt.barh(space, lidar_2D_Rx_term_sv['Accuracy'],
                color='teal', label='lidar 2D with rx term svar', alpha=0.5)
        plt.barh([i+barWidth for i in space], lidar_2D_Rx_term_coord['Accuracy'],
                color='darkred', label='lidar 2D with rx term and coord')
        plt.barh([i+2*barWidth for i in space], lidar_2D_Rx_term_coord_sv['Accuracy'],
                color='darkred', label='lidar 2D with rx term and coord svar', alpha=0.5)



        plt.yticks (ticks= space, labels=[str (i) for i in score_coord['top_k']])
        #plt.legend()
        plt.xlabel('Accuracy [%]', fontsize=12)
        plt.ylabel('Top-k', fontsize=12)
        plt.title('Desempenho da WiSARD TOP-K usando \n LiDAR 2D, coordenadas e LiDAR 2D + Coordenadas')
        plt.legend ( bbox_to_anchor=(-0.01, -0.26), loc="lower left", fontsize='small')


        for i, v in enumerate (score_coord['Accuracy']):
            plt.text(v+0.02, space[i]-(2.5*barWidth), str(round(v, 3)), color='salmon', fontsize=8)#, fontweight='bold')
        for i, v in enumerate (lidar_2D_Rx_term['Accuracy']):
            plt.text(v+0.02, space[i]-(1.5*barWidth), str(round(v, 3)), color='teal', fontsize=8)
        for i, v in enumerate (lidar_2D_Rx_term_sv['Accuracy']):
            plt.text(v+0.02, space[i]-(0.5*barWidth), str(round(v, 3)), color='teal', alpha=0.5, fontsize=8)
        for i, v in enumerate (lidar_2D_Rx_term_coord['Accuracy']):
            plt.text(v+0.02, space[i]+(0.5*barWidth), str(round(v, 3)), color='darkred', fontsize=8)
        for i, v in enumerate (lidar_2D_Rx_term_coord_sv['Accuracy']):
            plt.text(v+0.02, space[i]+(1.5*barWidth), str(round(v, 3)), color='darkred', alpha=0.5, fontsize=8)

        plt.subplots_adjust(bottom=0.19)
        plt.savefig (path + 'lidar_2D_top_k.png', dpi=300, bbox_inches='tight')
        plt.show ()


    else:
        fig, ax = plt.subplots ()
        ax.spines ['right'].set_visible (False)
        ax.spines ['top'].set_visible (False)


        y_pos = np.arange (len(space))

        accuracy = score_coord['Accuracy']* 100  # --> Percentual

        ax.barh (space, accuracy, align='center')
        #for i, v in enumerate (accuracy):
        #    ax.text (v - 35, i, 'Dataset size = ' + str (dataset_size [i]), color='white', fontsize=12)

        for i, v in enumerate (accuracy):
            ax.text (v + 0.5, i, str(round(v, 2)), color='red', fontsize=8)#, fontweight='bold')

        ax.set_yticks (space, labels=[str (i) for i in score_coord['top_k']])
        ax.invert_yaxis ()  # labels read top-to-bottom
        ax.set_xlabel ('Accuracy [%]', fontsize=12)
        ax.set_ylabel ('Top-k', fontsize=12)
        #ax.set_title (title, fontsize=14, fontweight='bold')
        plt.show()
        a=0



def plot_of_bars(data_1, data_2, data_3, data_4, data_5, path):

    a = data_1['Accuracy']
    b = data_2['Accuracy']
    c = data_3['Accuracy']
    d = data_4['Accuracy']
    e = data_5['Accuracy']

    mpl.rcParams ['axes.spines.right'] = False
    mpl.rcParams ['axes.spines.top'] = False

    # create plot
    fig, ax = plt.subplots(figsize=(5, 9))
    index = np.arange(0,98,14)
    bar_width = 2
    opacity = 1
    var = 0.2


    rects1 = plt.barh(index, a, bar_width,
                     alpha=opacity,
                     color='coral',
                     label='coord')

    rects2 = plt.barh(index+bar_width+var, b, bar_width,
                     alpha=opacity,
                     color='teal',
                      label='lidar 2D + rx term')

    rects3 = plt.barh (index+2*bar_width+2*var, c, bar_width,
                       alpha=0.5,
                       color='teal',
                       label='lidar 2D + rx term svar')

    rects4 = plt.barh (index + 3 * bar_width+3*var, d, bar_width,
                       alpha=opacity,
                       color='darkred',
                       label='lidar 2D + rx term \nand coord')

    rects5 = plt.barh (index + 4 * bar_width+4*var, e, bar_width,
                       alpha=0.5,
                       color='darkred',
                       label='lidar 2D + rx term \nand coord svar')

    for i, v in enumerate(a):
        plt.text(v+0.02, index[i]-1, str(round(v, 3)), color='salmon', fontsize=8)  # , fontweight='bold')
        #plt.text(v-0.5, index[i]-0.5, 'Dataset size = 4256' , color='white', fontsize=8)
    for i, v in enumerate(b):
        plt.text(v+0.02, index[i]+bar_width, str(round(v, 3)), color='teal', fontsize=8)
        #plt.text(v-0.5, index [i]+bar_width , 'Dataset size = 8.000', color='white', fontsize=8)
    for i, v in enumerate(c):
        plt.text(v+0.02, index[i]+2*bar_width, str(round (v, 3)), color='teal', alpha=0.5, fontsize=8)
        #plt.text(v-0.5, index[i]+2*bar_width, 'Dataset size = 1.888', color='white', fontsize=8)
    for i, v in enumerate(d):
        plt.text(v + 0.02, index[i]+3*bar_width, str(round (v, 3)), color='darkred', fontsize=8)
        #plt.text(v - 0.5, index[i]+3*bar_width, 'Dataset size = 12.256', color='white', fontsize=8)
    for i, v in enumerate(e):
        plt.text(v + 0.02, index[i]+4*bar_width, str(round(v, 3)), color='darkred', alpha=0.5, fontsize=8)
        #plt.text(v-0.5, index[i]+4*bar_width, 'Dataset size = 6.551', color='white', fontsize=8)

    plt.text(a[2] - 0.8, index[2]-0.5, 'Dataset size = 4.256 - Address Memory = 44', color='white', fontsize=8)
    plt.text(b[2] - 0.8, index[2]+ bar_width-0.2, 'Dataset size = 8.000 - Address Memory = 48', color='white', fontsize=8)
    plt.text(c[2] - 0.8, index[2]+ 2*bar_width, 'Dataset size = 1.888 - Address Memory = 24', color='white', fontsize=8)
    plt.text(d[2] - 0.8, index[2]+ 3*bar_width, 'Dataset size = 12.256 - Address Memory = 64', color='white', fontsize=8)
    plt.text(e[2] - 0.8, index[2]+ 4*bar_width, 'Dataset size = 6.551 - Address Memory = 44', color='white', fontsize=8)

    plt.xlabel('Acurácia [%]')
    plt.ylabel('Top-k')
    plt.title('Desempenho da WiSARD TOP-K usando \n LiDAR 2D, coordenadas e LiDAR 2D + Coordenadas' , fontsize=12)
    plt.yticks (index + bar_width, data_1['top_k'])
    #plt.legend(bbox_to_anchor=(0, -0.5),)
    plt.legend ( loc="lower right", fontsize=6)

    plt.tight_layout()
    plt.savefig (path + 'Top_k_lidar_2D.png', dpi=300, bbox_inches='tight')
    plt.show ()

def plot_lidar_powers_comparition(wisard, batool, ruseckas, mashhadi,
                                  label_wisard, label_batool, label_ruseckas, label_mashhadi,
                                  input, top_k, path, name_fig):
    #top_k =[1,5,10,20,30,40,50]

    type_of_marker = 'o'
    size_of_marker = 3
    color_wisard = 'red'
    color_batool = 'teal'
    color_ruseckas = 'blue'
    color_mashhadi = 'goldenrod'
    size_font = 8

    wisard = np.round(wisard, 3)
    batool = np.round(batool, 3)
    ruseckas = np.round(ruseckas, 3)
    mashhadi = np.round(mashhadi, 3)

    plt.plot(top_k, batool, label=label_batool, color=color_batool)
    for i, v in enumerate (batool):
        if top_k[i] == 1 or top_k[i] == 10 or top_k[i] == 50:
            plt.plot(top_k[i], v, marker=type_of_marker, markersize=size_of_marker, color=color_batool)
        if top_k[i] == 1:
            plt.text (top_k[i]+1, v+0.009, str (v), color=color_batool, size=size_font)
        if top_k[i] == 10:
            plt.text (top_k[i]+1, v-0.02, str (v), color=color_batool, size=size_font)
        if top_k[i] == 50:
            plt.text (top_k[i]+1, v-0.01, str (v), color=color_batool, size=size_font)

    plt.plot(top_k, ruseckas, label=label_ruseckas, color=color_ruseckas, alpha=0.5)
    for i, v in enumerate (ruseckas):
        if top_k[i] == 1 or top_k[i] == 10 or top_k[i] == 50:
            plt.plot(top_k[i], v, marker=type_of_marker, markersize=size_of_marker, color=color_ruseckas, alpha=0.5)
        if top_k [i] == 1:
            plt.text (top_k [i] + 1, v-0.02, str (v), color=color_ruseckas, size=size_font)
        if top_k [i] == 10:
            plt.text (top_k [i]+1, v+0.035, str (v), color=color_ruseckas, size=size_font)
        if top_k [i] == 50:
            plt.text (top_k [i]+1, v+0.01, str (v), color=color_ruseckas, size=size_font)

    plt.plot(top_k, mashhadi, label=label_mashhadi, color=color_mashhadi)
    for i, v in enumerate (mashhadi):
        if top_k[i] == 1 or top_k[i] == 10 or top_k[i] == 50:
            plt.plot(top_k[i], v, marker=type_of_marker, markersize=size_of_marker, color=color_mashhadi)
        if top_k[i] == 1:
            plt.text (top_k [i] + 1, v, str (v), color=color_mashhadi, size=size_font)
        if top_k[i] == 10:
            plt.text (top_k [i] + 1, v+0.025, str (v), color=color_mashhadi, size=size_font)
        if top_k[i] == 50:
            plt.text (top_k [i] + 1, v, str (v), color=color_mashhadi, size=size_font)

    plt.plot(top_k, wisard, label=label_wisard, color=color_wisard)
    for i, v in enumerate (wisard):
        if top_k[i] == 1 or top_k[i] == 10 or top_k[i] == 50:
            plt.plot(top_k[i], v, marker=type_of_marker, markersize=size_of_marker, color=color_wisard)
        if top_k[i] == 1:
            plt.text (top_k [i] + 1, v, str (v), color=color_wisard, size=size_font)
        if top_k[i] == 10:
            plt.text (top_k [i] + 1, v-0.01, str (v), color=color_wisard, size=size_font)
        if top_k[i] == 50:
            plt.text (top_k [i] + 1, v-0.02, str (v), color=color_wisard, size=size_font)

    plt.xlabel('top-k')
    plt.ylabel('Throughput Ratio')
    #plt.xticks(top_k)
    plt.title('Throughput Ratio of '+ input)
    plt.xlim([0, 55])
    plt.legend()
    plt.grid()
    #plt.show()

    plt.savefig(path + name_fig,  bbox_inches='tight', pad_inches=0.1, dpi=300) #transparent=True,
    plt.close()

def plot_powers_comparition(predicted_A, predicted_B, predicted_C,
                            label_A, label_B, label_C,
                            input, top_k, path, name_fig):
    #top_k =[1,5,10,20,30,40,50]

    type_of_marker ='o'
    size_of_marker = 3
    color_c = 'blue'
    color_b = 'teal'
    color_a = 'red'
    size_font = 8

    predicted_A = np.round(predicted_A, 3)
    predicted_B = np.round(predicted_B, 3)
    predicted_C = np.round(predicted_C, 3)


    plt.plot(top_k, predicted_A, label=label_A, color=color_a)
    for i, v in enumerate (predicted_A):
        if top_k[i] == 1 or top_k[i] == 10 or top_k[i] == 50:
            plt.plot(top_k[i], v, marker=type_of_marker, markersize=size_of_marker, color=color_a)
            plt.text (top_k [i] + 1, v-0.02, str (v), color=color_a, size=size_font)

    plt.plot(top_k, predicted_B, label=label_B, color=color_b)
    for i, v in enumerate (predicted_B):
        if top_k[i] == 1 or top_k[i] == 10 or top_k[i] == 50:
            plt.plot(top_k[i], v, marker=type_of_marker, markersize=size_of_marker, color=color_b)
            plt.text (top_k [i] + 1, v-0.02, str (v), color=color_b, size=size_font)

    plt.plot (top_k, predicted_C, label=label_C, color=color_c, alpha=0.5)
    for i, v in enumerate (predicted_C):
        if top_k[i] == 1 or top_k[i] == 10 or top_k[i] == 50:
            plt.plot(top_k[i], v, marker=type_of_marker, markersize=size_of_marker, color=color_c)
            plt.text (top_k [i] + 1, v+0.02, str (v), color=color_c, size=size_font)
    plt.xlabel('top-k')
    plt.ylabel('Throughput Ratio')
    #plt.xticks(top_k)
    plt.title('Throughput Ratio of '+ input)
    plt.legend()
    plt.grid()
    #plt.show()

    plt.savefig(path + name_fig,  bbox_inches='tight', pad_inches=0.1, dpi=300) #transparent=True,
    plt.close()

def plot_accum_score_top_k(pos_x, pos_y, path, title, filename, window_size=0):
    import tools as tls
    all_csv_data = pd.read_csv (path + filename)
    top_k = [1, 5, 10, 15, 20, 25, 30]
    color = ['blue', 'red', 'green', 'purple', 'orange', 'maroon',
             'teal']  # 'maroon', 'teal', 'black', 'gray', 'brown', 'cyan', 'magenta', 'yellow', 'olive', 'navy', 'lime', 'aqua', 'fuchsia', 'silver', 'white']

    plt.clf()
    for i in range (len (top_k)):
        top_1 = all_csv_data [all_csv_data ['top-k'] == top_k [i]]
        all_score_top_1 = top_1 ['score']
        mean_accum_top_1 = tls.calculate_mean_score (all_score_top_1)

        plt.plot (top_1 ['episode'], mean_accum_top_1, '.', color=color [i],
                  label='Top-' + str (top_k [i]))
        #plt.text (pos_x[i], pos_y+0.02, 'mean:', color=color [i],fontsize=7)
        plt.text (pos_x[i], pos_y,
                  str (np.round (mean_accum_top_1 [-1], 3)),
                  fontsize=8, color=color [i])
    plt.xlabel ('Episode', fontsize=10)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.ylabel ('Accumulative score', fontsize=10)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.title (title, fontsize=14)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.legend (ncol=4, loc='lower right')
    if window_size == 0:
        plt.savefig(path + 'accum_score_top_k.png', dpi=300)
    else:
        plt.savefig(path + str(window_size)+'_accum_score_top_k.png', dpi=300)
    plt.close ()

def plot_score_and_time_process_online_learning( pos_x, pos_y, path, title, filename):
    import tools as tls
    all_csv_data = pd.read_csv (path + filename)
    top_k = [1, 5, 10, 15, 20, 25, 30]
    color = ['blue', 'red', 'green', 'purple', 'orange', 'maroon',
             'teal']  # 'maroon', 'teal', 'black', 'gray', 'brown', 'cyan', 'magenta', 'yellow', 'olive', 'navy', 'lime', 'aqua', 'fuchsia', 'silver', 'white']

    for i in range (len (top_k)):
        top_1 = all_csv_data [all_csv_data ['top-k'] == top_k [i]]
        all_score_top_1 = top_1 ['score']
        mean_accum_top_1 = tls.calculate_mean_score (all_score_top_1)

        plt.plot (top_1 ['episode'], mean_accum_top_1, '.', color=color [i],
                  label='Top-' + str (top_k [i]))
        plt.text (pos_x [i], pos_y,
                  str (np.round (mean_accum_top_1[-1], 3)),
                  fontsize=8, color=color [i])
    plt.xlabel ('Episode', fontsize=10)#, fontweight='bold', fontname='Myanmar Sangam MN')
    plt.ylabel ('Accumulative score', fontsize=10)#, fontweight='bold', fontname='Myanmar Sangam MN')
    plt.title (title, fontsize=14)#, fontweight='bold', fontname='Myanmar Sangam MN')
    plt.legend (ncol=4, loc='lower right')
    plt.savefig (path + 'top-k_score.png', dpi=300)
    plt.close()

    top_1 = all_csv_data[all_csv_data ['top-k'] == 1]
    trainning_time = top_1['trainning_process_time'] * 1e-9
    mean_accum_time = tls.calculate_mean_score (trainning_time)
    plt.plot(top_1['episode'], trainning_time, marker=',', label='fixed window')
    plt.plot(top_1['episode'], mean_accum_time, marker='.', label='fixed window mean', color='red')
    plt.text(np.mean(top_1['episode']), np.mean(mean_accum_time)+2,
              'Mean: ' + str(np.round(np.mean(trainning_time), 3)),
             bbox={'facecolor': 'white',
                   'alpha': 0.8,
                   'pad': 0.1,
                   'edgecolor': 'white',
                   'boxstyle': 'round'},
              fontsize=10, color='red')
    plt.xlabel('Episode', fontsize=10)#, fontweight='bold', fontname='Arial')
    plt.ylabel('Trainning Time [s]', fontsize=10)#, fontweight='bold', fontname='Arial')
    plt.title(title, fontsize=12)#, fontweight='bold')#, family=['DejaVu Sans'])# fontname='Arial')
    plt.legend(loc='lower right', frameon=False)
    plt.savefig(path + 'trainning_time.png', dpi=300)
    plt.close()

def plot_time_process_online_learning( path, title, filename, window_size, window_type):
    import tools as tls
    all_csv_data = pd.read_csv(path + filename)

    if window_size == 0:
        path = path
    else:
        path = path + str(window_size) + '_'

    top_1 = all_csv_data[all_csv_data ['top-k'] == 1]
    trainning_time = top_1['trainning_process_time'] * 1e-9
    mean_accum_time = tls.calculate_mean_score (trainning_time)
    max_time = np.max(trainning_time)
    min_time = np.min(trainning_time)
    plt.plot(top_1['episode'], trainning_time, marker=',', label=window_type)
    plt.plot(top_1['episode'], mean_accum_time, marker='.', label=window_type+' mean', color='red')
    plt.text(np.mean(top_1['episode']), np.mean(mean_accum_time)+2,
              'Mean: ' + str(np.round(np.mean(trainning_time), 3)),
             bbox={'facecolor': 'white',
                   'alpha': 0.8,
                   'pad': 0.1,
                   'edgecolor': 'white',
                   'boxstyle': 'round'},
              fontsize=10, color='red')
    plt.text(1500, max_time-2, 'Max: ' + str(np.round(max_time, 3)), fontsize=10, color='red')
    plt.text(5, max_time-2, 'Min: ' + str(np.round(min_time, 3)), fontsize=10, color='red')
    plt.xlabel('Episode', fontsize=10)#, fontweight='bold', fontname='Arial')
    plt.ylabel('Trainning Time [s]', fontsize=10)#, fontweight='bold', fontname='Arial')
    plt.title(title, fontsize=12)#, fontweight='bold')#, family=['DejaVu Sans'])# fontname='Arial')
    plt.legend(loc='lower right', frameon=False)
    plt.savefig(path + 'trainning_time.png', dpi=300)
    plt.close()

def plot_time_process_vs_samples_online_learning( path, title, filename, ref, window_size=0, flag_fast_experiment=False):

    sns.set_theme (style="darkgrid")
    all_csv_data = pd.read_csv (path + filename)

    if flag_fast_experiment:
        data_time_with_valid_train_time = all_csv_data [all_csv_data ['trainning_process_time'] != 0]

        top_1 = data_time_with_valid_train_time [data_time_with_valid_train_time ['top-k'] == 1]
        trainning_time = top_1 ['trainning_process_time'] * 1e-9
    else:
        top_1 = all_csv_data [all_csv_data ['top-k'] == 1]
        trainning_time = top_1 ['trainning_process_time'] * 1e-9
    max_time = np.max (trainning_time)
    min_time = np.min (trainning_time)

    if ref == 'wisard':
        offset = 0
        print(ref)
    else:
        offset = 2

    fig, ax1 = plt.subplots(figsize=(15, 7))
    plt.plot(top_1['episode'], trainning_time)#, color='purple')
    plt.text(1750, max_time - offset, 'Max: ' + str(np.round(max_time, 3)) + ' seg', fontsize=10, color='red')
    plt.text(5, max_time - offset, 'Min: ' + str(np.round(min_time, 3)) + ' seg', fontsize=10, color='red')
    plt.text(np.mean(top_1['episode']), max_time - 2, 'Mean: ' + str(np.round(np.mean(trainning_time), 3)) + 'seg', fontsize=10, color='red')

    ax1.set_ylabel ('Trainning time [s]', fontsize=12, color='black', labelpad=10, fontweight='bold')
    ax1.set_xlabel ('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')

    # Criando um segundo eixo
    ax2 = ax1.twinx ()
    plt.plot(top_1['episode'], top_1['samples_trainning'], 'o', color='red')#, alpha=0.3)#, label='fixed window')

    ax2.set_ylabel ('Training samples', fontsize=12, color='black', labelpad=12, fontweight='bold')  # , color='red')

    # Adicionando título e legendas
    plt.title (title, fontsize=15, color='black', fontweight='bold')
    # plt.xticks(all_results_traditional['Episode'])
    plt.xlabel ('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')
    plt.legend (loc='best', ncol=3)  # loc=(0,-0.4), ncol=3)#loc='best')
    if window_size == 0:
        plt.savefig (path + 'time_and_samples_train_comparation.png', dpi=300)
    else:
        plt.savefig (path + str(window_size) + '_time_and_samples_train_comparation.png', dpi=300)
    plt.close ()

def plot_analyses_time_process_jumpy_sliding(path, filename, title):

    all_csv_data = pd.read_csv (path + filename)
    data_with_valid_train_time = all_csv_data [all_csv_data ['trainning_process_time'] != 0]

    top_1 = data_with_valid_train_time [all_csv_data ['top-k'] == 1]
    trainning_time = top_1 ['trainning_process_time'] * 1e-9
    max_time = np.max (trainning_time)
    min_time = np.min (trainning_time)
    offset = 0

    sns.set_theme (style="darkgrid")
    fig, ax1 = plt.subplots (figsize=(15, 7))
    plt.plot (top_1 ['episode'], trainning_time)  # , color='purple')
    plt.text (1750, max_time - offset, 'Max: ' + str (np.round (max_time, 3)) + ' seg', fontsize=10, color='red')
    plt.text (5, max_time - offset, 'Min: ' + str (np.round (min_time, 3)) + ' seg', fontsize=10, color='red')
    plt.text (np.mean (top_1 ['episode']), max_time - 2,
              'Mean: ' + str (np.round (np.mean (trainning_time), 3)) + 'seg', fontsize=10, color='red')

    ax1.set_ylabel ('Trainning time [s]', fontsize=12, color='black', labelpad=10, fontweight='bold')
    ax1.set_xlabel ('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')

    # Criando um segundo eixo
    ax2 = ax1.twinx ()
    plt.plot (top_1 ['episode'], top_1 ['samples_trainning'], 'o',
              color='red')  # , alpha=0.3)#, label='fixed window')

    ax2.set_ylabel ('Training samples', fontsize=12, color='black', labelpad=12,
                    fontweight='bold')  # , color='red')

    # Adicionando título e legendas
    plt.title (title, fontsize=15, color='black', fontweight='bold')
    # plt.xticks(all_results_traditional['Episode'])
    plt.xlabel ('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')
    plt.legend (loc='best', ncol=3)  # loc=(0,-0.4), ncol=3)#loc='best')
    plt.savefig (path + 'time_and_samples_train_comparation.png', dpi=300)
    plt.close ()

def plot_analyses_hist_of_jumpy_sliding(path, filename, graph_type, title):
    all_csv_data = pd.read_csv (path + filename)
    data_with_valid_train_time = all_csv_data [all_csv_data ['trainning_process_time'] != 0]

    top_1 = data_with_valid_train_time [all_csv_data ['top-k'] == 1]
    trainning_time = top_1 ['trainning_process_time'] * 1e-9
    top_k = [1]

    if graph_type == 'ecdf':
        #sns.set_style ('whitegrid', {'axes.edgecolor': '.6', 'grid.color': '.6'})
        sns.ecdfplot(trainning_time, label='Top-' + str (top_k[0]))

        plt.title(title + ' \n Trainning Time with ECDF')
        plt.grid(True)
        plt.savefig(path + 'ecdf_of_trainning_time.png', dpi=300)
        plt.close()


    if graph_type == 'hist':
        fig, ax1 = plt.subplots ()
        sns.kdeplot(trainning_time, label='kde density')
        ax2 = ax1.twinx()
        plt.grid (True)
        plt.hist(trainning_time, bins=60, alpha=0.3, label='Top-' + str (top_k [0]))
        ax2.set_ylabel('Counts')
        ax1.set_xlabel('Trainning Time [s]')
        ax1.legend(loc='upper right')
        plt.title(title)
        plt.grid(True)
        #ax2.legend (loc='upper right')
        plt.savefig (path + 'histogram_of_trainning_time.png', dpi=300)
        plt.close()
        #sns.reset_orig ()

    if graph_type == 'kde':
        sns.kdeplot(trainning_time, label='kde density', shade=True, alpha=0.3)
        plt.title(title)
        plt.grid()
        plt.savefig(path + 'kde_of_trainning_time.png', dpi=300)
        plt.close()
        #sns.reset_orig ()


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

def plot_histogram_of_trainning_time_Wisard(path, filename, title, graph_type, window_size):
    all_csv_data = pd.read_csv (path + filename)

    trainning_time = all_csv_data ['trainning Time top-1'] * 1e-9

    if graph_type == 'ecdf':
        #sns.set_style ('whitegrid', {'axes.edgecolor': '.6', 'grid.color': '.6'})
        sns.ecdfplot(trainning_time, label='Top-1')

        plt.title(title + ' \n Trainning Time with ECDF')
        plt.grid(True)
        plt.savefig(path + str(window_size)+'_ecdf_of_trainning_time.png', dpi=300)
        plt.close()

    if graph_type == 'hist':
        fig, ax1 = plt.subplots ()
        sns.kdeplot(trainning_time, label='kde density')
        ax2 = ax1.twinx()
        plt.grid (True)
        plt.hist(trainning_time, bins=60, alpha=0.3, label='Top-1')
        ax2.set_ylabel('Counts')
        ax1.set_xlabel('Trainning Time [s]')
        ax1.legend(loc='upper right')
        plt.title(title)
        plt.grid(True)
        #ax2.legend (loc='upper right')
        plt.savefig (path + str(window_size)+'_histogram_of_trainning_time.png', dpi=300)
        plt.close()

    if graph_type == 'kde':
        sns.kdeplot(trainning_time, label='kde density', shade=True, alpha=0.3)
        plt.title(title)
        plt.show()
        plt.savefig(path +str(window_size)+ '_kde_of_trainning_time.png', dpi=300)
        plt.close()
def plot_score_jumpy_online_learning( pos_x, pos_y, path, title, filename):
    import tools as tls
    all_csv_data = pd.read_csv(path + filename)
    top_k = [1, 5, 10, 15, 20, 25, 30]
    color = ['blue', 'red', 'green', 'purple', 'orange', 'maroon',
             'teal']  # 'maroon', 'teal', 'black', 'gray', 'brown', 'cyan', 'magenta', 'yellow', 'olive', 'navy', 'lime', 'aqua', 'fuchsia', 'silver', 'white']

    top_1 = all_csv_data [all_csv_data ['top-k'] == 1]
    all_score_top_1 = top_1 ['score']
    mean_accum_top_1 = tls.calculate_mean_score (all_score_top_1)

    plt.plot(top_1['episode'], mean_accum_top_1, '.', color=color[0],
                  label='Top-' + str(top_k[0]))
    #plt.text(pos_x[i], pos_y,
    #              str(np.round(np.mean(mean_accum_top_1), 3)),
    #              fontsize=8, color=color[i])
    plt.xlabel('Episode', fontsize=10)#, fontweight='bold', fontname='Myanmar Sangam MN')
    plt.ylabel('Accumulative score', fontsize=10)#, fontweight='bold', fontname='Myanmar Sangam MN')
    plt.title(title, fontsize=12)#, fontweight='bold', fontname='Myanmar Sangam MN')
    plt.legend(ncol=4, loc='lower right')
    plt.show()
    a = 0
    #plt.savefig(path + 'top-k_1_score.png', dpi=300)
    #plt.close()

    '''
    top_1 = all_csv_data[all_csv_data ['top-k'] == 1]
    trainning_time = top_1['trainning_process_time'] * 1e-9
    mean_accum_time = tls.calculate_mean_score (trainning_time)
    plt.plot(top_1['episode'], trainning_time, marker=',', label='fixed window')
    plt.plot(top_1['episode'], mean_accum_time, marker='.', label='fixed window mean', color='red')
    plt.text(np.mean(top_1['episode']), np.mean(mean_accum_time)+2,
              'Mean: ' + str(np.round(np.mean(trainning_time), 3)),
             bbox={'facecolor': 'white',
                   'alpha': 0.8,
                   'pad': 0.1,
                   'edgecolor': 'white',
                   'boxstyle': 'round'},
              fontsize=10, color='red')
    plt.xlabel('Episode', fontsize=10)#, fontweight='bold', fontname='Arial')
    plt.ylabel('Trainning Time [s]', fontsize=10)#, fontweight='bold', fontname='Arial')
    plt.title(title, fontsize=12)#, fontweight='bold')#, family=['DejaVu Sans'])# fontname='Arial')
    plt.legend(loc='lower right', frameon=False)
    plt.savefig(path + 'trainning_time_1.png', dpi=300)
    plt.close()
    '''

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



def get_scores_from_csv_results(data):
    all_mean_score_top_k = []
    top_k = [1, 5, 10, 15, 20, 25, 30]
    for i in range (len (top_k)):
        score_top_k = data [data ['top-k'] == top_k [i]]
        mean_score_top_k = np.mean (score_top_k ['score'])
        all_mean_score_top_k.append (mean_score_top_k)
    return all_mean_score_top_k, top_k

def plot_compare_windows_size_in_window_sliding(input_name, ref):
    window_type = '/sliding_window/'#window_size_var/'
    #path_result = '../../results/score/'+ref+'/online/results_server/top_k/'+input_name + window_type
    if ref == 'Wisard':
        path_result = '../results/score/' + ref + '/servidor_land/online/' + input_name + window_type
    else:
        path_result = '../../results/score/' + ref + '/servidor_land/online/' + input_name + window_type

    window_size = [100,  500, 1000, 1500, 2000]
    color = ['blue', 'red', 'green', 'purple', 'orange', 'maroon', 'teal', 'black', 'gray', 'brown', 'cyan', 'magenta', 'yellow', 'olive', 'navy', 'lime', 'aqua', 'fuchsia', 'silver', 'white']

    windows_score=[]
    for i in range(len(window_size)):
        file_name = '0_all_results_sliding_window_' + str(window_size[i]) + '_top_k.csv'
        all_csv_data = pd.read_csv (path_result+file_name)
        score, top_k = get_scores_from_csv_results(all_csv_data)
        windows_score.append(score)

    plt.clf ()
    for j in range(len(window_size)):

        plt.plot(top_k, windows_score[j], 'o-', label='Tamanho da Janela '+str(window_size[j]), color=color[j])
        score_round = np.round(windows_score[j], 3)

        #for i in range (len (score_round)):
        #    plt.text (top_k[i], score_round[i]-0.03, str(score_round [i]), color=color[j], fontsize=8)
    plt.legend(loc='best', ncol=2, fontsize=7)
    plt.xticks(top_k)
    plt.xlabel('k', font='Times New Roman', fontsize=16)
    plt.ylabel('Acurácia top-k', font='Times New Roman', fontsize=16)
    plt.grid()
    #plt.title('Beam selection using '+ref+' with '+ input_name + '\n in online learning with sliding window varying the window size')
    plt.savefig(path_result + 'score_comparation_window_size_without_title.png', dpi=300)
    plt.close()

def plot_compare_types_of_windows(input_name, ref):


    path = '../../results/score/' + ref + '/servidor_land/online/' + input_name + '/'


    if ref == 'Wisard':
        flag = '0_'

    window_type = 'fixed_window'
    path_result = path + window_type + '/'
    file_name = 'all_results_' + window_type + '_top_k.csv'
    if ref == 'Wisard':
        file_name = flag + file_name
    fixed_data = pd.read_csv(path_result + file_name)
    score_fixed_window, top_k = get_scores_from_csv_results (fixed_data)

    window_type = 'incremental_window'
    path_result = path + window_type + '/'
    file_name = 'all_results_' + window_type + '_top_k.csv'
    if ref == 'Wisard':
        file_name = flag + file_name
    incremental_data = pd.read_csv (path_result + file_name)
    score_incremental_window, top_k = get_scores_from_csv_results (incremental_data)

    window = 'sliding_window'
    path_result = path + window + '/'
    file_name = 'all_results_sliding_window_1000_top_k.csv'
    if ref == 'Wisard':
        file_name = flag + file_name
    sliding_data = pd.read_csv (path_result + file_name)
    score_sliding_window, top_k = get_scores_from_csv_results (sliding_data)


    color = ['blue', 'red', 'green', 'purple', 'orange', 'maroon', 'teal', 'black', 'gray', 'brown', 'cyan', 'magenta',
             'yellow', 'olive', 'navy', 'lime', 'aqua', 'fuchsia', 'silver', 'white']

    plt.figure()
    plt.clf()
    plt.plot(top_k, score_fixed_window, 'o-', label='fixed window', color=color[0])
    plt.plot(top_k, score_incremental_window, 'o-', label='incremental window', color=color[1])
    plt.plot(top_k, score_sliding_window, 'o-', label='sliding window 1000', color=color[2])

    for i in range(len(score_fixed_window)):
        plt.text(top_k[i]+1, score_fixed_window[i]-0.02,
                 str(np.round(score_fixed_window[i], 3)), fontsize=10, color=color[0])
        plt.text(top_k[i], score_incremental_window[i]+0.04,
                 str(np.round(score_incremental_window [i], 3)), fontsize=10, color=color[1])
        plt.text(top_k[i]+1, score_sliding_window[i]+0.02,
                 str(np.round(score_sliding_window [i], 3)), fontsize=10, color=color[2])

    plt.legend(loc='lower right')#, ncol=2, fontsize=7)
    plt.xticks(top_k)
    plt.yticks(np.arange(0.5, 1.1, 0.1))
    plt.grid()
    plt.title('Comparison of types of windows in online learning \n with '+ref+' and '+input_name)
    plt.xlabel('Top-k')
    plt.ylabel('score')
    plt.savefig(path + 'score_comparation_window_types.png', dpi=300)
    #plt.close()
    plt.show()


def plot_time_process_vs_samples_online_learning_wisard( path,
                                                         title,
                                                         filename,
                                                         ref,
                                                         type_window,
                                                         size_window):

    sns.set_theme (style="darkgrid")
    all_csv_data = pd.read_csv (path + filename)

    top_k = [1, 5, 10, 15, 20, 25, 30]
    trainning_time = all_csv_data['trainning Time top-1']* 1e-9
    max_time = np.max (trainning_time)
    min_time = np.min (trainning_time)

    if ref == 'wisard':
        offset = 0
        print(ref)
    else:
        offset = 2

    fig, ax1 = plt.subplots(figsize=(15, 7))
    plt.plot(all_csv_data['episode'], trainning_time)#, color='purple')
    plt.text(1750, max_time - offset, 'Max: ' + str(np.round(max_time, 3)) + ' seg', fontsize=10, color='red')
    plt.text(5, max_time - offset, 'Min: ' + str(np.round(min_time, 3)) + ' seg', fontsize=10, color='red')
    plt.text(np.mean(all_csv_data['episode']), max_time - 2, 'Mean: ' + str(np.round(np.mean(trainning_time), 3)) + 'seg', fontsize=10, color='red')

    ax1.set_ylabel ('Trainning time [s]', fontsize=12, color='black', labelpad=10, fontweight='bold')
    ax1.set_xlabel ('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')

    # Criando um segundo eixo
    ax2 = ax1.twinx ()
    plt.plot(all_csv_data['episode'], all_csv_data['samples train'], 'o', color='red')#, alpha=0.3)#, label='fixed window')

    ax2.set_ylabel ('Training samples', fontsize=12, color='black', labelpad=12, fontweight='bold')  # , color='red')

    # Adicionando título e legendas
    plt.title (title, fontsize=15, color='black', fontweight='bold')
    # plt.xticks(all_results_traditional['Episode'])
    plt.xlabel ('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')
    plt.legend (loc='best', ncol=3)  # loc=(0,-0.4), ncol=3)#loc='best')
    if type_window == 'sliding_window':
        plt.savefig (path + str(size_window)+'_time_and_samples_train_comparation.png', dpi=300)
    plt.savefig (path + 'time_and_samples_train_comparation.png', dpi=300)
    plt.close ()




#plot_results_lidar_with_coord_top_k()
#plot_of_bars()
#results_lidar_2D_binary_without_variance()
plot_compare_windows_size_in_window_sliding('coord', 'Wisard')
