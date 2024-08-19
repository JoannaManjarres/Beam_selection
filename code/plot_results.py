import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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
    plot_results_without_variance(Thresh, Acc, dataset_size, title, path, name_result)


    return Thresh, Acc
def plot_results_without_variance(threshold, acc, dataset_size, title, path, name_result):

    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    y_pos = np.arange(len(threshold))

    accuracy = acc*100 # --> Percentual

    ax.barh (y_pos, accuracy,  align='center')
    for i, v in enumerate (accuracy):
        ax.text(v - 35, i , 'Dataset size = '+str(dataset_size[i]), color='white', fontsize=12)

    for i, v in enumerate (accuracy):
        ax.text(v + 0.5, i - 0.05, str(round(v,2)), color='red', fontsize=12, fontweight='bold')

    ax.set_yticks(y_pos, labels=threshold)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Accuracy [%]', fontsize=12)
    ax.set_ylabel('Threshold', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.show()

    fig.savefig(path+name_result, transparent=True, bbox_inches='tight', pad_inches=0.1)
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


'''
def test:
    plt.bar ([i - (2 * barWidth) for i in space], score_coord ['Accuracy'],
              color='salmon', label='coord')
    plt.bar ([i - barWidth for i in space], lidar_2D_Rx_term ['Accuracy'],
              color='teal', label='lidar 2D with rx term')
    plt.bar (space, lidar_2D_Rx_term_sv ['Accuracy'],
              color='teal', label='lidar 2D with rx term svar', alpha=0.5)
    plt.bar ([i + barWidth for i in space], lidar_2D_Rx_term_coord ['Accuracy'],
              color='darkred', label='lidar 2D with rx term and coord')
    plt.bar ([i + 2 * barWidth for i in space], lidar_2D_Rx_term_coord_sv ['Accuracy'],
              color='darkred', label='lidar 2D with rx term and coord svar', alpha=0.5)
    plt.xticks (ticks=space, labels=[str (i) for i in score_coord ['top_k']])
'''
#plot_results_lidar_with_coord_top_k()
#plot_of_bars()

