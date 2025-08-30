import pandas as pd
import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt
import throughput as tp
from matplotlib.colors import Normalize

def plot_results_top_k_for_strategy(top_k, filename,
                                    accuracy_coord,
                                    accuracy_lidar,
                                    accuracy_lidar_coord,
                                    title):

    style_of_line_coord = 'dashed'  # 'solid'#
    type_of_marker_coord = 'x'
    size_of_marker_coord = 3
    width_of_line_coord = 1
    color_coord = 'blue'

    style_of_line_lidar = 'dashed'  # 'solid'#
    type_of_marker_lidar = 'x'
    size_of_marker_lidar = 3
    width_of_line_lidar = 1
    color_lidar = 'red'

    style_of_line_lidar_coord = 'solid'  # 'solid'#
    type_of_marker_lidar_coord = 'x'
    size_of_marker_lidar_coord = 3
    width_of_line_lidar_coord = 1
    color_lidar_coord = 'green'



    plt.plot (top_k, accuracy_coord, color=color_coord, marker=type_of_marker_coord, linestyle=style_of_line_coord,
              linewidth=width_of_line_coord, markersize=size_of_marker_coord, label='COORD')
    plt.plot (top_k, accuracy_lidar, color=color_lidar, marker=type_of_marker_lidar, linestyle=style_of_line_lidar,
                linewidth=width_of_line_lidar, markersize=size_of_marker_lidar, label='LIDAR')
    plt.plot (top_k, accuracy_lidar_coord, color=color_lidar_coord, marker=type_of_marker_lidar_coord, linestyle=style_of_line_lidar_coord,
                linewidth=width_of_line_lidar_coord, markersize=size_of_marker_lidar_coord, label='LIDAR + COORD')

    plt.title (title, color='steelblue', size=14, fontweight='bold')
    plt.xticks (top_k)
    plt.xlabel ('Top-k', color='steelblue', size=14, fontweight='bold')
    plt.ylabel ('Accuracy', color='steelblue', size=14, fontweight='bold')
    plt.legend ()
    plt.grid ()
    plt.savefig (filename, dpi=300, bbox_inches='tight')
    plt.show ()

def plot_LiDAR_score_rt_top_k(input, top_k, filename,
                        score_1, rt_1, label_1,
                        score_2, rt_2, label_2,
                        score_3, rt_3, label_3,
                        score_4, rt_4, label_4 ):

    plt.plot(top_k, score_1, color='blue', label=label_1, alpha=0.5)
    plt.plot(top_k, score_2, color='red', label=label_2)
    plt.plot(top_k, score_3, color='teal', label=label_3)
    plt.plot(top_k,  score_4, color='pink', label=label_4)

    label_complement = ' RT'
    plt.plot(top_k, rt_1, color='blue', label=label_1+label_complement, linestyle='dashed', alpha=0.5)
    plt.plot(top_k, rt_2, color='red', label=label_2+label_complement, linestyle='dashed')
    plt.plot(top_k, rt_3, color='teal', label=label_3+label_complement, linestyle='dashed')
    plt.plot (top_k, rt_4, color='pink', label=label_4 + label_complement, linestyle='dashed')

    plt.legend (loc='lower right')
    plt.title('Score and RT Top-K of the data '+input, color='steelblue', size=14, fontweight='bold')
    plt.xlabel('Top-k', color='steelblue', size=14, fontweight='bold')
    plt.ylabel('Score and RT', color='steelblue', size=14, fontweight='bold')
    plt.grid()
    plt.savefig (filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_score_rt_top_k(input, top_k, filename,
                        score_1, rt_1, label_1,
                        score_2, rt_2, label_2,
                        score_3, rt_3, label_3):

    plt.plot(top_k, score_1, color='blue', label=label_1, alpha=0.5)
    plt.plot(top_k, score_2, color='red', label=label_2)
    plt.plot(top_k, score_3, color='teal', label=label_3)

    label_complement = ' RT'
    plt.plot(top_k, rt_1, color='blue', label=label_1+label_complement, linestyle='dashed', alpha=0.5)
    plt.plot(top_k, rt_2, color='red', label=label_2+label_complement, linestyle='dashed')
    plt.plot(top_k, rt_3, color='teal', label=label_3+label_complement, linestyle='dashed')
    plt.legend (loc='lower right')
    plt.title('Score and RT Top-K of the data '+input, color='steelblue', size=14, fontweight='bold')
    plt.xlabel('Top-k', color='steelblue', size=14, fontweight='bold')
    plt.ylabel('Score and RT', color='steelblue', size=14, fontweight='bold')
    plt.grid()
    plt.savefig (filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_results_LiDAR_top_k(top_k, input, filename,
                             accuracy_ruseckas, accuracy_wisard, accuracy_batool, accuracy_mashhadi,
                             label_ruseckas, label_wisard, label_batool, label_mashhadi):

    color_wisard = 'red'
    color_batool = 'teal'
    color_ruseckas = 'blue'
    color_mashhadi ='goldenrod'

    style_of_line = 'solid'
    size_of_marker = 3
    type_of_marker = 'o'
    width_of_line = 1
    size_font = 8

    #sns.set()
    plt.figure()

    plt.plot(top_k, accuracy_ruseckas, color=color_ruseckas,
             linestyle=style_of_line, linewidth=width_of_line, label=label_ruseckas, alpha=0.5,)
    for i, v in enumerate (accuracy_ruseckas):
        if top_k[i] == 1 or top_k[i] == 10 or top_k[i] == 50:
            plt.plot(top_k[i], v, marker=type_of_marker, markersize=size_of_marker, color=color_ruseckas)
        if top_k[i] == 1:
            plt.text(top_k[i]+1, v, str(v), color=color_ruseckas, size=size_font)
        elif top_k[i] == 10:
            plt.text(top_k[i], v+0.03, str(v), color=color_ruseckas, size=size_font)
        elif top_k[i] == 50:
            plt.text(top_k[i]+1, v+0.015, str(v), color=color_ruseckas, size=size_font)

    plt.plot(top_k, accuracy_mashhadi, color=color_mashhadi,
             linestyle=style_of_line, linewidth=width_of_line, label=label_mashhadi)
    for i, v in enumerate(accuracy_mashhadi):
        if top_k[i] == 1 or top_k[i] == 10 or top_k[i] == 50:
            plt.plot(top_k[i], v, marker=type_of_marker, markersize=size_of_marker, color=color_mashhadi)
        if top_k [i] == 1:
            plt.text(top_k[i]+1, v-0.015, str(v), color=color_mashhadi, size=size_font)
        if top_k [i] == 10:
            plt.text(top_k[i], v-0.01, str(v), color=color_mashhadi, size=size_font)
        if top_k [i] == 50:
            plt.text(top_k[i]+1, v-0.01, str(v), color=color_mashhadi, size=size_font)

    plt.plot (top_k, accuracy_batool, color=color_batool,
              linestyle=style_of_line, linewidth=width_of_line, label=label_batool)
    for i, v in enumerate (accuracy_batool):
        if top_k[i] == 1 or top_k[i] == 10 or top_k[i] == 50:
            plt.plot(top_k[i], v, marker=type_of_marker, markersize=size_of_marker, color=color_batool)
        if top_k [i] == 1:
            plt.text(top_k[i] + 1, v - 0.015, str(v), color=color_batool, size=size_font)
        if top_k [i] == 10:
            plt.text(top_k[i], v - 0.04, str(v), color=color_batool, size=size_font)
        if top_k [i] == 50:
            plt.text(top_k[i] + 1, v-0.022, str(v), color=color_batool, size=size_font)

    plt.plot (top_k, accuracy_wisard, color=color_wisard,
              linestyle=style_of_line, linewidth=width_of_line, label=label_wisard)
    for i, v in enumerate (accuracy_wisard):
        if top_k[i] == 1 or top_k[i] == 10 or top_k[i] == 50:
            plt.plot(top_k [i], v, marker=type_of_marker, markersize=size_of_marker, color=color_wisard)
        if top_k[i] == 1:
            plt.text(top_k[i]+1, v, str(v), color=color_wisard, size=size_font)
        elif top_k[i] == 10:
            plt.text(top_k[i], v+0.02, str(v), color=color_wisard, size=size_font)
        elif top_k[i] == 50:
            plt.text(top_k[i] + 1, v-0.04, str(v), color=color_wisard, size=size_font)

    plt.title('Comparacao entre as referencias e a WiSARD \n Acuracia Top-K dos dados '+ input , color='steelblue', size=14, fontweight='bold')
    #plt.xticks(top_k)
    plt.xlabel('Top-k', color='steelblue', size=14, fontweight='bold')
    #plt.yscale('linear')
    plt.ylim([0.4, 1.05])
    plt.xlim ([0, 55])

    plt.grid()
    plt.ylabel('Accuracy', color='steelblue', size=14, fontweight='bold')
    plt.legend(loc='lower right')

    #plt.grid(False)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_results_top_k(top_k, input, filename,
                       ruseckas,
                       wisard,
                       batool,
                       label_ruseckas,
                       label_wisard,
                       label_batool):


    color_ruseckas = 'blue'
    color_wisard = 'red'
    color_batool = 'teal'

    style_of_line = 'solid'
    width_of_line = 1
    marker_type = 'o'
    marker_size = 3

    #sns.set()
    plt.figure()
    plt.plot (top_k, wisard, color=color_wisard,
              linestyle=style_of_line, linewidth=width_of_line, label=label_wisard)
    for i, v in enumerate (wisard):
        if top_k[i] == 1 or top_k[i] == 10 or top_k[i] == 50:
            plt.plot (top_k[i], v, marker=marker_type, markersize=3, color=color_wisard)
            if input == 'coord':
                if top_k [i] == 1:
                    plt.text(top_k[i]+0.5, v-0.03 , str(v), color=color_wisard, size=8)
                if top_k [i] == 10:
                    plt.text(top_k[i]+0.8, v-0.03 , str (v), color=color_wisard, size=8)
                if top_k [i] == 50:
                    plt.text(top_k[i]+0.8, v-0.03 , str (v), color=color_wisard, size=8)

        if input == 'lidar':
            if top_k [i] == 1:
                plt.text (top_k[i]+1, v, str(v), color=color_wisard, size=8)
            elif top_k [i] == 10:
                plt.text (top_k[i], v-0.018, str (v), color=color_wisard, size=8)
            elif top_k [i] == 50:
                plt.text (top_k[i]+1, v-0.018, str(v), color=color_wisard, size=8)
        if input == 'lidar_coord':
            if top_k [i] == 1:
                plt.text (top_k[i]+1, v, str(v), color=color_wisard, size=8)
            elif top_k [i] == 10:
                plt.text (top_k[i], v-0.018, str (v), color=color_wisard, size=8)
            elif top_k [i] == 50:
                plt.text (top_k[i]+1, v, str (v), color=color_wisard, size=8)

    plt.plot(top_k, ruseckas, color=color_ruseckas,
             linestyle=style_of_line, linewidth=width_of_line, label=label_ruseckas, alpha=0.5)
    for i, v in enumerate (ruseckas):
        if top_k[i] == 1 or top_k [i] == 10 or top_k[i] == 50:
            plt.plot (top_k[i], v, marker=marker_type, markersize=marker_size, color=color_ruseckas)
            if input == 'coord':
                if top_k [i] == 1:
                    plt.text (top_k[i]+0.5, v, str(v), color=color_ruseckas, size=8)
                if top_k [i] == 10:
                    plt.text (top_k[i]+0.8, v+0.03, str(v), color=color_ruseckas, size=8)
                if top_k [i] == 50:
                    plt.text (top_k[i]+1, v+0.02, str(v), color=color_ruseckas, size=8)

            if input == 'lidar':
                if top_k[i] == 1:
                    plt.text (top_k[i]+1, v, str(v), color=color_ruseckas, size=8)
                elif top_k[i] == 10:
                    plt.text (top_k[i], v+0.022, str(v), color=color_ruseckas, size=8)
                elif top_k[i] == 50:
                    plt.text(top_k[i]+1, v+0.022, str(v), color=color_ruseckas, size=8)

            if input == 'lidar_coord':
                if top_k [i] == 1:
                    plt.text (top_k[i]+1, v, str(v), color=color_ruseckas, size=8)
                elif top_k [i] == 10:
                    plt.text (top_k[i], v+0.022, str(v), color=color_ruseckas, size=8)
                elif top_k [i] == 50:
                    plt.text (top_k[i]+1, v+0.022, str(v), color=color_ruseckas, size=8)

    plt.plot(top_k, batool, color=color_batool,
             linestyle=style_of_line, linewidth=width_of_line, label=label_batool)
    for i, v in enumerate (batool):
        if top_k [i] == 1 or top_k [i] == 10 or top_k[i]==50:
            plt.plot (top_k [i], v, marker=marker_type, markersize=marker_size, color=color_batool)
            if input == 'coord':
                if top_k [i] == 1:
                    plt.text (top_k [i]+0.5, v, str (v), color=color_batool, size=8)
                if top_k [i] == 10:
                    plt.text (top_k [i], v-0.0480, str (v), color=color_batool, size=8)
                if top_k [i] == 50:
                    plt.text (top_k [i]+0.8, v-0.0480, str (v), color=color_batool, size=8)
        if input == 'lidar':
            if top_k [i] == 1:
                plt.text(top_k [i]+1, v-0.015, str(v), color=color_batool, size=8)
            if top_k [i] == 10:
                plt.text(top_k [i], v-0.04, str(v), color=color_batool, size=8)
            if top_k [i] == 50:
                plt.text(top_k [i]+1, v-0.01, str(v), color=color_batool, size=8)

        if input == 'lidar_coord':
            if top_k [i] == 1:
                plt.text(top_k [i]+1, v, str(v), color=color_batool, size=8)
            if top_k [i] == 10:
                plt.text(top_k [i], v-0.035, str(v), color=color_batool, size=8)
            if top_k [i] == 50:
                plt.text(top_k [i]+1, v-0.035, str(v), color=color_batool, size=8)




    plt.title('Comparacao entre as referencias e a WiSARD \n Acuracia Top-K dos dados '+ input , color='steelblue', size=14, fontweight='bold')
    #plt.xticks(top_k)
    plt.xlabel('Top-k', color='steelblue', size=14, fontweight='bold')
    #plt.yscale('linear')
    if input == 'lidar_coord' or input == 'lidar':
        plt.ylim([0.4, 1.1])
    else:
        plt.ylim ([0, 1.1])
    plt.xlim ([0, 55])

    plt.grid()
    plt.ylabel('Accuracy', color='steelblue', size=14, fontweight='bold')
    plt.legend(loc='lower right')

    #plt.grid(False)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def readCSVfile(input, reference):
    path = '../results/score/' + reference + '/top_k/' + input + '/'
    filename = 'score_'+ input + '_top_k.csv'
    usecols = ["Top-k", "Acuracia"]
    print(path + filename)
    data = pd.read_csv (path + filename,  delimiter=',')
    #data = pd.read_csv (path + filename, header=None, delimiter='\t', usecols=usecols, names=["Top-k", "Acuracia"])
    #print(path + filename)
    accuracy = data['Acuracia'].tolist()
    #accuracy = [float (i) for i in accuracy [1:]]
    #accuracy = [round (i, 2) for i in accuracy]
    accuracy = np.round(data['Acuracia'].tolist(),3)

    top_k = data ['Top-K'].tolist()
    #top_k = [float(i) for i in top_k[1:]]
    top_k = np.round(data['Top-K'].tolist(),3)

    return top_k, accuracy


def read_csv_file(input, filename):
    #path = '../results/accuracy/8x32/' + input + '/'
    path = '../results/accuracy/8x32/accuracy_new_labels/'+ input + '/'
    usecols = ["Top-k", "Acuracia"]
    print(path + filename)
    data = pd.read_csv (path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
    print(path + filename)
    accuracy = data['Acuracia'].tolist()
    accuracy = [float (i) for i in accuracy [1:]]
    accuracy = [round (i, 2) for i in accuracy]

    top_k = data ['Top-k'].tolist()
    top_k = [float(i) for i in top_k[1:]]

    return top_k, accuracy
def read_results_top_k_lidar_e_coord():
    only_coord = True
    only_lidar = False
    lidar_coord = False
    compare_by_ref = False

    path_of_compare = '../results/accuracy/8x32/'
    usecols = ["Top-k", "Acuracia"]

    if compare_by_ref:
        # Read the results of the strategies
        # Data of Ref-17
        filename = 'ref_17/acuracia_lidar_coord_top_k.csv'
        top_k, accuracy_lidar_coord_ref_17 = read_csv_file(input='coord_lidar', filename=filename)
        filename = 'ref_17/acuracia_lidar_coord_s008_top_k.csv'
        _, accuracy_lidar_coord_ref_17_s008 = read_csv_file(input='coord_lidar', filename=filename)
        filename = 'ref_17/acuracia_lidar_top_k.csv'
        _, accuracy_lidar_ref_17 = read_csv_file (input='lidar', filename=filename)
        filename = '/ref_17/acuracia_coord_top_k.csv'
        _, accuracy_coord_ref_17 = read_csv_file (input='coord', filename=filename)

        filename = path_of_compare + 'performance_ref_17.png'
        plot_results_top_k_for_strategy (top_k=top_k, filename=filename,
                                         accuracy_coord=accuracy_coord_ref_17,
                                         accuracy_lidar=accuracy_lidar_ref_17,
                                         accuracy_lidar_coord=accuracy_lidar_coord_ref_17,
                                         title='Performance of Strategy Ref_17 \n [Train: s008 - Test: s009]')

        # Data of Wisard
        #----Lidar + Coord
        _, accuracy_wisard_lidar_coord_s008_s009 = read_csv_file(input='coord_lidar', filename= 'acuracia_wisard_coord_+_lidar_s008-s009_top_k.csv')
        _, accuracy_wisard_lidar_coord_s008 = read_csv_file(input='coord_lidar', filename= 'acuracia_wisard_coord_lidar_s008_top_k.csv')
        _, accuracy_wisard_lidar_coord_s009 = read_csv_file(input='coord_lidar', filename= 'acuracia_wisard_coord_lidar_s009_top_k.csv')
        _, accuracy_wisard_lidar_2D_rx_therm_coord_s008_s009 = read_csv_file(input='lidar_2D_with_rx_2D_therm_and_coord', filename='acuracia_wisard_lidar_2D_with_rx_2D_therm_and_coord_s008-s009_top_k.csv')
        #----Lidar
        filename = 'acuracia_wisard_lidar_s008-s009_top_k.csv'
        _, accuracy_wisard_lidar_s008_s009 = read_csv_file (input='lidar', filename=filename)
        filename = 'acuracia_wisard_lidar_s008_top_k.csv'
        _, accuracy_wisard_lidar_s008 = read_csv_file (input='lidar', filename=filename)
        filename = 'acuracia_wisard_lidar_s009_top_k.csv'
        _, accuracy_wisard_lidar_s009 = read_csv_file (input='lidar', filename=filename)
        filename = ''
        _, accuracy_wisard_lidar_2D_rx_therm_s008_s009 = read_csv_file(input='lidar_2D_with_rx_2D_thermometer', filename='acuracia_wisard_lidar_2D_with_rx_2D_thermometer_s008-s009_top_k.csv')
        #----Coord
        filename = 'acuracia_wisard_coord_s008-s009_top_k.csv'
        _, accuracy_wisard_coord_s008_s009 = read_csv_file (input='coord', filename=filename)
        filename = 'acuracia_wisard_coord_s008_top_k.csv'
        _, accuracy_wisard_coord_s008 = read_csv_file (input='coord', filename=filename)
        filename = 'acuracia_wisard_coord_s009_top_k.csv'
        _, accuracy_wisard_coord_s009 = read_csv_file (input='coord', filename=filename)

        filename= path_of_compare + 'performance_wisard_lidar_2D_coord_s008-s009.png'
        plot_results_top_k_for_strategy(top_k=top_k, filename=filename,
                           accuracy_coord=accuracy_wisard_coord_s008_s009,
                           accuracy_lidar=accuracy_wisard_lidar_2D_rx_therm_s008_s009,
                           accuracy_lidar_coord=accuracy_wisard_lidar_2D_rx_therm_coord_s008_s009,
                           title='Performance of Strategy Wisard (2D) \n [Train: s008 - Test: s009]')

        filename = path_of_compare + 'performance_wisard_s008-s009.png'
        plot_results_top_k_for_strategy (top_k=top_k, filename=filename,
                                         accuracy_coord=accuracy_wisard_coord_s008_s009,
                                         accuracy_lidar=accuracy_wisard_lidar_s008_s009,
                                         accuracy_lidar_coord=accuracy_wisard_lidar_coord_s008_s009,
                                         title='Performance of Strategy Wisard \n [Train: s008 - Test: s009]')
        filename = path_of_compare + 'performance_wisard_s008.png'
        plot_results_top_k_for_strategy (top_k, filename,
                                         accuracy_coord=accuracy_wisard_coord_s008,
                                         accuracy_lidar=accuracy_wisard_lidar_s008,
                                         accuracy_lidar_coord=accuracy_wisard_lidar_coord_s008,
                                         title='Performance of Strategy Wisard \n [Train: s008 - Test: s008]')
        filename = path_of_compare + 'performance_wisard_s009.png'
        plot_results_top_k_for_strategy (top_k, filename,
                                         accuracy_coord=accuracy_wisard_coord_s009,
                                         accuracy_lidar=accuracy_wisard_lidar_s009,
                                         accuracy_lidar_coord=accuracy_wisard_lidar_coord_s009,
                                         title='Performance of Strategy Wisard \n [Train: s009 - Test: s009]')

        # Data of ref_batool
        filename = 'ref_batool/acuracia_batool_coord_lidar_top_k.csv'
        _, accuracy_lidar_coord_batool = read_csv_file(input='coord_lidar', filename=filename)
        filename = 'ref_batool/acuracia_batool_lidar_top_k.csv'
        _, accuracy_Lidar_batool = read_csv_file(input='lidar', filename=filename)
        filename = 'ref_batool/acuracia_batool_coord_top_k.csv'
        _, accuracy_coord_batool = read_csv_file(input='coord', filename=filename)

        filename = path_of_compare + 'performance_batool.png'
        plot_results_top_k_for_strategy (top_k, filename,
                                         accuracy_coord_batool,
                                         accuracy_Lidar_batool,
                                         accuracy_lidar_coord_batool,
                                         'Performance of Strategy Batool \n [Train: s008 - Test: s009]')
        a=0
    else:
        if only_lidar:

            path_of_compare = '../results/accuracy/8x32/'
            input = 'lidar'
            path = '../results/accuracy/8x32/'+input+'/' #-----
            filename = 'ref_17/acuracia_lidar_top_k.csv'
            _,accuracy_lidar_ref_17 = read_csv_file(input='lidar', filename=filename)

            '''
            usecols = ["Top-k", "Acuracia"]
            data_ref_17 = pd.read_csv(path+filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_lidar_ref_17 = data_ref_17['Acuracia'].tolist()
            accuracy_lidar_ref_17 = [float(i) for i in accuracy_lidar_ref_17[1:]]
            accuracy_lidar_ref_17 = [round(i, 2) for i in accuracy_lidar_ref_17]
            
            top_k = data_ref_17['Top-k'].tolist()
            top_k = [float(i) for i in top_k[1:]]
            '''

            filename = 'acuracia_wisard_lidar_2D_with_rx_2D_thermometer_s008-s009_top_k.csv'
            _, accuracy_wisard_lidar_2D_rx_therm_s008_s009 = read_csv_file(input='lidar_2D_with_rx_2D_thermometer',
                                                                           filename=filename)
            filename = 'acuracia_wisard_lidar_s008-s009_top_k.csv'
            _, accuracy_wisard_lidar_s008_s009 = read_csv_file(input='lidar', filename=filename)

            filename = 'acuracia_wisard_lidar_s008_top_k.csv'
            _, accuracy_wisard_lidar_s008 = read_csv_file(input='lidar', filename=filename)

            filename = 'acuracia_wisard_lidar_s009_top_k.csv'
            _, accuracy_wisard_lidar_s009 = read_csv_file(input='lidar', filename=filename)

            filename = 'ref_batool/acuracia_batool_lidar_top_k.csv'
            top_k, accuracy_Lidar_batool = read_csv_file(input='lidar', filename=filename)

            filename = 'acuracia_wisard_LiDAR_2D_+_Rx_Term_SVar_top_k.csv'
            _, accuracy_wisard_lidar_2D_rx_therm_sv = read_csv_file(input= 'lidar_+_coord/top_k' ,
                                                                    filename=filename)


            '''
            filename = 'acuracia_wisard_lidar_s008-s009_top_k.csv'
            _, accuracy_wisard_lidar_s008_s009 = read_csv_file(input='lidar', filename=filename)
            data_wisard_s008_s009 = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_wisard_lidar_s008_s009 = data_wisard_s008_s009['Acuracia'].tolist()
            accuracy_wisard_lidar_s008_s009 = [float(i) for i in accuracy_wisard_lidar_s008_s009[1:]]
            accuracy_wisard_lidar_s008_s009 = [round(i, 2) for i in accuracy_wisard_lidar_s008_s009]

            filename = 'acuracia_wisard_lidar_s008_top_k.csv'
            data_wisard_s008 = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_wisard_lidar_s008 = data_wisard_s008['Acuracia'].tolist()
            accuracy_wisard_lidar_s008 = [float(i) for i in accuracy_wisard_lidar_s008[1:]]
            accuracy_wisard_lidar_s008 = [round(i, 2) for i in accuracy_wisard_lidar_s008]

            filename = 'acuracia_wisard_lidar_s009_top_k.csv'
            data_wisard_s009 = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_wisard_lidar_s009 = data_wisard_s009['Acuracia'].tolist()
            accuracy_wisard_lidar_s009 = [float(i) for i in accuracy_wisard_lidar_s009[1:]]
            accuracy_wisard_lidar_s009 = [round(i, 2) for i in accuracy_wisard_lidar_s009]

            filename = 'ref_batool/acuracia_batool_lidar_top_k.csv'
            data_Lidar_batool = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_Lidar_batool = data_Lidar_batool ['Acuracia'].tolist()
            accuracy_Lidar_batool = [float(i) for i in accuracy_Lidar_batool[1:]]
            accuracy_Lidar_batool = [round(i, 2) for i in accuracy_Lidar_batool]
            '''

            label_ref_17 ='Ruseckas [CNN 2D]: LiDAR' #'LIDAR: Ref-17 [s008-s009]'
            label_wisard_lidar_2D_s008_s009 = 'LIDAR 2D: Wisard [s008-s009]'
            label_wisard_s008 = 'LIDAR: Wisard [s008]'
            label_wisard_s009 = 'LIDAR: Wisard [s009]'
            label_batool ='Batool [DNN 3D]: LiDAR' #'LIDAR: Batool [s008-s009]'
            label_wisard_lidar_2D_rx_term_sv = 'UFRJ [WiSARD] LiDAR 2D + Rx Term SVar'

            figure_name = path_of_compare+'campare_top_k_accuracy_lidar.png'
            input = 'LIDAR'
            plot_results_top_k (top_k,
                                input,
                                figure_name,
                                accuracy_lidar_ref_17,
                                accuracy_wisard_lidar_2D_rx_therm_sv,
                                #accuracy_wisard_lidar_2D_rx_therm_s008_s009,
                                #accuracy_wisard_lidar_s008,
                                #accuracy_wisard_lidar_s009,
                                accuracy_Lidar_batool,
                                label_ref_17,
                                label_wisard_lidar_2D_rx_term_sv,
                                #label_wisard_lidar_2D_s008_s009,
                                #label_wisard_s008,
                                #label_wisard_s009,
                                label_batool)
            a=0


        if only_coord:

            input = 'coord'
            usecols = ["Top-k", "Acuracia"]
            path = '../results/accuracy/8x32/'+input+'/'

            filename = 'ref_17/acuracia_coord_top_k.csv'
            _, accuracy_coord_ref_17 = read_csv_file(input='coord', filename=filename)

            filename = 'acuracia_wisard_coord_s008-s009_top_k.csv'
            _, accuracy_wisard_coord_s008_s009 = read_csv_file(input='coord', filename=filename)

            filename = 'acuracia_wisard_coord_s008_top_k.csv'
            _, accuracy_wisard_coord_s008 = read_csv_file(input='coord', filename=filename)

            filename = 'acuracia_wisard_coord_s009_top_k.csv'
            _, accuracy_wisard_coord_s009 = read_csv_file(input='coord', filename=filename)

            filename = 'ref_batool/acuracia_batool_coord_top_k.csv'
            top_k, accuracy_coord_batool = read_csv_file(input='coord', filename=filename)

            filename = 'acuracia_wisard_coord_top_k.csv'
            _, accuracy_wisard_coord = read_csv_file(input='lidar_+_coord/top_k', filename=filename)

            path = '../results/accuracy/8x32/' + input + '/'

            '''
                        

            data_coord_ref_17 = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_coord_ref_17 = data_coord_ref_17['Acuracia'].tolist()
            accuracy_coord_ref_17 = [float(i) for i in accuracy_coord_ref_17[1:]]
            accuracy_coord_ref_17 = [round(i, 2) for i in accuracy_coord_ref_17]

            top_k = data_coord_ref_17 ['Top-k'].tolist ()
            top_k = [float (i) for i in top_k [1:]]

            
            
            
           
            data_wisard_coord_s008_s009 = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_wisard_coord_s008_s009 = data_wisard_coord_s008_s009['Acuracia'].tolist()
            accuracy_wisard_coord_s008_s009 = [float(i) for i in accuracy_wisard_coord_s008_s009[1:]]
            accuracy_wisard_coord_s008_s009 = [round(i, 2) for i in accuracy_wisard_coord_s008_s009]
             
             
            data_wisard_coord_s008 = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_wisard_coord_s008 = data_wisard_coord_s008['Acuracia'].tolist()
            accuracy_wisard_coord_s008 = [float(i) for i in accuracy_wisard_coord_s008[1:]]
            accuracy_wisard_coord_s008 = [round(i, 2) for i in accuracy_wisard_coord_s008]

           
            data_wisard_coord_s009 = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_wisard_coord_s009 = data_wisard_coord_s009['Acuracia'].tolist()
            accuracy_wisard_coord_s009 = [float(i) for i in accuracy_wisard_coord_s009[1:]]
            accuracy_wisard_coord_s009 = [round(i, 2) for i in accuracy_wisard_coord_s009]

            
            data_coord_batool = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_coord_batool = data_coord_batool ['Acuracia'].tolist()
            accuracy_coord_batool = [float(i) for i in accuracy_coord_batool[1:]]
            accuracy_coord_batool = [round(i, 2) for i in accuracy_coord_batool]
            '''

            label_ref_17 = 'COORD: Ruseckas [MLP]'
            label_wisard_s008_s009 = 'COORD: UFRJ [WiSARD]'
            label_wisard_s008 = 'COORD: Wisard [s008]'
            label_wisard_s009 = 'COORD: Wisard [s009]'
            label_batool = 'COORD: Batool [CNN-1D]'

            figure_name = path_of_compare+'campare_top_k_accuracy_coord.png'
            input = 'COORD'

            plot_results_top_k(top_k,
                               input,
                               figure_name,
                               accuracy_coord_ref_17,
                               accuracy_wisard_coord_s008_s009,
                               #accuracy_wisard_coord_s008,
                               #accuracy_wisard_coord_s009,
                               accuracy_coord_batool,
                               label_ref_17,
                               label_wisard_s008_s009,
                               #label_wisard_s008,
                               #label_wisard_s009,
                               label_batool)

        if lidar_coord:

            input = 'coord_lidar'
            usecols = ["Top-k", "Acuracia"]
            path = '../results/accuracy/8x32/'+input+'/'


            filename = 'ref_17/acuracia_lidar_coord_top_k.csv'
            _, accuracy_lidar_coord_ref_17 = read_csv_file(input='coord_lidar', filename=filename)

            filename = 'ref_batool/acuracia_batool_coord_lidar_top_k.csv'
            _, accuracy_coord_lidar_batool = read_csv_file (input='coord_lidar', filename=filename)

            filename = 'acuracia_wisard_coord_lidar_s008_top_k.csv'
            _, accuracy_wisard_lidar_coord_s008 = read_csv_file (input='coord_lidar', filename=filename)

            filename = 'acuracia_wisard_coord_lidar_s009_top_k.csv'
            _, accuracy_wisard_lidar_coord_s009 = read_csv_file (input='coord_lidar', filename=filename)

            filename = 'acuracia_wisard_coord_lidar_s008-s009_top_k.csv'
            _, accuracy_wisard_lidar_coord_s008_s009 = read_csv_file (input='coord_lidar', filename=filename)

            filename = 'acuracia_wisard_lidar_2D_with_rx_2D_therm_and_coord_s008-s009_top_k.csv'
            top_k, accuracy_wisard_lidar_2D_coord_s008_s009 = read_csv_file(input='lidar_2D_with_rx_2D_therm_and_coord', filename=filename)

            filename = 'acuracia_wisard_LiDAR_2D_+_Rx_Term_+_Coord_16_SVar_top_k.csv'
            _, accuracy_wisard_lidar_2D_rx_therm_coord_sv = read_csv_file(input='lidar_+_coord/top_k', filename=filename)

            '''
            data_lidar_coord_ref_17 = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_lidar_coord_ref_17 = data_lidar_coord_ref_17['Acuracia'].tolist()
            accuracy_lidar_coord_ref_17 = [float(i) for i in accuracy_lidar_coord_ref_17[1:]]
            accuracy_lidar_coord_ref_17 = [round(i, 2) for i in accuracy_lidar_coord_ref_17]

            top_k = data_lidar_coord_ref_17['Top-k'].tolist()
            top_k = [float(i) for i in top_k[1:]]



            data_lidar_coord_ref_17_s008 = pd.read_csv(path+filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_lidar_coord_ref_17_s008 = data_lidar_coord_ref_17_s008['Acuracia'].tolist()
            accuracy_lidar_coord_ref_17_s008 = [float(i) for i in accuracy_lidar_coord_ref_17_s008[1:]]
            accuracy_lidar_coord_ref_17_s008 = [round(i, 2) for i in accuracy_lidar_coord_ref_17_s008]

            path = '../results/accuracy/8x32/'+input+'/'


            data_wisard_lidar_coord_s008_s009 = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_wisard_lidar_coord_s008_s009 = data_wisard_lidar_coord_s008_s009['Acuracia'].tolist()
            accuracy_wisard_lidar_coord_s008_s009 = [float(i) for i in accuracy_wisard_lidar_coord_s008_s009[1:]]
            accuracy_wisard_lidar_coord_s008_s009 = [round(i, 2) for i in accuracy_wisard_lidar_coord_s008_s009]


            data_wisard_lidar_coord_s008 = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_wisard_lidar_coord_s008 = data_wisard_lidar_coord_s008['Acuracia'].tolist()
            accuracy_wisard_lidar_coord_s008 = [float(i) for i in accuracy_wisard_lidar_coord_s008[1:]]
            accuracy_wisard_lidar_coord_s008 = [round(i, 2) for i in accuracy_wisard_lidar_coord_s008]


            data_wisard_lidar_coord_s009 = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_wisard_lidar_coord_s009 = data_wisard_lidar_coord_s009['Acuracia'].tolist()
            accuracy_wisard_lidar_coord_s009 = [float(i) for i in accuracy_wisard_lidar_coord_s009[1:]]
            accuracy_wisard_lidar_coord_s009 = [round(i, 2) for i in accuracy_wisard_lidar_coord_s009]


            data_coord_lidar_batool = pd.read_csv(path + filename, header=None, usecols=usecols, names=["Top-k", "Acuracia"])
            accuracy_coord_lidar_batool = data_coord_lidar_batool ['Acuracia'].tolist()
            accuracy_coord_lidar_batool = [float(i) for i in accuracy_coord_lidar_batool[1:]]
            accuracy_coord_lidar_batool = [round(i, 2) for i in accuracy_coord_lidar_batool]
            '''

            label_ref_17 = 'COORD + LIDAR: Ruseckas [MLP+CNN]'
            label_batool = 'COORD + LIDAR: Batool [DNN]'
            label_wisard_2D_s008_s009 = 'COORD + LIDAR 2D: UFRJ [WiSARD]'

            label_ref_17_s008 = 'COORD + LIDAR: Ref-17 [s008]'
            label_wisard_s008_s009 = 'COORD + LIDAR: Wisard [s008-s009]'
            label_wisard_s008 = 'COORD + LIDAR: Wisard [s008]'
            label_wisard_s009 = 'COORD + LIDAR: Wisard [s009]'


            figure_name = path_of_compare+'campare_top_k_accuracy_coord_lidar.png'
            input = 'COORD + LIDAR'

            plot_results_top_k(top_k,
                               input,
                               figure_name,
                               accuracy_lidar_coord_ref_17,
                               accuracy_wisard_lidar_2D_rx_therm_coord_sv,
                               #accuracy_wisard_lidar_2D_coord_s008_s009,
                               #accuracy_wisard_lidar_coord_s008_s009,
                               #accuracy_wisard_lidar_coord_s008,
                               #accuracy_wisard_lidar_coord_s009,
                               accuracy_coord_lidar_batool,
                               label_ref_17,
                               label_wisard_2D_s008_s009,
                               #label_ref_17_s008,
                               #label_wisard_s008_s009,
                               #label_wisard_s008,
                               #label_wisard_s009,
                               label_batool)
def plot_compare_accuracy_top_k():
    '''Read and plot accuracy of 2 reference and wisard
    using the beams generated by  '''
    input = 'lidar_coord' #lidar' #'coord', 'lidar_coord'
    #path_to_save = '../results/accuracy/8x32/accuracy_new_labels/'
    path_to_save = '../results/score/'

    reference = 'batool'
    top_k, accuracy_batool = readCSVfile(input=input, reference=reference)

    reference = 'ruseckas'
    _, accuracy_ruseckas = readCSVfile (input=input, reference=reference)

    reference = 'wisard'
    filename = 'score_' + reference + '_' + input + '_top_k.csv'
    _, accuracy_wisard = read_csv_file (input=input, filename=filename)
    reference = 'Wisard'
    _, accuracy_wisard = readCSVfile (input=input, reference=reference)

    if input =='lidar':
        reference = 'Mashhadi'
        _, accuracy_mashhadi = readCSVfile(input=input, reference=reference)

    figure_name = path_to_save + 'score_top_k_compare_'+input+'.png'

    if input == 'coord':
        label_ruseckas = 'COORD: Ruseckas [CNN]'
        label_batool = 'COORD: Batool [DNN]'
        label_wisard = 'COORD: UFRJ [WiSARD]'
    elif input == 'lidar':
        label_ruseckas = 'LiDAR: Ruseckas[CNN]'
        label_batool = 'LiDAR: Batool[DNN]'
        label_wisard = 'LiDAR: UFRJ[WiSARD]'
        label_mashhadi = 'LiDAR: Mashhadi[CNN]'
    elif input == 'lidar_coord':
        label_ruseckas = 'COORD + LIDAR: Ruseckas [MLP+CNN]'
        label_batool = 'COORD + LIDAR: Batool [DNN]'
        label_wisard = 'COORD + LIDAR 2D: UFRJ [WiSARD]'

    if input == 'lidar':
        plot_results_LiDAR_top_k(top_k, input, figure_name,
                                 accuracy_ruseckas, accuracy_wisard, accuracy_batool, accuracy_mashhadi,
                                 label_ruseckas, label_wisard, label_batool, label_mashhadi)
    else:
        plot_results_top_k (top_k, input, figure_name,
                            accuracy_ruseckas, accuracy_wisard, accuracy_batool,
                            label_ruseckas, label_wisard, label_batool)


def calculate_and_save_rt_all_ref():
    input = 'lidar_coord'  # lidar' #'coord', 'lidar_coord'
    if input =='lidar':
        reference = 'Mashhadi'
        filename = 'score_' + reference + '_' + input + '_top_k.csv'
        _, accuracy_mashhadi = read_csv_file (input=input, filename=filename)
        ratio_thr_wisard, ratio_thr_batool, ratio_thr_ruseckas, ratio_thr_mashhadi = tp.throughput_ratio_for_all_techniques(input)
    else:
        ratio_thr_wisard, ratio_thr_batool, ratio_thr_ruseckas = tp.throughput_ratio_for_all_techniques(input)

    top_k = list(range(1, 51))
    df = pd.DataFrame(data={"top-k": top_k,
                            "rt_wisard": ratio_thr_wisard,
                            "rt_batool": ratio_thr_batool,
                            "rt_ruseckas": ratio_thr_ruseckas} )
    df.to_csv('../results/index_beams_predict/througput_ratio_all_models_' + input + '_top_k.csv', index=False, sep=',')
    

def plot_compare_score_and_rt_top_k():
    input = 'lidar'  # lidar' #'coord', 'lidar_coord'
    path_to_save = '../results/'

    #READ SCORE'S FILES
    reference = 'batool'
    filename = 'score_' + reference + '_' + input + '_top_k.csv'
    top_k, accuracy_batool = read_csv_file (input=input, filename=filename)

    reference = 'ruseckas'
    filename = 'score_' + reference + '_' + input + '_top_k.csv'
    _, accuracy_ruseckas = read_csv_file (input=input, filename=filename)

    reference = 'wisard'
    filename = 'score_' + reference + '_' + input + '_top_k.csv'
    _, accuracy_wisard = read_csv_file (input=input, filename=filename)

    #RT RESULTS
    if input =='lidar':
        reference = 'Mashhadi'
        filename = 'score_' + reference + '_' + input + '_top_k.csv'
        _, accuracy_mashhadi = read_csv_file (input=input, filename=filename)
        ratio_thr_wisard, ratio_thr_batool, ratio_thr_ruseckas, ratio_thr_mashhadi = tp.throughput_ratio_for_all_techniques(input)
    else:
        ratio_thr_wisard, ratio_thr_batool, ratio_thr_ruseckas = tp.throughput_ratio_for_all_techniques(input)

    figure_name = path_to_save + 'campare_score_RT_' + input + '.png'

    if input == 'coord':
        label_ruseckas = 'COORD: Ruseckas[CNN]'
        label_batool = 'COORD: Batool[DNN]'
        label_wisard = 'COORD: UFRJ[WiSARD]'
    elif input == 'lidar':
        label_ruseckas = 'LiDAR: Ruseckas[CNN]'
        label_batool = 'LiDAR: Batool[DNN]'
        label_wisard = 'LiDAR: UFRJ[WiSARD]'
        label_mashhadi = 'LiDAR: Mashhadi[CNN]'
    elif input == 'lidar_coord':
        label_ruseckas = 'COORD + LIDAR: Ruseckas[MLP+CNN]'
        label_batool = 'COORD + LIDAR: Batool[DNN]'
        label_wisard = 'COORD + LIDAR 2D: UFRJ[WiSARD]'


    if input =='lidar':
        plot_LiDAR_score_rt_top_k(input=input, filename=figure_name,
                                  top_k=top_k,
                                  score_1=accuracy_ruseckas,
                                  rt_1=ratio_thr_ruseckas,
                                  label_1=label_ruseckas,
                                  score_2=accuracy_wisard,
                                  rt_2=ratio_thr_wisard,
                                  label_2=label_wisard,
                                  score_3=accuracy_batool,
                                  rt_3=ratio_thr_batool,
                                  label_3=label_batool,
                                  score_4=accuracy_mashhadi,
                                  rt_4=ratio_thr_mashhadi,
                                  label_4=label_mashhadi)

    else:
        plot_compare_score_and_rt_top_k(input=input, filename=figure_name, top_k=top_k,
                            score_1=accuracy_ruseckas, rt_1=ratio_thr_ruseckas, label_1=label_ruseckas,
                            score_2=accuracy_wisard, rt_2=ratio_thr_wisard, label_2=label_wisard,
                            score_3=accuracy_batool, rt_3=ratio_thr_batool, label_3=label_batool)


#####----------------------------------------------------
# Comparacao das 3 metricas:
# Tamanho da memoria, Acuracia e tempos de treino e teste
def readCSVfile_memorySize_variation(input):
    path = '../results/score/Wisard/' + input + '/'
    filename = 'acuracia_' + input + '.csv'
    usecols = ["memory_Size", "score", "DP_score"]

    data = pd.read_csv (path + filename, delimiter='\t', names=usecols, index_col=False)
    score = data['score'].tolist()
    DP_score = data['DP_score'].tolist()
    memory_size = data['memory_Size'].tolist()

    return memory_size, score, DP_score
def readCSVfile_memorySizeVar_processTime(input):
    path = '../results/processingTime/Wisard/' + input + '/'
    filename = 'time_train_' + input + '.csv'
    usecols = ["memory_Size", "processTime", "DP_processTime"]
    data = pd.read_csv (path + filename, delimiter='\t', names=usecols, index_col=False)
    train_time = data ['processTime'].tolist ()
    DP_train_time = data ['DP_processTime'].tolist ()
    memory_size = data ['memory_Size'].tolist ()

    filename = 'time_test_' + input + '.csv'
    usecols = ["memory_Size", "processTime", "DP_processTime"]
    data = pd.read_csv (path + filename, delimiter='\t', names=usecols, index_col=False)
    test_time = data ['processTime'].tolist ()
    DP_test_time = data ['DP_processTime'].tolist ()
    memory_size = data ['memory_Size'].tolist ()

    return memory_size, train_time, DP_train_time, test_time, DP_test_time
def plot_of_score_memorySize_and_processTime(memory_size, time, score, DP_score, input, flag):
    sns.set_theme(style="darkgrid")
    fig, ax1 = plt.subplots (figsize=(15, 7))
    ax1.bar (memory_size, time, width=1.5, alpha=0.2, label=flag)
    ax1.set_ylabel ('Tempo de '+flag, color='blue')
    ax1.set_xlabel ('Tamanho da Memória')
    for i in range (len (memory_size)):
        ax1.text (memory_size[i] - 0.2, time[i] - 0.2,
                  str (np.round(time[i], 3)),
                  color='blue',
                  size=9,
                  rotation=90)

    # Criando um segundo eixo
    ax2 = ax1.twinx ()
    plt.plot (memory_size, score, marker='o', color='red', linewidth=2, label='acurácia')
    plt.errorbar (memory_size, score, yerr=DP_score, fmt='o', color='red', ecolor='red', capsize=5)
    if flag == 'teste':
        for i in range (len(memory_size)):
            ax2.text(memory_size[i], score[i]-0.004,
                      str(np.round(score[i], 3)),
                      color='red',
                      size=9,
                      rotation=0)

    ax2.set_ylabel ('Acurácia', color='red')

    # Adicionando título e legendas
    plt.title ("Relacao entre Acurácia e Tempo de "+flag+" \n [" + input + "]")
    plt.xticks (memory_size)
    plt.xlabel ('Tamanho da Memória')
    path ='../results/compare_score_time_memory/'
    nameFigure = input + '_[score_memorySize_tempo_'+flag+'].png'
    plt.savefig (path+nameFigure, dpi=300)
    plt.show()
def compare_of_score_memorySize_and_processTime():
    input = 'lidar_coord'  # lidar' #'coord', 'lidar_coord'

    if input == 'coord':
        memory_size, score, DP_score = readCSVfile_memorySize_variation(input)
        _memory_size, train_time, DP_train_time, test_time, DP_test_time = readCSVfile_memorySizeVar_processTime(input)

    if input == 'lidar':
        memory_size, score, DP_score = readCSVfile_memorySize_variation (input)
        _memory_size, train_time, DP_train_time, test_time, DP_test_time = readCSVfile_memorySizeVar_processTime(input)

    if input == 'lidar_coord':
        memory_size, score, DP_score = readCSVfile_memorySize_variation (input)
        _memory_size, train_time, DP_train_time, test_time, DP_test_time = readCSVfile_memorySizeVar_processTime(input)

    plot_of_score_memorySize_and_processTime (memory_size, train_time, score, DP_score, input, 'treinamento')
    plot_of_score_memorySize_and_processTime (memory_size, test_time, score, DP_score, input, 'teste')

def plot_results_dataset_inverter():

    ref = 'Batool'
    input_type = ['coord', 'lidar','lidar_coord']
    batool = []
    data = []
    for i in range(len(input_type) ):
        filename = 'accuracy_'+input_type[i]+'.csv'
        path = '../results/inverter_dataset/score/' + ref + '/'+ input_type[i] + '/ALL/'
        data = pd.read_csv (path + filename, delimiter=',')
        batool.append(data)

    #plt.plot(        batool[0]['top-k'], batool[0]['score'], marker='o', label='Batool Coord')

    ref = 'Ruseckas'
    ruseckas = []
    r = {}
    data = []
    for i in range(len(input_type)):
        filename = 'accuracy_'+input_type[i]+'.csv'
        path = '../results/inverter_dataset/score/' + ref + '/'+ input_type[i] + '/ALL/'
        data = pd.read_csv (path + filename, delimiter=',')
        ruseckas.append(data)

    #plt.plot (ruseckas [0]['top-k'], ruseckas [0] ['score'], marker='o', label='Batool Coord')

    ref = 'Wisard'
    wisard = []
    data = []
    for i in range(len(input_type) ):
        filename = 'accuracy_'+input_type[i]+'.csv'
        path = '../results/inverter_dataset/score/' + ref + '/top-k/'+ input_type[i] + '/ALL/'
        data = pd.read_csv (path + filename, delimiter=',')
        data = data[data['top-k'] < 31]  # Remove rows where 'top-k' is 0
        wisard.append(data)

    #plot results


    fig, ax = plt.subplots (1, 3, figsize=(14, 6), sharey=True)
    plt.subplots_adjust (left=0.08, right=0.98, bottom=0.1, top=0.9, hspace=0.12, wspace=0.05)

    size_of_font = 12
    labels = ['Coordenadas', 'LiDAR', 'LiDAR + Coordenadas']
    for i in range(len(input_type)):
        ax[i].plot(batool[i]['top-k'], batool[i]['score'], marker='o', label='Batool')
        ax[i].plot(ruseckas[i]['top-k'], ruseckas[i]['score'], marker='o', label='Ruseckas')
        ax[i].plot(wisard[i]['top-k'], wisard[i]['score'], marker='o', label='Wisard')
        #ax[i].set_xticks(batool[i]['top-k'])
        ax[i].grid(True, linestyle='--', alpha=0.5)

        ax[i].set_xlabel('Top-k \n' + labels[i])
        ax[0].set_ylabel('Acurácia')
        ax[1].legend(loc='lower right', fontsize=size_of_font)
    plt.tight_layout()
    path = '../results/inverter_dataset/score/'
    nameFigure = 'dataset_inverter_compare_accuracy.png'
    plt.savefig (path + nameFigure, dpi=300)
    a=0

def read_csv(filename, path):
    """
    Read a CSV file and return the data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(path + filename, delimiter=',')
        return data
    except FileNotFoundError:
        print(f"File {filename} not found in {path}.")
        return None

def read_all_results_inverter_dataset(ref, cenario):
    input_type = ['coord', 'lidar', 'lidar_coord']
    if cenario == 1:
        general_path = '../results/score/' + ref + '/top_k/'
        key = 'Acuracia'
        key_2 = 'Top-K'
        #if ref == 'Wisard' or ref == 'Ruseckas':
        #    key_2 = 'top-k'
    if cenario == 2:
        general_path = '../results/inverter_dataset/score/' + ref + '/'
        key = 'score'
        key_2 = 'top-k'
    for i in range (len (input_type)):
        if cenario == 1:
            filename = 'score_' + input_type [i] + '_top_k.csv'
            path = general_path + input_type [i] + '/'
        if cenario == 2:
            filename = 'accuracy_' + input_type [i] + '.csv'
            if ref == 'Wisard':
                path = general_path + '/top-k/' + input_type [i] + '/ALL/'
            else:
                path = general_path + '/' + input_type [i] + '/ALL/'
        data = read_csv (filename, path)
        if i == 0:
            accuracy = data[key].tolist()
            top_k = data[key_2].tolist()
            results_coord_cenario_2 = [float(i) for i in accuracy]
            results_coord_cenario_2 = [round (i, 3) for i in results_coord_cenario_2]
            results_coord = pd.DataFrame({'Top-k': top_k, 'Acuracia': results_coord_cenario_2})
            #a = pd.DataFrame( {'Acuracia': results_coord_cenario_2}, index=top_k,)

        if i == 1:
            accuracy = data[key].tolist()
            top_k = data[key_2].tolist()
            results_lidar_cenario_2 = [float(i) for i in accuracy]
            results_lidar_cenario_2 = [round (i, 3) for i in results_lidar_cenario_2]
            results_lidar = pd.DataFrame({'Top-k': top_k, 'Acuracia': results_lidar_cenario_2})
        if i == 2:
            accuracy = data[key].tolist()
            top_k = data[key_2].tolist()
            results_lidar_coord_cenario_2 = [float(i) for i in accuracy]
            results_lidar_coord_cenario_2 = [round (i, 3) for i in results_lidar_coord_cenario_2]
            results_lidar_coord = pd.DataFrame({'Top-k': top_k, 'Acuracia': results_lidar_coord_cenario_2})

    return results_coord, results_lidar, results_lidar_coord
    #return results_coord_cenario_2, results_lidar_cenario_2, results_lidar_coord_cenario_2
def read_all_results_invert_and_not_inverted_dataset():

    batool_coord_cenario_2, batool_lidar_cenario_2, batool_lidar_coord_cenario_2 = read_all_results_inverter_dataset('Batool', 2)
    wisard_coord_cenario_2, wisard_lidar_cenario_2, wisard_lidar_coord_cenario_2 = read_all_results_inverter_dataset('Wisard',2)
    ruseckas_coord_cenario_2, ruseckas_lidar_cenario_2, ruseckas_lidar_coord_cenario_2 = read_all_results_inverter_dataset('Ruseckas',2)

    batool_coord_scenario_1, batool_lidar_scenario_1, batool_lidar_coord_scenario_1 = read_all_results_inverter_dataset('Batool', 1)
    wisard_coord_scenario_1, wisard_lidar_scenario_1, wisard_lidar_coord_scenario_1 = read_all_results_inverter_dataset('Wisard', 1)
    ruseckas_coord_scenario_1, ruseckas_lidar_scenario_1, ruseckas_lidar_coord_scenario_1 = read_all_results_inverter_dataset('Ruseckas', 1)

    input_type = 'coord' # 'lidar', 'lidar_coord'
    if input_type == 'coord':
        plot_compare_accuracy_top_k_inverter_and_not_inverter_dataset(batool_coord_scenario_1,
                                                                      batool_coord_cenario_2,
                                                                 wisard_coord_scenario_1,
                                                                 wisard_coord_cenario_2,
                                                                 ruseckas_coord_scenario_1,
                                                                 ruseckas_coord_cenario_2,
                                                                      input_type)
    elif input_type == 'lidar':
        plot_compare_accuracy_top_k_inverter_and_not_inverter_dataset(batool_lidar_scenario_1,
                                                                      batool_lidar_cenario_2,
                                                                 wisard_lidar_scenario_1,
                                                                 wisard_lidar_cenario_2,
                                                                 ruseckas_lidar_scenario_1,
                                                                 ruseckas_lidar_cenario_2,
                                                                      input_type)
    elif input_type == 'lidar_coord':
        plot_compare_accuracy_top_k_inverter_and_not_inverter_dataset(batool_lidar_coord_scenario_1,
                                                                      batool_lidar_coord_cenario_2,
                                                                 wisard_lidar_coord_scenario_1,
                                                                 wisard_lidar_coord_cenario_2,
                                                                 ruseckas_lidar_coord_scenario_1,
                                                                 ruseckas_lidar_coord_cenario_2,
                                                                      input_type)

def plot_compare_accuracy_top_k_inverter_and_not_inverter_dataset(batool_coord_scenario_1,
                                                                  batool_coord_cenario_2,
                                                                 wisard_coord_scenario_1,
                                                                 wisard_coord_cenario_2,
                                                                 ruseckas_coord_scenario_1,
                                                                 ruseckas_coord_cenario_2,
                                                                  input_type):
    batool_s2 = batool_coord_cenario_2[batool_coord_cenario_2['Top-k']<=30]
    batool_s1 = batool_coord_scenario_1[batool_coord_scenario_1['Top-k']<=30]

    plt.figure(figsize=(10, 6))
    plt.plot(batool_s1['Top-k'], batool_s1['Acuracia'],
             label='Batool Coord Conf. a',  color='blue')
    plt.plot (batool_s2 ['Top-k'], batool_s2 ['Acuracia'],
              label='Batool Coord Conf. b', linestyle='--', color='blue', alpha=0.7)

    wisard_s2 = wisard_coord_cenario_2[wisard_coord_cenario_2['Top-k']<=30]
    wisard_s1 = wisard_coord_scenario_1[wisard_coord_scenario_1['Top-k']<=30]
    plt.plot(wisard_s1['Top-k'], wisard_s1['Acuracia'],
             label='Wisard Coord Conf. a', color='green')
    plt.plot (wisard_s2 ['Top-k'], wisard_s2 ['Acuracia'],
              label='Wisard Coord Conf. b', linestyle='--',color='green',alpha=0.7)

    ruseckas_s2 = ruseckas_coord_cenario_2[ruseckas_coord_cenario_2['Top-k']<=30]
    ruseckas_s1 = ruseckas_coord_scenario_1[ruseckas_coord_scenario_1['Top-k']<=30]
    plt.plot(ruseckas_s1['Top-k'], ruseckas_s1['Acuracia'],
             label='Ruseckas Coord Conf. a', color='red')
    plt.plot (ruseckas_s2 ['Top-k'], ruseckas_s2 ['Acuracia'],
              label='Ruseckas Coord Conf. b', color='red', linestyle='--', alpha=0.7 )
    plt.xlabel('Top-k')
    plt.legend()
    plt.ylabel('Acurácia')
    plt.xticks(batool_s1['Top-k'], fontsize=7)
    plt.grid(True, linestyle='--', alpha=0.5)
    #plt.title('Comparação de Acurácia entre Cenarios 1 e 2 \n Entrada: '+ input_type)
    plt.tight_layout()
    plt.savefig ('../results/' + input_type + '_compare_accuracy_inverter_and_not_inverted_dataset.png', dpi=300)


    a=0


    # Create a DataFrame for each scenarioa
    a = 0


def plot_wisard_results_compare_dataset_inverter_LOS_NLOS():
    input_type = ['coord', 'lidar', 'lidar_coord']
    type_conection = ['ALL','LOS', 'NLOS']

    scenario = 1 #Train S008 Test S009
    if scenario == 1:
        general_path = '../results/score/Wisard/split_dataset/'

        ALL_data = []
        for i in range(len(type_conection)):
            path = general_path + type_conection [0] + '/' + input_type [i] + '/'
            file_name = input_type[i]+'_results_top_k_wisard_'+type_conection[0]+'.csv'
            file = read_csv(file_name, path)
            ALL_data.append(file)

        LOS_data = []
        for i in range(len(type_conection)):
            path = general_path + type_conection [1] + '/' + input_type [i] + '/'
            file_name = input_type[i]+'_results_top_k_wisard_'+type_conection[1]+'.csv'
            file = read_csv(file_name, path)
            LOS_data.append(file)

        NLOS_data = []
        for i in range(len(type_conection)):
            path = general_path + type_conection [2] + '/' + input_type [i] + '/'
            file_name = input_type[i]+'_results_top_k_wisard_'+type_conection[2]+'.csv'
            file = read_csv(file_name, path)
            NLOS_data.append(file)

    All_data_DF = pd.DataFrame ({'coord': ALL_data [0] ['Acurácia'],
                                 'lidar': ALL_data [1] ['Acurácia'],
                                 'lidar_coord': ALL_data [2] ['Acurácia'],
                                 'top-k': ALL_data [0] ['top_k']})
    All_data_DF = All_data_DF [All_data_DF ['top-k'] < 31]
    LOS_data_DF = pd.DataFrame ({'coord': LOS_data [0] ['Acurácia'],
                                 'lidar': LOS_data [1] ['Acurácia'],
                                 'lidar_coord': LOS_data [2] ['Acurácia'],
                                 'top-k': LOS_data [0] ['top_k']})
    LOS_data_DF = LOS_data_DF [LOS_data_DF ['top-k'] < 31]
    NLOS_data_DF = pd.DataFrame ({'coord': NLOS_data [0] ['Acurácia'],
                                  'lidar': NLOS_data [1] ['Acurácia'],
                                  'lidar_coord': NLOS_data [2] ['Acurácia'],
                                  'top-k': NLOS_data [0] ['top_k']})
    NLOS_data_DF = NLOS_data_DF [NLOS_data_DF ['top-k'] < 31]

    scenario = 2
    if scenario == 2: #Train S009 Test S008
        general_path = '../results/inverter_dataset/score/Wisard/top-k/'

        coord_data = []
        for i in range(len(type_conection)):
            path = general_path + input_type[0] + '/' + type_conection [i] + '/'
            print(path)
            if i==0:
                file_name = 'accuracy_'+input_type[0]+'.csv'
            else:
                file_name = 'accuracy_'+input_type[0]+'_res_8_'+type_conection [i]+'.csv'
            file = read_csv(file_name, path)
            coord_data.append(file)

        lidar_data = []
        for i in range(len(type_conection)):
            path = general_path + input_type[1] + '/' + type_conection [i] + '/'
            if i==0:
                file_name = 'accuracy_'+input_type[1]+'.csv'
            else:
                file_name = 'accuracy_'+input_type[1]+'_'+type_conection [i]+'_thr_01.csv'
            file = read_csv(file_name, path)
            lidar_data.append(file)

        lidar_coord_data = []
        for i in range(len(type_conection)):
            path = general_path + input_type[2] + '/' + type_conection [i] + '/'
            if i==0:
                file_name = 'accuracy_'+input_type[2]+'.csv'
            else:
                file_name = 'accuracy_'+input_type[2]+'_res_8_'+type_conection [i]+'_thr_01.csv'
            file = read_csv(file_name, path)
            lidar_coord_data.append(file)


    coord_data_DF = pd.DataFrame({'ALL': coord_data[0]['score'],
                                'LOS': coord_data[1]['score'],
                                'NLOS': coord_data[2]['score'],
                                'top-k': coord_data[0]['top-k']})
    coord_data_DF = coord_data_DF [coord_data_DF ['top-k'] < 31]


    lidar_data_DF = pd.DataFrame({'ALL': lidar_data[0]['score'],
                                    'LOS': lidar_data[1]['score'],
                                    'NLOS': lidar_data[2]['score'],
                                    'top-k': lidar_data[0]['top-k']})
    lidar_data_DF = lidar_data_DF [lidar_data_DF ['top-k'] < 31]

    lidar_coord_data_DF = pd.DataFrame({'ALL': lidar_coord_data[0]['score'],
                                        'LOS': lidar_coord_data[1]['score'],
                                        'NLOS': lidar_coord_data[2]['score'],
                                        'top-k': lidar_coord_data[0]['top-k']})
    lidar_coord_data_DF = lidar_coord_data_DF [lidar_coord_data_DF ['top-k'] < 31]

    #plot results

    fig = plt.figure (figsize=(15, 8), tight_layout=True)

    ax1 = fig.add_subplot (131)
    ax1.plot(All_data_DF['coord'],  label='ALL', color='darkorange')
    ax1.plot(LOS_data_DF['coord'], label='LOS', color='steelblue')
    ax1.plot(NLOS_data_DF['coord'],  label='NLOS', color='forestgreen')

    ax1.plot(coord_data_DF['ALL'], label='ALL', color='darkorange', linestyle='--')
    ax1.plot(coord_data_DF['LOS'],  label='LOS', color='steelblue', linestyle='--')
    ax1.plot(coord_data_DF['NLOS'],  label='NLOS', color='forestgreen', linestyle='--')

    ax2 = fig.add_subplot (132)
    ax2.plot (All_data_DF['lidar'],  label='ALL', color='darkorange')
    ax2.plot (LOS_data_DF['lidar'],  label='LOS', color='steelblue')
    ax2.plot (NLOS_data_DF['lidar'],  label='NLOS', color='forestgreen')

    ax2.plot (lidar_data_DF['ALL'], label='ALL', color='darkorange', linestyle='--')
    ax2.plot (lidar_data_DF['LOS'], label='LOS', color='steelblue', linestyle='--')
    ax2.plot (lidar_data_DF['NLOS'], label='NLOS', color='forestgreen', linestyle='--')

    ax3 = fig.add_subplot (133)
    ax3.plot (All_data_DF['lidar_coord'],  label='ALL', color='darkorange')
    ax3.plot (LOS_data_DF['lidar_coord'],  label='LOS', color='steelblue')
    ax3.plot (NLOS_data_DF['lidar_coord'],  label='NLOS', color='forestgreen')

    ax3.plot (lidar_coord_data_DF['ALL'], label='ALL', color='darkorange', linestyle='--')
    ax3.plot (lidar_coord_data_DF['LOS'], label='LOS', color='steelblue', linestyle='--')
    ax3.plot (lidar_coord_data_DF['NLOS'], label='NLOS', color='forestgreen', linestyle='--')

    ax1.set_title('Coord')
    ax2.set_title('LiDAR')
    ax3.set_title('LiDAR + Coord')



    ax1.set_xlabel('Top-k')
    ax2.set_xlabel('Top-k')
    ax3.set_xlabel('Top-k')

    ax1.set_ylabel('Acurácia')
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    ax3.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax3.grid(True, linestyle='--', alpha=0.5)

    path_save = '../results/inverter_dataset/score/Wisard/'
    plt.savefig (path_save+ 'compare_results_inverter_LOS_NLOS.png', dpi=300, bbox_inches='tight')

                 #'score/Wisard/split_dataset/compare_results_inverter_LOS_NLOS.png', dpi=300)
    a=0


#####----------------------------------------------------



#compare_of_score_memorySize_and_processTime()
#plot_compare_accuracy_top_k()
calculate_and_save_rt_all_ref()
#plot_compare_score_and_rt_top_k()
#plot_results_dataset_inverter()
#plot_wisard_results_compare_dataset_inverter_LOS_NLOS()
#read_all_results_invert_and_not_inverted_dataset()
