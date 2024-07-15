import pandas as pd
import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt

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




def plot_results_top_k(top_k, input, filename,
                       accuracy_ref_17,
                       accuracy_wisard_s008_s009,
                       accuracy_batool,
                       label_ref_17,
                       label_wisard_s008_s009,
                       label_batool):

    style_of_line_lidar = 'dashed'  # 'solid'#
    style_of_line_s008_s009 = 'solid'
    type_of_marker_s008_s009 = 'o'
    type_of_marker_lidar = 'x'
    size_of_marker_lidar = 3
    width_of_line_lidar = 1

    color_ref_17 = 'blue'
    color_wisard_s008_s009 = 'red'#'seagreen'

    color_ref_17_s008 = 'pink'
    color_wisard_s008 = 'goldenrod'
    color_wisard_s009 = 'blue'


    #sns.set()
    plt.figure()
    plt.plot (top_k,
              accuracy_wisard_s008_s009,
              color=color_wisard_s008_s009,
              marker=type_of_marker_s008_s009,
              linestyle=style_of_line_s008_s009,
              linewidth=width_of_line_lidar,
              markersize=size_of_marker_lidar,
              label=label_wisard_s008_s009)
    for i, v in enumerate (accuracy_wisard_s008_s009):
        if input == 'coord':
            plt.text (top_k [i]+0.8, v-0.03 , str (v), color=color_wisard_s008_s009, size=8)
        elif input == 'lidar':
            plt.text (top_k[i], v-0.065, str (v), color=color_wisard_s008_s009, size=8)

    plt.plot(top_k,
             accuracy_ref_17,
             color=color_ref_17,
             marker=type_of_marker_s008_s009,
             linestyle=style_of_line_s008_s009,
             linewidth=width_of_line_lidar, markersize=size_of_marker_lidar, label=label_ref_17, alpha=0.5,)
    for i, v in enumerate (accuracy_ref_17):
        if input == 'coord':
            plt.text (top_k[i], v+0.03 , str (v), color=color_ref_17, size=8)
        elif input == 'lidar':
            plt.text (top_k[i], v+0.038, str (v), color=color_ref_17, size=8)

    plt.plot(top_k,
             accuracy_batool,
             color='teal',
             marker=type_of_marker_s008_s009,
             linestyle=style_of_line_s008_s009,
             linewidth=width_of_line_lidar,
             markersize=size_of_marker_lidar,
             label=label_batool)
    for i, v in enumerate (accuracy_batool):
        if input == 'coord':
            plt.text (top_k [i], v-0.0480, str (v), color='teal', size=8)
        elif input == 'lidar':
            plt.text (top_k [i], v-0.038, str (v), color='teal', size=8)



    plt.title('Comparacao entre as referencias e a WiSARD \n Acuracia Top-K dos dados '+ input , color='steelblue', size=14, fontweight='bold')
    #plt.xticks(top_k)
    plt.xlabel('Top-k', color='steelblue', size=14, fontweight='bold')
    #plt.yscale('linear')
    plt.ylim([0, 1.1])
    plt.xlim([0, 55])
    plt.grid()
    plt.ylabel('Accuracy', color='steelblue', size=14, fontweight='bold')
    plt.legend()

    #plt.grid(False)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

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
    input = 'coord'
    path_to_save = '../results/accuracy/8x32/accuracy_new_labels/'

    reference = 'batool'
    filename = 'acuracia_'+reference+'_'+input+'_top_k.csv'
    top_k, accuracy_batool = read_csv_file (input=input, filename=filename)

    reference = 'ruseckas'
    filename = 'acuracia_' + reference + '_' + input + '_top_k.csv'
    _, accuracy_ruseckas = read_csv_file (input=input, filename=filename)

    reference = 'wisard'
    filename = 'acuracia_' + reference + '_' + input + '_top_k.csv'
    _, accuracy_wisard = read_csv_file (input=input, filename=filename)

    figure_name = path_to_save + 'campare_top_k_accuracy_'+input+'.png'

    if input == 'coord':
        label_ruseckas = 'COORD: Ruseckas [CNN]'
        label_batool = 'COORD: Batool [DNN]'
        label_wisard = 'COORD: UFRJ [WiSARD]'
    elif input == 'lidar':
        label_ruseckas = 'COORD: Ruseckas [CNN]'
        label_batool = 'COORD: Batool [DNN]'
        label_wisard = 'COORD: UFRJ [WiSARD]'
    elif input == 'coord_lidar':
        label_ruseckas = 'COORD + LIDAR: Ruseckas [MLP+CNN]'
        label_batool = 'COORD + LIDAR: Batool [DNN]'
        label_wisard = 'COORD + LIDAR 2D: UFRJ [WiSARD]'

    plot_results_top_k (top_k,
                        input,
                        figure_name,
                        accuracy_ruseckas,
                        accuracy_wisard,
                        accuracy_batool,
                        label_ruseckas,
                        label_wisard,
                        label_batool)



    if input == 'coord':
        filename ='acuracia_batool_[coord]_top_k.csv'
        top_k, accuracy_batool_coord = read_csv_file (input=input, filename=filename)


        filename = 'acuracia_ref_17_coord_top_k.csv'
        _, accuracy_coord_ref_17 = read_csv_file (input=input, filename=filename)

        filename = 'acuracia_wisard_coord_top_k.csv'
        _, accuracy_wisard_coord = read_csv_file (input=input, filename=filename)

        accuracy_batool = accuracy_batool_coord
        accuracy_ref_17 = accuracy_coord_ref_17
        accuracy_wisard = accuracy_wisard_coord

        figure_name = path_to_save + 'campare_top_k_accuracy_coord.png'
        label_ref_17 = 'COORD: Ruseckas [MLP]'
        label_batool = 'COORD: Batool [DNN 1D]'
        label_wisard = 'COORD: UFRJ [WiSARD]'

    elif input == 'lidar':
        filename = 'acuracia_batool_[lidar]_top_k.csv'
        top_k, accuracy_batool_lidar = read_csv_file(input=input, filename=filename)

        filename = 'acuracia_ref_17_lidar_top_k.csv'
        _, accuracy_ref_17_lidar = read_csv_file(input=input, filename=filename)

        filename = 'acuracia_wisard_LiDAR_2D_+_Rx_Term_SVar_top_k.csv'
        _, accuracy_wisard_lidar = read_csv_file(input=input, filename=filename)

        accuracy_batool = accuracy_batool_lidar
        accuracy_ref_17 = accuracy_ref_17_lidar
        accuracy_wisard = accuracy_wisard_lidar

        figure_name = path_to_save + 'campare_top_k_accuracy_lidar.png'
        label_ref_17 = 'COORD: Ruseckas [CNN]'
        label_batool = 'COORD: Batool [DNN]'
        label_wisard = 'COORD: UFRJ [WiSARD]'

    elif input == 'coord_lidar':
        filename = 'acuracia_batool_[coord_lidar]_top_k.csv' #TODO: falta rodar el modelo con estas entradas
        #top_k, accuracy_batool_coord_lidar = read_csv_file(input='coord_lidar', filename=filename)

        filename = 'acuracia_ref_17_lidar_coord_top_k.csv'
        _, accuracy_ref_17_lidar_coord = read_csv_file(input='coord_lidar', filename=filename)

        filename = 'acuracia_wisard_LiDAR_2D_+_Rx_Term_+_Coord_16_SVar_top_k.csv'
        _, accuracy_wisard_lidar_coord = read_csv_file(input='lidar_+_coord/top_k', filename=filename)

        #accuracy_batool = accuracy_batool_coord_lidar
        accuracy_ref_17 = accuracy_ref_17_lidar_coord
        accuracy_wisard = accuracy_wisard_lidar_coord

        figure_name = path_to_save + 'campare_top_k_accuracy_coord_lidar.png'
        label_ref_17 = 'COORD + LIDAR: Ruseckas [MLP+CNN]'
        label_batool = 'COORD + LIDAR: Batool [DNN]'
        label_wisard = 'COORD + LIDAR 2D: UFRJ [WiSARD]'

    plot_results_top_k (top_k,
                        input,
                        figure_name,
                        accuracy_ref_17,
                        accuracy_wisard,
                        accuracy_batool,
                        label_ref_17,
                        label_wisard,
                        label_batool)
    a=0


plot_compare_accuracy_top_k()