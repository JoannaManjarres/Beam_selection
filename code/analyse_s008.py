import numpy as np
import csv
import seaborn as sns
import pandas as pd
import mimo_best_beams
import matplotlib.pyplot as plt

import seaborn as sns
import random

random.seed(0)


def read_beams_raymobtime_using_batool_method():
    data_path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_train.npz'
    y, num_classes = getBeamOutput (output_file=data_path)
    best_beam_index = []
    num_antennas_rx = 8
    for sample in range (y.shape [0]):
        best_beam_index.append (np.argmax (y [sample, :]))

    beam_index_rx = np.array (best_beam_index)

    tx_index = np.zeros ((y.shape [0]), dtype=int)
    rx_index = np.zeros ((y.shape [0]), dtype=int)

    for sample in range (len (beam_index_rx)):
        index_tx = best_beam_index [sample] // int (num_antennas_rx)
        index_rx = best_beam_index [sample] % int (num_antennas_rx)
        tx_index [sample] = index_tx
        rx_index [sample] = index_rx

    plot_distribution_beams_displot (tx_index, rx_index)
def read_beams_raymobtime(num_antennas_rx, path_of_data):

    #config_antenna = num_antennas_rx+'x'+num_antennas_tx
    #data_path = '/Users/Joanna/git/beam_selection_wisard/data/beams/Ailton/beam_output/beams_output_'+config_antenna+'.npz'
    #data_path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_train.npz'
    data_path = path_of_data
    beams = np.load(data_path)['output_classification']

    best_beam_index = []
    for sample in range(beams.shape[0]):
        best_beam_index.append(np.argmax(beams[sample, :]))

    beam_index_rx = np.array(best_beam_index)

    tx_index = np.zeros((beams.shape[0]), dtype=int)
    rx_index = np.zeros((beams.shape[0]), dtype=int)

    for sample in range(len(beam_index_rx)):
        index_tx = best_beam_index[sample] // int(num_antennas_rx)
        index_rx = best_beam_index[sample] % int(num_antennas_rx)
        tx_index[sample] = index_tx
        rx_index[sample] = index_rx

    return tx_index, rx_index, best_beam_index

def plot_distribution_beams_displot(beams_tx,
                                    beams_rx,
                                    path):
                                    #pp_folder, connection, set):

    #path = pp_folder + 'histogram/'+connection + '/'
    #plt.Figure(figsize=(32, 8))
    #sns.set_style('darkgrid')
    plot = sns.displot(x=beams_tx,
                y=beams_rx,
                row_order=range(2),
                col_order=range(32),
                binwidth=(2, 2),
                cmap='Blues',
                aspect=2.9, #10.67,
                height=3,#2.5,
                       cbar=True,
                       #cbar_kws={'panchor':(0.5,1.0)}
                )

    #plt.title("Beams distribuition (Tx-Rx) ["+set+"]")
    plt.xlabel("Indices dos feixes no Tx", font='Times New Roman', fontsize=14)
    plt.ylabel("Indices dos feixes no Rx", font='Times New Roman', fontsize=14)
    plt.gca().invert_yaxis()

    plt.subplots_adjust(left=0.08)
    #plt.subplots_adjust(right=5)
    plt.subplots_adjust(bottom=0.179)
    plt.subplots_adjust(top=0.879)
    #plot.fig.set_figwidth(10)
    #plot.fig.set_figheight(6)
    #name = path+"Beams_distribution_"+set+".png"
    #print(name)
    plt.savefig(path, transparent=False, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.cla ()
    plt.close()

    #plt.show()

def generated_beams_output_from_ray_tracing():

    inputPath = '../data/beams_output/dataset_s008/beam_output_generate_rt_s008/ray_tracing_data_s008_carrier60GHz/rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e'
    insiteCSVFile = '../data/coord/CoordVehiclesRxPerScene_s008.csv'
    numEpisodes = 2086  # total number of episodes
    outputFolder = '../data/beams_output/dataset_s008/beam_output_generate_rt_s008/testes/'

    mimo_best_beams.processBeamsOutput(csvFile=insiteCSVFile, num_episodes=numEpisodes, outputFolder=outputFolder, inputPath=inputPath)

def read_beams_output_generated_by_rt():
    path = '../data/beams_output/dataset_s008/beam_output_generate_rt_s008/'
    beam_output = np.load(path + "beams_output_8x32.npz", allow_pickle=True)['output_classification']
    print("\t\tGeracao de Beams pelo Ray-tracing")
    index_beams = calculate_index_beams(beam_output)

    index_beam_output_train, index_beam_output_validation = separete_beam_output_train_test(index_beams)
    index_beam_train_str = [str(i) for i in index_beam_output_train]
    index_beam_validation_str = [str(i) for i in index_beam_output_validation]

    return index_beam_train_str, index_beam_validation_str, index_beam_output_train, index_beam_output_validation, index_beams

def read_beams_output_from_baseline():
    path = '../data/beams_output/beam_output_baseline_raymobtime_s008/'
    beam_output_train = np.load(path + "beams_output_train.npz", allow_pickle=True)#['output_classification']
    key_beam_output_train = list(beam_output_train.keys())
    beam_output_train = beam_output_train[key_beam_output_train[0]]

    beam_output_validation = np.load (path + "beams_output_validation.npz", allow_pickle=True)
    key_index_beam_validation = list(beam_output_validation.keys())
    beam_output_validation = beam_output_validation[key_index_beam_validation[0]]

    #print("\t\tBeams generation from Baseline")
    #print("------------------------------------------------------------")
    #print ("\t\tGerando Beams do Baseline...")
    index_beams_train = calculate_index_beams(beam_output_train)
    index_beam_train_str = [str (i) for i in index_beams_train]
    index_beams_test = calculate_index_beams(beam_output_validation)
    index_beam_test_str = [str(i) for i in index_beams_test]

    #index_beam_output_train, index_beam_output_validation = separete_beam_output_train_test(index_beams)

    return index_beam_train_str, index_beam_test_str, index_beams_train, index_beams_test

def read_beams_used_in_article():
    path = '../data/beams_output/dataset_s008/beam_output_s008_article/'
    beam_output_train = np.load(path + "index_beams_combined_train.npz", allow_pickle=True)
    keys_train = list(beam_output_train.keys())
    index_beam_article_train = beam_output_train[keys_train[0]]
    index_beam_article_train_str = [str(i) for i in index_beam_article_train]

    beam_output_test = np.load(path + "index_beams_combined_test.npz", allow_pickle=True)
    keys_test = list(beam_output_test.keys())
    index_beam_article_test = beam_output_test[keys_test[0]]
    index_beam_article_test_str = [str(i) for i in index_beam_article_test]

    return index_beam_article_train_str, index_beam_article_test_str, index_beam_article_train, index_beam_article_test

def stats_index_beams_article():
    _,_, index_beam_article_train, index_beam_article_test = read_beams_used_in_article()
    labels, counts = np.unique (index_beam_article_train, return_counts=True)

    nro_max_beam = np.max(counts)
    arg_max_beam = np.argmax(counts)

    labels_tst, counts_test = np.unique (index_beam_article_test, return_counts=True)

    nro_max_beam_test = np.max (counts)
    arg_max_beam_test = np.argmax (counts)


    plt.bar (labels, counts, align='center')
    plt.title ('Best beam pair index')
    plt.ylabel ('Probability')
    plt.xlabel ('Data')
    plt.show ()
    a=0

def read_beams_output_used_in_ref_17():
    path_ref_17 = '../data/beams_output/dataset_s008/beam_output_ref_17/'
    filename = "index_beams_train.npz"

    beam_output_ref_17 = np.load(path_ref_17 + filename, allow_pickle=True)
    keys_train_ref_17 = list(beam_output_ref_17.keys())
    index_beam_train_ref_17 = beam_output_ref_17[keys_train_ref_17[0]]
    index_beam_train_ref_17_str = [str(i) for i in index_beam_train_ref_17]

    filename = "index_beams_val.npz"
    beam_output_ref_17 = np.load(path_ref_17 + filename, allow_pickle=True)
    keys_validation_ref_17 = list(beam_output_ref_17.keys())
    index_beam_validation_ref_17 = beam_output_ref_17[keys_validation_ref_17[0]]
    index_beam_validation_ref_17_str = [str(i) for i in index_beam_validation_ref_17]

    filename = "index_beams_test.npz"
    beam_output_test = np.load(path_ref_17 + filename, allow_pickle=True)
    keys_test_ref_17 = list(beam_output_test.keys())
    index_beam_test_ref17 = beam_output_test[keys_test_ref_17[0]]
    index_beam_test_ref17_str = [str(i) for i in index_beam_test_ref17]

    return index_beam_train_ref_17_str, index_beam_validation_ref_17_str, index_beam_test_ref17_str

def read_beams_output_generated_by_batool():
    path = '../data/beams_output/dataset_s008/beam_output_batool/'
    filename = 'y_train.npz'
    beam_output = np.load (path + filename, allow_pickle=True)
    keys = list (beam_output.keys ())
    index_beam_batool_train = beam_output [keys [0]]
    print("\t\tGeracao de Beams pelo Batool")
    index_beams_train = np.argmax(index_beam_batool_train, axis=1).tolist()
    index_beams_batool_train = [str(x) for x in index_beams_train]

    filename = 'y_validation.npz'
    beam_output = np.load (path + filename, allow_pickle=True)
    keys = list (beam_output.keys ())
    index_beam_batool_val = beam_output [keys [0]]
    index_beams_val = np.argmax(index_beam_batool_val, axis=1).tolist()
    index_beams_batool_val = [str(x) for x in index_beams_val]

    filename = 'y_test.npz'
    beam_output_test = np.load (path + filename, allow_pickle=True)
    keys = list (beam_output_test.keys ())
    index_beam_batool_test = beam_output_test [keys [0]]
    index_beams_test = np.argmax(index_beam_batool_test, axis=1).tolist()
    index_beams_batool_test = [str(x) for x in index_beams_test]

    return index_beams_batool_train, index_beams_batool_val, index_beams_batool_test

def read_beams_output_ailton():
    path = '../data/beams_output/dataset_s008/beam_output_ailton/'
    filename = 'best_beam_index.npz'

    beam_output = np.load (path + filename, allow_pickle=True)
    keys = list (beam_output.keys())
    index_beam_ailton = beam_output[keys[0]]
    index_beam_ailton_str = [str (i) for i in index_beam_ailton]

    index_train, index_test = separete_beam_output_train_test(index_beam_ailton_str)
    index_train = index_train.tolist()
    index_test = index_test.tolist()

    return index_train, index_test


def calculate_index_beams(beam_output):
    # calculate the index of the best beam
    tx_size = beam_output.shape[2]
    #print(beam_output.shape)

    # Reshape beam pair index
    num_classes = beam_output.shape[1] * beam_output.shape[2]
    beams = beam_output.reshape(beam_output.shape[0], num_classes)


    # Beams distribution
    best_beam_index = []
    for sample in range(beams.shape[0]):
        best_beam_index.append(np.argmax(beams[sample, :]))


    return(best_beam_index)

def separete_beam_output_train_test(index_beam):
    # data processed by Ailton code
    #path = '../data/beams_output/dataset_s008/'
    #input_cache_file = np.load (path + "beam_output_generate_rt_s008/beams_output_8x32.npz", allow_pickle=True)
    #key = list (input_cache_file.keys ())
    #beam_output_rt = input_cache_file [key [0]].astype (str)
    #index_beam = calculate_index_beams (beam_output_rt)

    #index_beam = read_beams_output_generated_by_rt()

    filename = '../data/coord/CoordVehiclesRxPerScene_s008.csv'
    limit_ep_train = 1564

    with open (filename) as csvfile:
        reader = csv.DictReader (csvfile)
        number_of_rows = len (list (reader))

    all_info_coord_val = np.zeros ([11194, 6], dtype=object)

    with open (filename) as csvfile:
        reader = csv.DictReader (csvfile)
        cont = 0
        for row in reader:
            if row ['Val'] == 'V':
                all_info_coord_val [cont] = int (row ['EpisodeID']), float (row ['x']), float (row ['y']), float (row ['z']), row ['LOS'], str(index_beam [cont])
                cont += 1

    # all_info_coord = np.array(all_info_coord)

    coord_train = all_info_coord_val [(all_info_coord_val [:, 0] < limit_ep_train + 1)]
    coord_test = all_info_coord_val [(all_info_coord_val [:, 0] > limit_ep_train)]

    index_beam_output_train = coord_train[:,5]
    index_beam_output_test = coord_test[:,5]

    return index_beam_output_train, index_beam_output_test #, coord_train, coord_test

def plot_hist_prob_beam(beam, title):#, set, pp_folder, connection, x_label='indice dos beams'):

    #path = pp_folder + 'histogram/'+connection + '/'
    #print(path)
    path = '../analyses/'
    plt.rcParams.update({'font.size': 8})
    plt.rcParams.update({'figure.subplot.bottom':0.146})# 0.127, 0.9, 0.9]
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(beam,
            bins=256,
            density=True,
            color="steelblue",
            ec="steelblue")
    # ax.plot(data, pdf_lognorm)
    ax.set_ylabel('P', fontsize=12, rotation=0, color='steelblue', fontweight='bold')
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.set_xlabel('Beam pair index', color='steelblue', fontweight='bold' )
    #ax.xaxis.set_label_coords(1.05, -0.025)
    plt.grid(axis='y', alpha=0.9, color='white')
    ax.set_facecolor('#EEEEF5')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    #plt.tick_params(axis='x', colors='red', direction='out', length=7, width=2)
    #
    title = title
    plt.title(title)

    #plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))


    plt.savefig(path+"histogram_s008_prob_all_Beams_combined.png", bbox_inches='tight')
    plt.show()

def plot_histogram_beam(index_beam, title):#, user, color, connection, set, pp_folder, config):
    color = 'blue'
    #print("Histograma dos indices dos Beams do ", user ," [" + connection + "] em config ", config , " \n usando dados de ",  set)

    #path = pp_folder + '/histogram/'+connection + '/'
    #title = 'Distribuicao dos indices dos Beams do ' + user +' [' + connection + '] em config ' + config + ' \n usando dados de ' + set
    title = title
    sns.set(style='darkgrid')
    sns.set(rc={'figure.figsize': (8, 4)})
    plot = sns.histplot(data=index_beam,
                        bins=256,
                        stat='frequency',
                        #color='color',
                        legend=False)
    plt.title(title, fontweight='bold')
    plt.xlabel('Indices')
    plt.ylabel('Frequência')
    plt.legend(bbox_to_anchor=(1.05, 1),
               borderaxespad=0,
               loc='upper left',
               title='Amostras',
               labels=[str(len(index_beam))])
    # plot.fig.set_figwidth(4)
    # plot.fig.set_figheight(8)
    plt.subplots_adjust(right=0.786, bottom=0.155)

    #name = path + 'Histogram_dos_Beams_' + user + '_' + connection + '_' + set + '.png'
    #plt.savefig(name, transparent=False, dpi=300)
    plt.show()

def relation_coord_with_beams_Plot2D():
    _,_, all_data_train, all_data_test = separete_beam_output_train_test()
    data = pd.DataFrame(all_data_train, columns=['EpisodeID', 'x', 'y', 'z', 'LOS', 'index_beams'])

    sns.set (style='darkgrid')
    plot = sns.relplot (data=data,
                        x='x',
                        y='y',
                        kind='scatter',
                        hue='index_beams',
                        # size='combinedBeams',
                        # sizes=(12,200),
                        alpha=0.5,
                        palette='dark',
                        legend=False)

    # sns.set(rc={'figure.figsize': (60, 5)})

    plot.fig.suptitle ('Distribuicao dos indices dos Beams \n relativo à posicao usando dados de Train',
                        fontweight='bold')
    plot.fig.subplots_adjust (top=0.90)
    plot.fig.set_figwidth (6)
    plot.fig.set_figheight (8)
    #name = path + 'relation_coord_with_combined_beams_' + connection + '_' + set + '.png'
    #plt.savefig (name, transparent=False, dpi=300)
    plt.show ()

def compare_beamoutput_matrices_from_RT():
    path = '../data/beams_output/dataset_s008/beam_output_generate_rt_s008/testes/'
    beam_output_1 = np.load(path + "beams_output_8x32_2.npz", allow_pickle=True)['output_classification']
    index_beams_1 = calculate_index_beams(beam_output_1)
    index_beam_output_train_1, index_beam_output_validation_1 = separete_beam_output_train_test(index_beams_1)

    beam_output_2 = np.load(path + "beams_output_8x32_1.npz", allow_pickle=True) ['output_classification']
    index_beams_2 = calculate_index_beams(beam_output_2)
    index_beam_output_train_2, index_beam_output_validation_2 = separete_beam_output_train_test(index_beams_2)

    index_beams_baseline_train, index_beams_baseline_test, _, _ = read_beams_output_from_baseline ()

    print('1rst set \t 2nd set \t baseline set')
    print(len(index_beam_output_train_1),'\t\t', len(index_beam_output_train_2),'\t\t', len(index_beams_baseline_train))

    RT1_RT2 = 0
    for i in range(len(index_beam_output_train_1)):
        if (index_beam_output_train_2[i] != index_beams_baseline_train[i]):
            RT1_RT2 += 1

    print('Dados diferentes de treino', RT1_RT2)

    RT1_val_RT2_val = 0
    for i in range(len(index_beam_output_validation_1)):
        if (index_beam_output_validation_1[i]!= index_beam_output_validation_2[i]):
            RT1_val_RT2_val += 1
    print('Dados diferentes de validacao', RT1_val_RT2_val)

    fig, ax = plt.subplots (figsize=(8, 4))

    ax.hist (index_beams_baseline_train, bins=50, density=True, color="red", alpha=0.5,
             label='Baseline')  # , ec="steelblue")
    ax.hist (index_beam_output_train_1, bins=50, density=True, color="green", alpha=0.5,
             label='Beams Generated by RT')  # , ec="steelblue")
    # ax.plot(data, pdf_lognorm)
    ax.set_ylabel ('P', fontsize=12, rotation=0, color='steelblue', fontweight='bold')
    ax.set_xlabel ('Beam pair index', color='steelblue', fontweight='bold', fontsize='12')
    # plt.grid (axis='y', alpha=0.9, color='white')
    plt.legend ()
    plt.show ()

def analyse_index_beams_s008():
    index_beam_ailton_train, index_beam_ailton_test = read_beams_output_ailton()
    index_beam_RT_train, index_beam_RT_validation,_,_,_ = read_beams_output_generated_by_rt ()
    index_beams_baseline_train, index_beams_baseline_test,_,_ = read_beams_output_from_baseline()
    index_beam_article_train, index_beam_article_test,_,_ = read_beams_used_in_article()
    index_beam_ref_17_train = read_beams_output_used_in_ref_17()
    index_beam_batool_train, index_beam_batool_val = read_beams_output_generated_by_batool()

    if (len(index_beam_RT_train) != len(index_beam_article_train) != len(index_beams_baseline_train) != len(index_beam_ref_17_train)):
        print('Diferentes')
    else:
        print('Quantidade de amostras Iguais')


    RT_baseline=0
    RT_article=0
    RT_ailton=0
    RT_ref_17=0
    RT_batool = 0

    ref_17_baseline=0
    ref_17_article = 0
    ref_17_ailton = 0
    ref_17_batool =0

    article_baseline=0
    article_ailton = 0
    article_batool = 0

    baseline_ailton = 0
    baseline_batool = 0

    ailton_batool = 0

    article_article = 0
    RT_RT = 0
    baseline_baseline = 0
    ailton_ailton = 0
    ref_17_ref_17 = 0
    batool_batool=0

    for i in range (len (index_beam_RT_train)):
        if (index_beam_RT_train [i] != index_beams_baseline_train [i]):
            RT_baseline += 1
        if (index_beam_RT_train [i] != index_beam_article_train [i]):
            RT_article += 1
        if (index_beam_RT_train [i] != index_beam_ailton_train[i]):
            RT_ailton += 1
        if (index_beam_RT_train [i] != index_beam_ref_17_train [i]):
            RT_ref_17 += 1
        if (index_beam_RT_train [i] != index_beam_batool_train [i]):
            RT_batool += 1

        if (index_beam_ref_17_train [i] != index_beam_article_train [i] ):
            ref_17_article += 1
        if (index_beam_ref_17_train [i] != index_beams_baseline_train [i]):
            ref_17_baseline += 1
        if (index_beam_ref_17_train[i] != index_beam_ailton_train[i]):
            ref_17_ailton += 1
        if (index_beam_ref_17_train[i] != index_beam_batool_train[i]):
            ref_17_batool += 1

        if (index_beam_article_train [i] != index_beams_baseline_train [i]):
            article_baseline += 1
        if (index_beam_article_train[i] != index_beam_ailton_train[i] ):
            article_ailton += 1
        if (index_beam_ailton_train[i] != index_beams_baseline_train[i]):
            article_baseline += 1
        if (index_beam_article_train[i] != index_beam_batool_train[i]):
            article_batool += 1

        if (index_beams_baseline_train[i] != index_beam_ailton_train[i]):
            baseline_ailton += 1
        if (index_beams_baseline_train[i] != index_beam_batool_train[i]):
            baseline_batool += 1

        if (index_beam_ailton_train[i] != index_beam_batool_train[i]):
            ailton_batool += 1

        if (index_beam_article_train[i] != index_beam_article_train[i]):
            article_article += 1
        if (index_beam_RT_train[i] != index_beam_RT_train[i]):
            RT_RT += 1
        if (index_beams_baseline_train[i] != index_beams_baseline_train[i]):
            baseline_baseline += 1
        if (index_beam_ailton_train[i] != index_beam_ailton_train[i]):
            ailton_ailton += 1
        if (index_beam_ref_17_train[i] != index_beam_ref_17_train[i]):
            ref_17_ref_17 += 1
        if (index_beam_batool_train[i] != index_beam_batool_train[i]):
            batool_batool += 1


    print ('total amostras:', len (index_beam_RT_train))
    print ('Amostras diferentes:')

    print(           '\t\t\t RT \t baseline \t Article \t Ref17 \t Ailton \t Batool\n'
          'RT \t\t\t', RT_RT,       '\t\t', RT_baseline,    '\t\t',   RT_article,     '\t\t',   RT_ref_17, '\t',         RT_ailton,       '\t\t', RT_batool,'\n'
          'Ref17 \t\t',RT_ref_17,   '\t', ref_17_baseline,  '\t\t\t', ref_17_article, '\t\t',   ref_17_ref_17, '\t\t',   ref_17_ailton,   '\t\t', ref_17_batool,'\n'
          'baseline \t',RT_baseline,'\t', baseline_baseline,'\t\t\t', article_baseline, '\t\t', ref_17_baseline, '\t\t', baseline_ailton, '\t\t', baseline_batool,'\n'
          'Article \t', RT_article, '\t', article_baseline, '\t\t',   article_article, '\t\t\t', ref_17_article,'\t',    article_ailton,  '\t\t', article_batool,'\n'
          'Ailton \t\t', RT_ailton, '\t', baseline_ailton,  '\t\t',   article_ailton,  '\t\t',   ref_17_ailton, '\t',    ailton_ailton,   '\t\t\t', ailton_batool, '\n'
          'Batool \t\t', RT_batool,  '\t', baseline_batool,  '\t\t\t',   article_batool,  '\t\t',   ref_17_batool, '\t\t',    ailton_batool,    '\t\t', batool_batool
          )



    fig, ax = plt.subplots (figsize=(8, 4))

    #ax.hist (index_beam_RT_train, bins=256, density=True, color="steelblue", label='RT')  # , ec="steelblue")
    ax.hist (index_beams_baseline_train, bins=256, density=True, color="red", alpha=0.5, label='Baseline')  # , ec="steelblue")
    #ax.hist (index_beam_article_train, bins=256, density=True, color="blue", alpha=0.5, label='article')  # , ec="steelblue")
    ax.hist(index_beam_ref_17_train, bins=256, density=True, color="green", alpha=0.5, label='Ref17')  # , ec="steelblue")
    # ax.plot(data, pdf_lognorm)
    ax.set_ylabel('P', fontsize=12, rotation=0, color='steelblue', fontweight='bold')
    ax.set_xlabel('Beam pair index', color='steelblue', fontweight='bold')
    # plt.grid (axis='y', alpha=0.9, color='white')
    plt.legend()
    plt.show()


def getBeamOutput(output_file):
    thresholdBelowMax = 6
    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]

    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y = beamsLogScale(y, thresholdBelowMax)


    return y,num_classes

def beamsLogScale(y, thresholdBelowMax):
    y_shape = y.shape  # shape is (#,256)

    for i in range (0, y_shape [0]):
        thisOutputs = y [i, :]
        logOut = 20 * np.log10 (thisOutputs + 1e-30)
        minValue = np.amax (logOut) - thresholdBelowMax
        zeroedValueIndices = logOut < minValue
        thisOutputs [zeroedValueIndices] = 0
        thisOutputs = thisOutputs / sum (thisOutputs)
        y [i, :] = thisOutputs
    return y

def plot_distribution_of_beams():
    path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_train.npz'
    tx_index_train, rx_index_train, best_beam_index_train = read_beams_raymobtime (num_antennas_rx=8, path_of_data=path)
    data_path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_validation.npz'
    tx_index_val, rx_index_val, best_beam_index_val = read_beams_raymobtime (num_antennas_rx=8, path_of_data=data_path)

    tx_index = np.concatenate ((tx_index_train, tx_index_val), axis=0)
    rx_index = np.concatenate ((rx_index_train, rx_index_val), axis=0)

    path_to_save = '../analyses/histogram_s008.png'
    plot_distribution_beams_displot (tx_index, rx_index, path_to_save)

    data_path = '../data/beams_output/beam_output_baseline_raymobtime_s009/beams_output_test.npz'
    tx_index_s009, rx_index_s009, best_beam_index_s009 = read_beams_raymobtime (num_antennas_rx=8, path_of_data=data_path)

    path_to_save = '../analyses/histogram_s009.png'
    plot_distribution_beams_displot (tx_index_s009, rx_index_s009, path_to_save)

def plot_beams_with_coord():
    type_connection = 'NLOS'
    filename = '../data/coord/CoordVehiclesRxPerScene_s008.csv'
    limit_ep_train = 1564
    data = pd.read_csv(filename)
    valid_data = data[data['Val'] == 'V']
    train_data = valid_data[valid_data['EpisodeID'] <= limit_ep_train]

    path = '../data/beams_output/beam_output_baseline_raymobtime_s008/beams_output_train.npz'
    tx_index_train, rx_index_train, best_beam_index_train = read_beams_raymobtime (num_antennas_rx=8, path_of_data=path)

    train_data.insert(6, 'index_beams', best_beam_index_train)
    train_data.insert(7, 'tx_index', tx_index_train)
    train_data.insert(8, 'rx_index', rx_index_train)

    train_data_LOS = train_data [train_data ['LOS'] == 'LOS=1']
    train_data_NLOS = train_data [train_data ['LOS'] == 'LOS=0']
    type_index = 'tx_index'

    if type_connection == 'LOS':
        data_plot = train_data_LOS
    if type_connection == 'NLOS':
        data_plot = train_data_NLOS
    if type_connection == 'ALL':
        data_plot = train_data

    name = type_index + '_'+type_connection



    sns.set (style='darkgrid')
    sns.set (font_scale=1.3)
    #sns.set_context ("paper", rc={"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14})
    #fig, ax = plt.subplots(figsize=(11.7, 8.27))
    #fig.set_size_inches (11.7, 8.27)

    sns.scatterplot(data=(745,550))
    g = sns.relplot (data=data_plot,
                        x='x',
                        y='y',
                        kind='scatter',
                        hue=type_index,
                        # size='combinedBeams',
                        # sizes=(12,200),
                        alpha=0.5,
                        palette='dark',
                        legend=False)
                        #aspect=15/23)
    plt.scatter (745, 550, color='red', s=200, marker='v')
    plt.text (744.5, 555, 'Tx', fontsize=16, color='red')
    g.figure.set_size_inches (7.5, 10.5)
    plt.savefig('../analyses/'+name+'_distribution_with_position_s008.png', dpi=300, bbox_inches='tight')
    #plt.show()

    g = sns.relplot (data=train_data,
                     x='x',
                     y='y',
                     kind='scatter',
                     hue='tx_index',
                     # size='combinedBeams',
                     # sizes=(12,200),
                     alpha=0.5,
                     palette='dark',
                     legend=True)
    # aspect=15/23)
    plt.scatter (745, 550, color='red', s=200, marker='v')
    plt.text (744.5, 555, 'Tx', fontsize=16, color='red')
    g.figure.set_size_inches (7.5, 10.5)
    plt.savefig ('../analyses/tx_beam_distribution_with_position_s008.png', dpi=300, bbox_inches='tight')


#compare_beamoutput_matrices_from_RT()
#generated_beams_output_from_ray_tracing()
#stats_index_beams_article()
#plot_distribution_of_beams()
plot_beams_with_coord()




