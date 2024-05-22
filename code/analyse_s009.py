import numpy as np
import csv
import mimo_best_beams
import matplotlib.pyplot as plt
import seaborn as sns

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌃' to toggle the breakpoint.

def generated_beams_output_from_ray_tracing():

    inputPath = '../data/ray_tracing_data_s009_carrier60GHz/rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e'
    insiteCSVFile = '../data/coord/CoordVehiclesRxPerScene_s009.csv'
    numEpisodes = 2000  # total number of episodes
    outputFolder = '../data/beams_output/beams_output_generate_s009/'

    mimo_best_beams.processBeamsOutput(csvFile=insiteCSVFile, num_episodes=numEpisodes, outputFolder=outputFolder, inputPath=inputPath)
def read_beams_output_generated_by_rt():
    path = '../data/beams_output/beams_output_generate_s009/'
    beam_output = np.load(path + "beams_output_8x32.npz", allow_pickle=True)['output_classification']

    index_beam_pair= calculate_index_beams (beam_output)

    return beam_output , index_beam_pair
def calculate_index_beams(beam_output):


    # calculate the index of the best beam
    tx_size = beam_output.shape [2]

    # Reshape beam pair index
    num_classes = beam_output.shape [1] * beam_output.shape [2]
    beams = beam_output.reshape (beam_output.shape [0], num_classes)
    # Beams distribution
    best_beam_index = []
    for sample in range(beams.shape[0]):
        best_beam_index.append(np.argmax(beams[sample, :]))

    return(best_beam_index)
def read_raw_coord():
    path = '../data/coord/'
    filename = 'CoordVehiclesRxPerScene_s009.csv'

    with open (path+filename) as csvfile:
        reader = csv.DictReader(csvfile)
        number_of_rows = len(list(reader))

    all_info_coord = np.zeros((number_of_rows, 4))
    info_coord = []

    with open(path+filename) as csvfile:
        reader = csv.DictReader(csvfile)
        cont = 0
        for row in reader:
            if row['Val'] == 'V':
                all_info_coord[cont] = int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z'])
                info_coord.append([int(row['EpisodeID']), row['LOS'], float(row['x']), float(row['y']), float(row['z'])])
                cont += 1

    return(info_coord)

def read_beam_output_generated_by_ref_17():
    path = '../data/beams_output/beams_output_generate_s009/ref_17/'
    input_cache_file = np.load(path + "index_beams_test.npz", allow_pickle=True)
    key = list(input_cache_file.keys())
    index_beam_output = input_cache_file[key[0]]
    index_beam_str = [str(i) for i in index_beam_output]

    return (index_beam_str)

def read_beam_output_generated_by_raymobtime_baseline():
    path = '../data/beams_output/beam_output_baseline_raymobtime_s009/'
    input_cache_file = np.load(path + "beams_output_test.npz", allow_pickle=True)
    key = list(input_cache_file.keys())
    beam_output = input_cache_file[key[0]]
    index_beam = calculate_index_beams(beam_output)
    index_beam_str = [str(i) for i in index_beam]

    return (index_beam_str)

def comparison_beams_output_generated_by_raymobtime_baseline_and_ray_tracing():
    beam_output_rt = read_beams_output_generated_by_rt()
    index_beam_pair_by_rt = calculate_index_beams (beam_output_rt)
    plot_histogram_beam (index_beam_pair_by_rt, title = 'Distribuicao dos indices dos Beams [S009] \n gerados pelo Ray-tracing')
    plot_hist_prob_beam (index_beam_pair_by_rt, title = 'Probabilidade dos indices dos Beams [S009] \n gerados pelo Ray-tracing')

    beam_output_baseline = read_beam_output_generated_by_raymobtime_baseline()
    index_beam_pair_by_baseline = calculate_index_beams(beam_output_baseline)
    plot_histogram_beam (index_beam_pair_by_baseline,
                         title='Distribuicao dos indices dos Beams [S009] \n gerados pelo Raymobtime Baseline')
    plot_hist_prob_beam (index_beam_pair_by_baseline,
                         title='Probabilidade dos indices dos Beams [S009] \n gerados pelo Raymobtime Baseline')

def compare_index_beams_baseline_and_ref_17():
    index_beam_baseline = read_beam_output_generated_by_raymobtime_baseline()
    index_beam_ref_17 = read_beam_output_generated_by_ref_17()

    ref17_baseline=0
    for i in range (len (index_beam_baseline)):
        if (index_beam_baseline [i] != index_beam_ref_17 [i]):
            ref17_baseline += 1
    print('diferencas: ', ref17_baseline)

    plot_histogram_beam(index_beam_baseline,
                        title='Distribuicao dos indices dos Beams [S009] \n gerados pelo Raymobtime Baseline')
    plot_hist_prob_beam(index_beam_baseline,
                        title='Probabilidade dos indices dos Beams [S009] \n gerados pelo Raymobtime Baseline')

    plot_histogram_beam(index_beam_ref_17,
                        title='Distribuicao dos indices dos Beams [S009] \n gerados pelo Ref_17')
    plot_hist_prob_beam(index_beam_ref_17,
                        title='Probabilidade dos indices dos Beams [S009] \n gerados pelo Ref_17')
def plot_hist_prob_beam(beam, title):#, set, pp_folder, connection, x_label='indice dos beams'):

    #path = pp_folder + 'histogram/'+connection + '/'
    #print(path)
    path = '../analyses/'
    plt.rcParams.update ({'font.size': 8})
    plt.rcParams.update ({'figure.subplot.bottom': 0.146})  # 0.127, 0.9, 0.9]
    fig, ax = plt.subplots (figsize=(8, 4))

    ax.hist (beam,
             bins=256,
             density=True,
             color="steelblue",
             ec="steelblue")
    # ax.plot(data, pdf_lognorm)
    ax.set_ylabel ('P', fontsize=12, rotation=0, color='steelblue', fontweight='bold')
    ax.yaxis.set_label_coords (-0.12, 0.5)
    ax.set_xlabel ('Beam pair index', color='steelblue', fontweight='bold')
    # ax.xaxis.set_label_coords(1.05, -0.025)
    plt.grid (axis='y', alpha=0.9, color='white')
    ax.set_facecolor ('#EEEEF5')
    ax.spines ['right'].set_visible (False)
    ax.spines ['top'].set_visible (False)
    ax.spines ['left'].set_visible (True)
    ax.spines ['bottom'].set_visible (True)
    # plt.tick_params(axis='x', colors='red', direction='out', length=7, width=2)
    #
    title = title
    plt.title (title)

    #plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))


    plt.savefig(path+"histogram_prob_all_Beams_combined.png", bbox_inches='tight')
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
    plt.savefig(name, transparent=False, dpi=300)
    plt.show()




#if __name__ == '__main__':
#    print_hi('PyCharm')
#    beam_output , index_beams = read_beams_output_generated_by_rt()
#    plot_histogram_beam(index_beams, title = 'Distribuicao dos indices dos Beams [S009] \n gerados pelo Ray-tracing')
#    a=0
#_, index_beams = read_beams_output_generated_by_rt()
#plot_hist_prob_beam(title='Distribuicao dos indices dos Beams [S009] \n gerados pelo Ray-tracing', beam=index_beams)
#labels, counts = np.unique(index_beams, return_counts=True)
