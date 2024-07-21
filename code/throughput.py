import numpy as np
import mimo_channels
from operator import itemgetter, attrgetter
import generate_beams
import plot_results




def power_of_sinal_rx_1(tx_index, rx_index,):
    # Tx and Rx codebooks
    wr = mimo_channels.dft_codebook(8)
    wt = mimo_channels.dft_codebook(32)

    #Read H matrix
    h_matrix_train, h_matrix_test = read_h_matrix()


    true_all_power = []

    # calculate the best beam of each codebook tx and rx
    for i in range(len(tx_index)):
        best_wr = wr[rx_index[i]].A1
        best_wt = wt[tx_index[i]].T
        best_wt = best_wt.A1
        h = h_matrix_train[i]
        true_power_normalized = np.linalg.norm(np.dot(np.dot(best_wr, h), best_wt))**2
        true_all_power.append(true_power_normalized)

    return true_all_power
def calculated_index_beams_from_beam_output(beam_output):
    best_beam_index = []
    for sample in range (beam_output.shape [0]):
        best_beam_index.append (np.argmax (beam_output [sample, :]))
    return best_beam_index
def throughput_ratio():
    # Beam output true
    beam_output_train, beam_output_test = read_beams_output_generated_by_ray_tracing ()
    index_beam_train = calculated_index_beams_from_beam_output(beam_output_train)
    index_beam_test = calculated_index_beams_from_beam_output(beam_output_test)

    # calculate the best beam index for each pair of antennas tx and rx
    tx_index, rx_index = calculate_best_beam_index_wt_wr(index_beams=index_beam_test, num_antennas_rx=8)
    true_power = power_of_sinal_rx(tx_index, rx_index,)




    # Index Beam  predicted
    index_top_1, index_top_5, index_top_10, index_top_20, index_top_30, index_top_40, index_top_50 = read_index_beams_estimated()

    # Top 1
    tx_index_estimated, rx_index_estimated = calculate_best_beam_index_wt_wr(index_beams=index_top_1, num_antennas_rx=8)
    predicted_power_top_1 = power_of_sinal_rx(tx_index_estimated, rx_index_estimated,)

    #Top 5


    a=0






    # calculate the power of the signal received
def read_h_matrix():
    path = '../data/beams_output/beams_generate_by_me/'

    file = np.load (path + "h_matrix_8x32_train.npz", allow_pickle=True)
    keys = list (file.keys())
    h_matrix_train = file[keys [0]]

    file = np.load (path + "h_matrix_8x32_test.npz", allow_pickle=True)
    keys = list (file.keys ())
    h_matrix_test = file[keys [0]]

    return h_matrix_train, h_matrix_test
def calculate_best_beam_index_wt_wr(index_beams, num_antennas_rx):

    tx_index = np.zeros((len(index_beams)), dtype=int)
    rx_index = np.zeros((len(index_beams)), dtype=int)

    for sample in range(len(index_beams)):
        index_tx = index_beams[sample] // int(num_antennas_rx)
        index_rx = index_beams[sample] % int(num_antennas_rx)
        tx_index[sample] = index_tx
        rx_index[sample] = index_rx

    return tx_index, rx_index






def read_beams_output_generated_by_ray_tracing():
    print ("\t\tRead Beams output generated from Ray-tracing ")
    path = '../data/beams_output/beams_generate_by_me/'
    beam_output_train = np.load (path + "beams_output_8x32_train.npz", allow_pickle=True) ['output_classification']

    path = '../data/beams_output/beams_generate_by_me/'
    beam_output_test = np.load(path + "beams_output_8x32_test.npz", allow_pickle=True)['output_classification']

    return beam_output_train, beam_output_test

def read_estimated_index_into_dict(path):

    top_k = np.arange(1, 51, 1)
    dict = {}

    for i in range(len(top_k)):
        file_name = 'index_beams_predict_top_'+str(top_k[i])+'.npz'
        file = np.load(path + file_name, allow_pickle=True)
        keys = list(file.keys())
        dict[top_k[i]] = file[keys[0]].astype(int)
        a=0

    return dict

def read_index_beams_estimated_novo(path, filename):
    #path = '../results/index_beams_predict/Ruseckas/top_k/lidar/'
    #filename = path + 'index_beams_predict_top_k.npz'
    cache = np.load (path+filename, allow_pickle=True)
    keys = list (cache.keys ())
    all_index = cache [keys [0]]

    top_k = np.arange (1, 51, 1)

    estimate_index_top_k = {}
    for sample in range(len(top_k)):
        estimate_index_top_k[top_k[sample]] = all_index[:,:top_k[sample]].astype(int)

    return estimate_index_top_k




def read_index_beams_estimated(path):


    top_k = [1,5,10,20,30,40,50]
    index_beam_predict = []
    index_top_1 = []
    index_top_5 = []
    index_top_10 = []
    index_top_20 = []
    index_top_30 = []
    index_top_40 = []
    index_top_50 = []

    for i in range(len(top_k)):
        file_name = 'index_beams_predict_top_'+str(top_k[i])+'.npz'
        file = np.load (path + file_name, allow_pickle=True)
        keys = list (file.keys ())

        if top_k[i] == 1:
            top_1 = file[keys[0]]
            #index_top_1 = [int(i) for i in top_1]
            for x in range(len(top_1)):
                val = [int(top_1[x])]
                index_top_1.append(val)
            #index_top1 = [int(j) for j in top_1[i]]
            #index_top_1.append(index_top1)
        elif top_k[i] == 5:
            top_5 = file[keys[0]]
            for i in range(len(top_5)):
                index_top_5.append([int(j) for j in top_5[i]])
        elif top_k[i] == 10:
            top_10 = file[keys[0]]
            for i in range(len(top_10)):
                index_top_10.append([int(j) for j in top_10[i]])
        elif top_k[i] == 20:
            top_20 = file[keys[0]]
            for i in range(len(top_20)):
                index_top_20.append([int(j) for j in top_20[i]])
        elif top_k[i] == 30:
            top_30 = file[keys[0]]
            for i in range(len(top_30)):
                index_top_30.append([int(j) for j in top_30[i]])
        elif top_k[i] == 40:
            top_40 = file[keys[0]]
            for i in range(len(top_40)):
                index_top_40.append([int(j) for j in top_40[i]])
        elif top_k[i] == 50:
            top_50 = file[keys[0]]
            for i in range(len(top_50)):
                index_top_50.append([int(j) for j in top_50[i]])

    return index_top_1, index_top_5, index_top_10, index_top_20, index_top_30, index_top_40, index_top_50
def power_of_sinal_rx():
    beam_output_train, beam_output_test = read_beams_output_generated_by_ray_tracing()

    true_all_power_norm = np.zeros ((beam_output_test.shape[0], 1))
    estimated_all_power_norm = np.zeros ((beam_output_test.shape[0], 1))

    # Beam index true
    #true_beam_index = calculated_index_beams_from_beam_output (beam_output_test)
    true_beam_index = generate_beams.calculate_index_beams (beam_output_test)

    # Calculate the power of the signal received TRUE
    for i in range(len(beam_output_test)):
        a = beam_output_test[i].flatten()
        power_norm = [np.linalg.norm(i)**2 for i in a]
        true_power_norm = power_norm[true_beam_index[i]]
        true_all_power_norm[i] = true_power_norm
        #true_all_power_norm.append(true_power_norm)

    #calculate ALL possible power of the signal received
    all_possible_power_norm = np.zeros ((beam_output_test.shape[0], 256))
    for i in range (len (beam_output_test)):
        a = beam_output_test [i].flatten ()
        power_norm = [np.linalg.norm(i) ** 2 for i in a]
        all_possible_power_norm[i] = power_norm

    return true_all_power_norm, all_possible_power_norm, true_beam_index


    #plot_results.plot_powers_comparition(true_all_power_norm, estimated_all_power_norm)

def calculate_top_k_all_power(index_top_k, all_possible_power_norm):
    index_top_k = np.array(index_top_k)

    all_order_power = np.zeros((index_top_k.shape[0], index_top_k.shape[1]))
    top_k_power = np.zeros((index_top_k.shape[1]))

    for i in range(len(all_possible_power_norm)):
        sample = all_possible_power_norm[i]
        for j in range(index_top_k.shape[1]):
            top_k_power[j] = sample[index_top_k[i,j]]
        order_top_k_power = np.sort(top_k_power, axis=-1)[::-1]
        all_order_power[i] = order_top_k_power

    best_power = all_order_power[:, 0]

    return best_power, all_order_power

def througput_ratio(true_power, estimated_power):

     #np.log2(np.array(true_all_power_norm)+1)
     numerator = np.log2(1+estimated_power)
     denominator = np.log2 (1 + true_power)

     rt = np.mean(numerator/denominator[:,0])
     #rt = np.isclose(numerator, denominator[:, 0]).mean()

     return rt

def calculate_RT_top_k(index_top_1,
                       index_top_5,
                       index_top_10,
                       index_top_20,
                       index_top_30,
                       index_top_40,
                       index_top_50):

    true_all_power_norm, all_possible_power_norm, true_beam_index = power_of_sinal_rx ()

    best_power_top_1, all_power_order_top_1 = calculate_top_k_all_power(index_top_1,
                                                                        all_possible_power_norm)
    best_power_top_5, all_power_order_top_5 = calculate_top_k_all_power(index_top_5,
                                                                        all_possible_power_norm)
    best_power_top_10, all_power_order_top_10 = calculate_top_k_all_power(index_top_10,
                                                                          all_possible_power_norm)
    best_power_20, all_power_order_top_20 = calculate_top_k_all_power(index_top_20,
                                                                      all_possible_power_norm)
    best_power_top_30, all_power_order_top_30 = calculate_top_k_all_power(index_top_30,
                                                                          all_possible_power_norm)
    best_power_top_40, all_power_order_top_40 = calculate_top_k_all_power(index_top_40,
                                                                          all_possible_power_norm)
    best_power_top_50, all_power_order_top_50 = calculate_top_k_all_power(index_top_50,
                                                                          all_possible_power_norm)

    rt_top_1 = througput_ratio (true_all_power_norm, best_power_top_1)
    rt_top_5 = througput_ratio (true_all_power_norm, best_power_top_5)
    rt_top_10 = througput_ratio (true_all_power_norm, best_power_top_10)
    rt_top_20 = througput_ratio (true_all_power_norm, best_power_20)
    rt_top_30 = througput_ratio (true_all_power_norm, best_power_top_30)
    rt_top_40 = througput_ratio (true_all_power_norm, best_power_top_40)
    rt_top_50 = througput_ratio (true_all_power_norm, best_power_top_50)

    rt = [rt_top_1, rt_top_5, rt_top_10, rt_top_20, rt_top_30, rt_top_40, rt_top_50]

    return rt

def throughput_ratio_for_all_techniques(input):

    #input =  'coord'#, 'lidar_coord' 'coord' 'lidar'
    old_version = False
    true_all_power_norm, all_possible_power_norm, true_beam_index = power_of_sinal_rx ()


    technique = 'WiSARD'
    path_index_beams_estimated = '../results/index_beams_predict/' + technique + '/top_k/' + input + '/'
    index_estimated_wisard = read_estimated_index_into_dict(path_index_beams_estimated)
    througput_ratio_wisard = {}

    for i in range(len(index_estimated_wisard)):
        best_power_top_k, all_power_order_top_k = calculate_top_k_all_power(index_estimated_wisard[i+1],
                                                                         all_possible_power_norm)
        througput_ratio_wisard[i+1] = througput_ratio(true_all_power_norm, best_power_top_k)
        a=0


    technique = 'Ruseckas'
    path = '../results/index_beams_predict/' + technique + '/top_k/' + input + '/'
    filename = 'index_beams_predict_top_k.npz'
    index_estimated_ruseckas = read_index_beams_estimated_novo(path, filename)
    througput_ratio_ruseckas = {}


    for i in range (len(index_estimated_ruseckas)):
        best_power_top_k, all_power_order_top_k = calculate_top_k_all_power (index_estimated_ruseckas[i+1],
                                                                             all_possible_power_norm)
        througput_ratio_ruseckas[i+1] = througput_ratio (true_all_power_norm, best_power_top_k)

    technique = 'Batool'
    path_index_beams_estimated = '../results/index_beams_predict/' + technique + '/top_k/' + input + '/'
    filename = 'index_beams_predict_top_k.npz'
    #index_estimated_batool = read_estimated_index_into_dict (path_index_beams_estimated)
    index_estimated_batool = read_index_beams_estimated_novo(filename=filename, path=path_index_beams_estimated)
    througput_ratio_batool = {}

    for i in range(len(index_estimated_batool)):
        best_power_top_k, all_power_order_top_k = calculate_top_k_all_power (index_estimated_batool[i+1],
                                                                             all_possible_power_norm)
        througput_ratio_batool[i+1] = througput_ratio (true_all_power_norm, best_power_top_k)

    top_k = np.arange(1, 51, 1)
    path_save_comparision = '../results/index_beams_predict/'
    name_figure = 'througput_ratio_comparition_'+input

    ratio_thr_wisard = [througput_ratio_wisard[key] for key in througput_ratio_wisard.keys()]
    ratio_thr_wisard = np.array(ratio_thr_wisard)

    ratio_thr_batool = [througput_ratio_batool[key] for key in througput_ratio_batool.keys()]
    ratio_thr_batool = np.array(ratio_thr_batool)

    ratio_thr_ruseckas = [througput_ratio_ruseckas[key] for key in througput_ratio_ruseckas.keys()]
    ratio_thr_ruseckas = np.array(ratio_thr_ruseckas)

    plot_results.plot_powers_comparition (ratio_thr_wisard,
                                          ratio_thr_batool,
                                          ratio_thr_ruseckas,
                                          'WiSARD',
                                          'Batool',
                                          'Ruseckas',
                                          input,
                                          top_k,
                                          path_save_comparision,
                                          name_figure)

    return ratio_thr_wisard, ratio_thr_batool, ratio_thr_ruseckas

    # --------------
    # old version

    if old_version:
        technique = 'WiSARD'
        input_1 = 'coord/coord_top_k_old'
        path_index_beams_estimated = '../results/index_beams_predict/' + technique + '/top_k/' + input_1 + '/'

        index_top_1_W, index_top_5_W, index_top_10_W, index_top_20_W, index_top_30_W, index_top_40_W, index_top_50_W = read_index_beams_estimated (
            path_index_beams_estimated)
        rt_WiSARD = calculate_RT_top_k (index_top_1_W,
                                        index_top_5_W,
                                        index_top_10_W,
                                        index_top_20_W,
                                        index_top_30_W,
                                        index_top_40_W,
                                        index_top_50_W)

        technique = 'Batool'
        path_index_beams_estimated = '../results/index_beams_predict/' + technique + '/top_k/coord/14_index/'# + input + '/'
        index_top_1_B, index_top_5_B, index_top_10_B, index_top_20_B, index_top_30_B, index_top_40_B, index_top_50_B = read_index_beams_estimated (
            path_index_beams_estimated)
        rt_Batool = calculate_RT_top_k (index_top_1_B,
                                        index_top_5_B,
                                        index_top_10_B,
                                        index_top_20_B,
                                        index_top_30_B,
                                        index_top_40_B,
                                        index_top_50_B)

        technique = 'Ruseckas'
        path_index_beams_estimated = '../results/index_beams_predict/' + technique + '/top_k/coord/14_index/'# + input + '/'
        index_top_1_R, index_top_5_R, index_top_10_R, index_top_20_R, index_top_30_R, index_top_40_R, index_top_50_R = read_index_beams_estimated (
            path_index_beams_estimated)
        rt_Ruseckas = calculate_RT_top_k (index_top_1_R,
                                          index_top_5_R,
                                          index_top_10_R,
                                          index_top_20_R,
                                          index_top_30_R,
                                          index_top_40_R,
                                          index_top_50_R)

        top_k = [1, 5, 10, 20, 30, 40, 50]
        path_save_comparision = '../results/index_beams_predict/'
        name_figure = 'rt_comparition_' + input
        plot_results.plot_powers_comparition (rt_WiSARD,
                                              rt_Batool,
                                              rt_Ruseckas,
                                              'WiSARD',
                                              'Batool',
                                              'Ruseckas',
                                              input,
                                              top_k,
                                              path_save_comparision,
                                              name_figure)


def test_calculo_rt():
    true_all_power_norm, all_possible_power_norm = power_of_sinal_rx ()
    top_1_wisard = []
    top_1_batool = []
    top_5_wisard = []
    top_5_batool = []

    a = []
    a1 = []
    for i in range (len (index_top_1)):
        top_1_wisard.append (all_possible_power_norm [i] [index_top_1 [i]])
        # top_1_batool.append (all_possible_power_norm [i] [index_top_1_B [i]])
        top_1_batool.append (all_possible_power_norm [i] [index_top_1_B [i]])

        top_5_wisard.append (np.flip (np.sort (all_possible_power_norm [i] [index_top_5 [i]])))
        top_5_batool.append (np.flip (np.sort (all_possible_power_norm [i] [index_top_5_B [i]])))
        # top_5_wisard.append (all_possible_power_norm[i] [index_top_5[i]])
        # top_5_batool.append(all_possible_power_norm[i][index_top_5_B[i]])

    rt_1_w = througput_ratio (true_all_power_norm, np.array (top_1_wisard) [:, 0])
    rt_1_b = througput_ratio (true_all_power_norm, np.array (top_1_batool) [:, 0])
    rt_5_w = througput_ratio (true_all_power_norm, np.array (top_5_wisard) [:, 0])
    rt_5_b = througput_ratio (true_all_power_norm, np.array (top_5_batool) [:, 0])


#throughput_ratio_for_all_techniques()
#throughput_ratio_for_all_techniques()
#throughput_ratio_for_all_techniques()
#throughput_ratio()
#read_index_beams_estimated()





