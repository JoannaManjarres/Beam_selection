import numpy as np
import mimo_channels
from operator import itemgetter, attrgetter
import generate_beams
import plot_results


def read_beams_output_generated_by_ray_tracing():
    print ("\t\tRead Beams output generated from Ray-tracing ")
    path = '../data/beams_output/beams_generate_by_me/'
    beam_output_train = np.load (path + "beams_output_8x32_train.npz", allow_pickle=True) ['output_classification']

    path = '../data/beams_output/beams_generate_by_me/'
    beam_output_test = np.load(path + "beams_output_8x32_test.npz", allow_pickle=True)['output_classification']

    return beam_output_train, beam_output_test

def calculate_best_beam_index_wt_wr(index_beams, num_antennas_rx):

    tx_index = np.zeros((len(index_beams)), dtype=int)
    rx_index = np.zeros((len(index_beams)), dtype=int)

    for sample in range(len(index_beams)):
        index_tx = index_beams[sample] // int(num_antennas_rx)
        index_rx = index_beams[sample] % int(num_antennas_rx)
        tx_index[sample] = index_tx
        rx_index[sample] = index_rx

    return tx_index, rx_index

def read_h_matrix():
    path = '../data/beams_output/beams_generate_by_me/'

    file = np.load (path + "h_matrix_8x32_train.npz", allow_pickle=True)
    keys = list (file.keys())
    h_matrix_train = file[keys [0]]

    file = np.load (path + "h_matrix_8x32_test.npz", allow_pickle=True)
    keys = list (file.keys ())
    h_matrix_test = file[keys [0]]

    return h_matrix_train, h_matrix_test

def power_of_sinal_rx():
    beam_output_train, beam_output_test = read_beams_output_generated_by_ray_tracing()

    true_all_power_norm = np.zeros ((beam_output_test.shape[0], 1))
    estimated_all_power_norm = np.zeros ((beam_output_test.shape[0], 1))

    # Beam index true
    true_beam_index = calculated_index_beams_from_beam_output (beam_output_test)
    #true_beam_index_1 = generate_beams.calculate_index_beams (beam_output_test)

    # Calculate the power of the signal received true
    for i in range(len(beam_output_test)):
        a = beam_output_test[i].flatten()
        power_norm = [np.linalg.norm(i)**2 for i in a]
        true_power_norm = power_norm[true_beam_index[i]]
        true_all_power_norm[i] = true_power_norm
        #true_all_power_norm.append(true_power_norm)

    #calculate all possible power of the signal received
    all_possible_power_norm = np.zeros ((beam_output_test.shape [0], 256))
    for i in range (len (beam_output_test)):
        a = beam_output_test [i].flatten ()
        power_norm = [np.linalg.norm(i) ** 2 for i in a]
        all_possible_power_norm[i] = power_norm


    # Beam index estimated
    index_top_1, index_top_5, index_top_10, index_top_20, index_top_30, index_top_40, index_top_50 = read_index_beams_estimated ()

    best_power_top_1, all_power_order_top_1 = calculate_top_k_all_power(index_top_1, all_possible_power_norm)
    best_power_top_5,  all_power_order_top_5 = calculate_top_k_all_power(index_top_5, all_possible_power_norm)
    best_power_top_10, all_power_order_top_10 = calculate_top_k_all_power(index_top_10, all_possible_power_norm)
    best_power_20, all_power_order_top_20 = calculate_top_k_all_power(index_top_20, all_possible_power_norm)
    best_power_top_30, all_power_order_top_30 = calculate_top_k_all_power(index_top_30, all_possible_power_norm)
    best_power_top_40, all_power_order_top_40 = calculate_top_k_all_power(index_top_40, all_possible_power_norm)
    best_power_top_50, all_power_order_top_50 = calculate_top_k_all_power(index_top_50, all_possible_power_norm)

    rt_top_1 = througput_ratio (true_all_power_norm, best_power_top_1)
    rt_top_5 = througput_ratio (true_all_power_norm, best_power_top_5)
    rt_top_10 = througput_ratio (true_all_power_norm, best_power_top_10)
    rt_top_20 = througput_ratio (true_all_power_norm, best_power_20)
    rt_top_30 = througput_ratio (true_all_power_norm, best_power_top_30)
    rt_top_40 = througput_ratio (true_all_power_norm, best_power_top_40)
    rt_top_50 = througput_ratio (true_all_power_norm, best_power_top_50)

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
        n=0

    return all_order_power[:, 0], all_order_power

def througput_ratio(true_power, estimated_power):

     #np.log2(np.array(true_all_power_norm)+1)
     numerator = np.log2(1+estimated_power)
     denominator = np.log2 (1 + true_power)

     rt = sum(numerator)/sum(denominator)

     return rt




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

def read_index_beams_estimated():
    path = '../results/index_beams_predict/top_k/'

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
power_of_sinal_rx()
#throughput_ratio()
#read_index_beams_estimated()





