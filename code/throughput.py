import numpy as np
import mimo_channels


def read_beams_output_generated_by_ray_tracing():
    print ("\t\tRead Beams output generated from Ray-tracing ")
    path = '../data/beams_output/beams_generate_by_me/'
    beam_output_train = np.load (path + "beams_output_8x32_train.npz", allow_pickle=True) ['output_classification']

    path = '../data/beams_output/beams_generate_by_me/'
    beam_output_test = np.load(path + "beams_output_8x32_test.npz", allow_pickle=True)['output_classification']

    return beam_output_train, beam_output_test

def calculate_best_beam_index_wt_wr(beams, num_antennas_rx):

    best_beam_index = []
    for sample in range (beams.shape [0]):
        best_beam_index.append(np.argmax (beams [sample, :]))

    beam_index_rx = np.array (best_beam_index)

    tx_index = np.zeros ((beams.shape [0]), dtype=int)
    rx_index = np.zeros ((beams.shape [0]), dtype=int)

    for sample in range (len (beam_index_rx)):
        index_tx = best_beam_index [sample] // int (num_antennas_rx)
        index_rx = best_beam_index [sample] % int (num_antennas_rx)
        tx_index [sample] = index_tx
        rx_index [sample] = index_rx

    return tx_index, rx_index, best_beam_index

def read_h_matrix():
    path = '../data/beams_output/beams_generate_by_me/'

    file = np.load (path + "h_matrix_8x32_train.npz", allow_pickle=True)
    keys = list (file.keys())
    h_matrix_train = file[keys [0]]

    file = np.load (path + "h_matrix_8x32_test.npz", allow_pickle=True)
    keys = list (file.keys ())
    h_matrix_test = file[keys [0]]

    return h_matrix_train, h_matrix_test

def power_of_sinal_rx(tx_index, rx_index,):
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


def throughput_ratio():
    # Beam output true
    beam_output_train, beam_output_test = read_beams_output_generated_by_ray_tracing ()
    # calculate the best beam index for each pair of antennas tx and rx
    tx_index, rx_index, best_beam_index = calculate_best_beam_index_wt_wr (beam_output_train, 8)
    true_power = power_of_sinal_rx(tx_index, rx_index,)


    # Beam output predicted


    a=0






    # calculate the power of the signal received
throughput_ratio()





