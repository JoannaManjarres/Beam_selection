import analyse_s009 as s009
import analyse_s008 as s008
import numpy as np
import pandas as pd
import pre_process_coord
import wisardpkg as wp
import timeit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
import pre_process_lidar
import seaborn as sns


def beam_selection_wisard(data_train,
                          data_validation,
                          label_train,
                          addressSize):
    # addressSize: number of addressing bits in the ram
    ignoreZero = False  # optional; causes the rams to ignore the address 0

    # False by default for performance reasons,
    # when True, WiSARD prints the progress of train() and classify()
    verbose = False

    wsd = wp.Wisard (addressSize, ignoreZero=ignoreZero, verbose=verbose, bleachingActivated=True)

    tic ()
    wsd.train (data_train, label_train)
    trainning_time = toc ()

    tic ()
    # classify some data
    wisard_result = wsd.classify (data_validation)
    test_time = toc ()


    return wisard_result, trainning_time, test_time

def tic():
    global tic_s
    tic_s = timeit.default_timer()
def toc():
    global tic_s
    toc_s = timeit.default_timer()
    return (toc_s - tic_s)



def read_s009_data(preprocess_resolution):
    coord_s009 = s009.read_raw_coord()
    beams_s009 = s009.read_beam_output_generated_by_raymobtime_baseline()

    info_s009 = pd.DataFrame(coord_s009,columns=['Episode','LOS', 'x', 'y', 'z'])
    #info_s009.insert(5, 'beams_index', beams_s009)


    encoding_coord_test = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s009(escala=preprocess_resolution)

    return info_s009, encoding_coord_test, beams_s009
def read_s008_data(preprocess_resolution):

    _, _, encoding_coord_train = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s008 (escala=preprocess_resolution)
    index_beams_train, index_beam_validation, _, _ = s008.read_beams_output_from_baseline ()
    beams_train = np.concatenate ((index_beams_train, index_beam_validation), axis=0)

    all_info_s008, coord_train, coord_validation = pre_process_coord.read_valid_coordinates_s008()
    info_s008 = pd.DataFrame(all_info_s008, columns=['Episode', 'x', 'y', 'z', 'LOS'])

    return info_s008, encoding_coord_train, beams_train

def fit_incremental(nro_of_episodes, input_type):
    preprocess_resolution = 16
    th = 0.15
    all_info_s009, encoding_coord_s009, beams_s009 = read_s009_data (preprocess_resolution)
    all_info_s008, encoding_coord_s008, beams_s008 = read_s008_data (preprocess_resolution)
    data_lidar_2D_with_rx_s008, data_lidar_2D_with_rx_s009 = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer()
    data_lidar_s008, data_lidar_s009 = pre_process_lidar.data_lidar_2D_binary_without_variance(data_lidar_2D_with_rx_s008,
                                                                                               data_lidar_2D_with_rx_s009,
                                                                                               th)

    s008_data = all_info_s008[['Episode']].copy()
    s008_data ['index_beams'] = beams_s008.tolist()
    s008_data ['encoding_coord'] = encoding_coord_s008.tolist()
    s008_data ['lidar'] = data_lidar_s008.tolist()

    # info_of_episode = s008_data[s008_data['Episode'] == 0]

    s009_data = all_info_s009[['Episode']].copy ()
    s009_data ['index_beams'] = beams_s009
    s009_data ['encoding_coord'] = encoding_coord_s009.tolist ()
    s009_data ['lidar'] = data_lidar_s009.tolist ()

    # episode_for_test = np.arange(0, 2000, 1)
    episode_for_test = np.arange (0, nro_of_episodes, 1)

    label_input_type = input_type
    s008_data_copy = s008_data.copy ()

    #labels_for_next_train = []
    #samples_for_next_train = []
    all_score = []
    all_trainning_time = []
    all_test_time = []
    all_episodes = []
    all_samples_train = []
    all_samples_test = []

    for i in range (len (episode_for_test)):
        if i in s009_data ['Episode'].tolist ():
            if i == 0:
                label_train = s008_data_copy['index_beams'].tolist()
                if label_input_type == 'coord':
                    input_train = s008_data_copy['encoding_coord'].tolist()
                elif label_input_type == 'lidar':
                    input_train = s008_data_copy['lidar'].tolist()

                label_test = s009_data[s009_data ['Episode'] == i]['index_beams'].tolist ()
                if label_input_type == 'coord':
                    input_test = s009_data[s009_data ['Episode'] == i]['encoding_coord'].tolist ()
                elif label_input_type == 'lidar':
                    input_test = s009_data[s009_data ['Episode'] == i]['lidar'].tolist ()

                labels_for_next_train = label_test
                samples_for_next_train = input_test
            else:
                for j in range (len (labels_for_next_train)):
                    label_train.append(labels_for_next_train[j])
                    input_train.append(samples_for_next_train[j])

                label_test = s009_data [s009_data['Episode'] == i]['index_beams'].tolist()
                if label_input_type == 'coord':
                    input_test = s009_data [s009_data['Episode'] == i]['encoding_coord'].tolist()
                elif label_input_type == 'lidar':
                    input_test = s009_data [s009_data['Episode'] == i]['lidar'].tolist()

                labels_for_next_train=label_test
                samples_for_next_train=input_test

            index_predict, trainning_time, test_time = beam_selection_wisard (data_train=input_train,
                                                                              data_validation=input_test,
                                                                              label_train=label_train,
                                                                              addressSize=44)
            score = accuracy_score(label_test, index_predict)
            all_score.append(score)
            all_trainning_time.append(trainning_time)
            all_test_time.append(test_time)
            all_episodes.append(i)
            all_samples_train.append(len(input_train))
            all_samples_test.append(len(input_test))

        else:
            continue

    average_score = []
    for i in range(len(all_score)):
        i = i + 1
        average_score.append(np.mean(all_score[0:i]))

    path_result = '../results/score/Wisard/online/' + label_input_type + '/incremental_window/'
    plt.plot (all_episodes, all_score, 'o--', color='red', label='Accuracy per episode')
    plt.plot(all_episodes, average_score, 'o-', color='blue', label='Cumulative average accuracy')

    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right', bbox_to_anchor=(1.04, 0))
    plt.title('Beam selection using WiSARD with ' + label_input_type + ' in incremental window')
    plt.savefig(path_result + 'score_incremental_window.png')
    plt.close()

    plt.plot(all_episodes, all_trainning_time, 'o-', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Trainning Time')
    plt.title('Trainning Time using fit with incremental window')
    plt.savefig(path_result + 'time_train_incremental_window.png')
    plt.close()

    plt.plot(all_episodes, all_test_time, 'o-', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Test Time')
    plt.title('Test Time using fit with incremental window')
    plt.savefig(path_result + 'time_test_incremental_window.png')
    plt.close()

    headerList = ['Episode', 'Score', 'Trainning Time', 'Test Time', 'Samples Train', 'Samples Test']

    with open (path_result + 'all_results_incremental_window.csv', 'w') as f:
        writer_results = csv.writer(f, delimiter=',')
        writer_results.writerow(headerList)
        writer_results.writerows(zip(all_episodes,
                                     all_score,
                                     all_trainning_time, all_test_time,
                                     all_samples_train, all_samples_test))
def fit_sliding_window(nro_of_episodes, input_type):
    preprocess_resolution = 16
    th = 0.15
    all_info_s009, encoding_coord_s009, beams_s009 = read_s009_data(preprocess_resolution)
    all_info_s008, encoding_coord_s008, beams_s008 = read_s008_data(preprocess_resolution)
    data_lidar_2D_with_rx_s008, data_lidar_2D_with_rx_s009 = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer()
    data_lidar_s008, data_lidar_s009 = pre_process_lidar.data_lidar_2D_binary_without_variance(data_lidar_2D_with_rx_s008,
                                                                                               data_lidar_2D_with_rx_s009,
                                                                                               th)

    s008_data = all_info_s008[['Episode']].copy()
    s008_data['index_beams'] = beams_s008.tolist()
    s008_data['encoding_coord'] = encoding_coord_s008.tolist()
    s008_data['lidar'] = data_lidar_s008.tolist()

    #info_of_episode = s008_data[s008_data['Episode'] == 0]

    s009_data = all_info_s009[['Episode']].copy()
    s009_data['index_beams'] = beams_s009
    s009_data['encoding_coord'] = encoding_coord_s009.tolist()
    s009_data['lidar'] = data_lidar_s009.tolist()


    #episode_for_test = np.arange(0, 2000, 1)
    episode_for_test = np.arange (0, nro_of_episodes, 1)

    label_input_type = input_type
    s008_data_copy = s008_data.copy()

    labels_for_next_train = []
    samples_for_next_train = []
    all_score = []
    all_trainning_time = []
    all_test_time = []
    all_episodes = []
    all_samples_train = []
    all_samples_test = []

    for i in range(len(episode_for_test)):
        if i in s009_data['Episode'].tolist():

            if i == 0:
                label_train = s008_data_copy['index_beams'].tolist()
                if label_input_type == 'coord':
                    input_train = s008_data_copy['encoding_coord'].tolist()
                elif label_input_type == 'lidar':
                    input_train = s008_data_copy['lidar'].tolist()

                label_test = s009_data[s009_data['Episode'] == i]['index_beams'].tolist()
                if label_input_type == 'coord':
                    input_test = s009_data[s009_data['Episode'] == i]['encoding_coord'].tolist()
                elif label_input_type == 'lidar':
                    input_test = s009_data[s009_data['Episode'] == i]['lidar'].tolist()
                for k in range(len(label_test)):
                    labels_for_next_train.append(label_test[k])
                    samples_for_next_train.append(input_test[k])

                #print(i, len(label_train), len(label_test))
            else:
                s008_data_copy = s008_data_copy.drop(s008_data_copy[s008_data_copy['Episode'] == i-1].index)
                label_train = s008_data_copy['index_beams'].tolist()
                if label_input_type == 'coord':
                    input_train = s008_data_copy['encoding_coord'].tolist()
                elif label_input_type == 'lidar':
                    input_train = s008_data_copy['lidar'].tolist()
                for j in range(len(labels_for_next_train)):
                    label_train.append(labels_for_next_train[j])
                    input_train.append(samples_for_next_train[j])

                label_test = s009_data[s009_data ['Episode'] == i]['index_beams'].tolist()
                if label_input_type == 'coord':
                    input_test = s009_data[s009_data ['Episode'] == i]['encoding_coord'].tolist()
                elif label_input_type == 'lidar':
                    input_test = s009_data[s009_data ['Episode'] == i]['lidar'].tolist()
                for k in range(len(label_test)):
                    labels_for_next_train.append(label_test[k])
                    samples_for_next_train.append(input_test[k])

                #print(i, len(label_train), len(label_test))

            index_predict, trainning_time, test_time = beam_selection_wisard(data_train=input_train,
                                                                             data_validation=input_test,
                                                                             label_train=label_train,
                                                                             addressSize=44)

            score = accuracy_score (label_test, index_predict)
            all_score.append(score)
            all_trainning_time.append(trainning_time)
            all_test_time.append(test_time)
            all_episodes.append(i)
            all_samples_train.append(len(input_train))
            all_samples_test.append(len(input_test))
        else:
            continue

    average_score = []
    for i in range(len(all_score)):
        i = i+1
        average_score.append(np.mean(all_score[0:i]))

    path_result = '../results/score/Wisard/online/' + label_input_type + '/sliding_window/'
    # plt.plot (episode_for_test, average_score, 'o-', label='Cumulative average accuracy')
    plt.plot (all_episodes, all_score, 'o--', color='red', label='Accuracy per episode')
    plt.plot(all_episodes, average_score, 'o-', color='blue', label='Cumulative average accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right', bbox_to_anchor=(1.04, 0))
    plt.title('Beam selection using WiSARD with ' + label_input_type + ' in sliding window')
    plt.savefig(path_result + 'score_sliding_window.png')
    #plt.show ()
    plt.close ()

    plt.plot (all_episodes, all_trainning_time, 'o-', color='green')
    plt.xlabel ('Episode')
    plt.ylabel ('Trainning Time')
    plt.title ('Trainning Time using fit with sliding window')
    plt.savefig (path_result +'time_train_sliding_window.png')
    plt.close ()

    plt.plot (all_episodes, all_test_time, 'o-', color='blue')
    plt.xlabel ('Episode')
    plt.ylabel ('Test Time')
    plt.title ('Test Time using fit with sliding window')
    plt.savefig (path_result +'time_test_sliding_window.png')
    plt.close ()

    headerList = ['Episode', 'Score', 'Trainning Time', 'Test Time', 'Samples Train', 'Samples Test']

    with open(path_result + 'all_results_sliding_window.csv', 'w') as f:
        writer_results = csv.writer (f, delimiter=',')
        writer_results.writerow (headerList)
        writer_results.writerows(zip(all_episodes,
                                     all_score,
                                     all_trainning_time, all_test_time,
                                     all_samples_train, all_samples_test))

    #with open (path_result + 'all_results_sliding_window.csv', 'w') as f:
    #    writer_time_test = csv.writer (f, delimiter=',')
    #    #writer_time_test.writerows(['Episode','Score','Trainning Time','Test Time','Samples Train','Samples Test'])
    #    writer_time_test.writerows (zip (all_episodes, all_score, all_trainning_time, all_test_time, all_samples_train, all_samples_test))
def fit_traditional(nro_of_episodes, label_input_type):

    preprocess_resolution = 16
    th = 0.15
    # data of coordinates, episodes and beams from s009 and s008
    all_info_s009, encoding_coord_s009, beams_s009 = read_s009_data(preprocess_resolution)
    all_info_s008, encoding_coord_s008, beams_s008 = read_s008_data(preprocess_resolution)

    #data of lidar 2D with rx from s009 and s008
    data_lidar_2D_with_rx_s008, data_lidar_2D_with_rx_s009 = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer()
    data_lidar_s008, data_lidar_s009 = pre_process_lidar.data_lidar_2D_binary_without_variance(data_lidar_2D_with_rx_s008,
                                                                                               data_lidar_2D_with_rx_s009,
                                                                                               th)

    s008_data = all_info_s008[['Episode']].copy()
    s008_data['index_beams'] = beams_s008.tolist()
    s008_data['encoding_coord'] = encoding_coord_s008.tolist()
    s008_data['lidar'] = data_lidar_s008.tolist()

    s009_data = all_info_s009[['Episode']].copy()
    s009_data['index_beams'] = beams_s009
    s009_data['encoding_coord'] = encoding_coord_s009.tolist()
    s009_data['lidar'] = data_lidar_s009.tolist()


    #episode_for_test = np.arange(0, 2000, 1)
    episode_for_test = np.arange (0, nro_of_episodes, 1)

    #label_input_type = 'coord'

    all_score = []
    all_trainning_time = []
    all_test_time = []
    all_episodes = []
    all_samples_train = []
    all_samples_test = []

    label_train =  s008_data['index_beams'].tolist()
    if label_input_type == 'coord':
        input_train = s008_data['encoding_coord'].tolist()
    elif label_input_type == 'lidar':
        input_train = s008_data['lidar'].tolist()

    for i in range(len(episode_for_test)):
        if i in s009_data['Episode'].tolist():
            label_test = s009_data[s009_data['Episode'] == i]['index_beams'].tolist()
            if label_input_type == 'coord':
                input_test = s009_data[s009_data['Episode'] == i]['encoding_coord'].tolist()
            elif label_input_type == 'lidar':
                input_test = s009_data[s009_data['Episode'] == i]['lidar'].tolist()

            index_predict, trainning_time, test_time = beam_selection_wisard (data_train=input_train,
                                                                                  data_validation=input_test,
                                                                                  label_train=label_train,
                                                                                  addressSize=44)
            score = accuracy_score (label_test, index_predict)
            all_score.append(score)
            all_trainning_time.append(trainning_time)
            all_test_time.append(test_time)
            all_episodes.append(i)
            all_samples_train.append(len(input_train))
            all_samples_test.append(len(input_test))
        else:
            continue

    average_score = []
    for i in range(len(all_score)):
        i = i+1
        average_score.append(np.mean(all_score[0:i]))

    ## SAVE RESULTS
    path_result = '../results/score/Wisard/online/' + label_input_type + '/traditional_fit/'
    plt.plot (all_episodes, all_score, 'o--', color='red', label='Accuracy per episode')
    plt.plot(all_episodes, average_score, 'o-', color='blue', label='Cumulative average accuracy')

    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right', bbox_to_anchor=(1.04, 0))
    plt.title('Beam selection using WiSARD with ' + label_input_type + ' in traditional fit')
    plt.savefig(path_result + 'score_traditional_train.png')
    plt.close()

    plt.plot(all_episodes, all_trainning_time, 'o-', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Trainning Time')
    plt.title ('Trainning Time using Traditional fit ')
    plt.savefig(path_result +'time_train_traditional_fit.png')
    plt.close()

    plt.plot(all_episodes, all_test_time, 'o-', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Test Time')
    plt.title ('Test Time using Traditional fit ')
    plt.savefig (path_result +'time_test_traditional_fit.png')
    plt.close()

    headerList = ['Episode', 'Score', 'Trainning Time', 'Test Time', 'Samples Train', 'Samples Test']

    with open (path_result + 'all_results_traditional_fit.csv', 'w') as f:
        writer_results = csv.writer(f, delimiter=',')
        writer_results.writerow(headerList)
        writer_results.writerows (zip (all_episodes, all_score, all_trainning_time, all_test_time, all_samples_train, all_samples_test))

def fit_fixed_window(nro_of_episodes_test, nro_of_episodes_train, input_type, rodada):
    preprocess_resolution = 16
    th = 0.15
    all_info_s009, encoding_coord_s009, beams_s009 = read_s009_data (preprocess_resolution)
    all_info_s008, encoding_coord_s008, beams_s008 = read_s008_data (preprocess_resolution)
    data_lidar_2D_with_rx_s008, data_lidar_2D_with_rx_s009 = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer ()
    data_lidar_s008, data_lidar_s009 = pre_process_lidar.data_lidar_2D_binary_without_variance (
        data_lidar_2D_with_rx_s008,
        data_lidar_2D_with_rx_s009,
        th)

    s008_data = all_info_s008 [['Episode']].copy ()
    s008_data ['index_beams'] = beams_s008.tolist ()
    s008_data ['encoding_coord'] = encoding_coord_s008.tolist ()
    s008_data ['lidar'] = data_lidar_s008.tolist ()

    # info_of_episode = s008_data[s008_data['Episode'] == 0]

    s009_data = all_info_s009 [['Episode']].copy ()
    s009_data ['index_beams'] = beams_s009
    s009_data ['encoding_coord'] = encoding_coord_s009.tolist ()
    s009_data ['lidar'] = data_lidar_s009.tolist ()

    # episode_for_test = np.arange(0, 2000, 1)
    episode_for_test = np.arange (0, nro_of_episodes_test, 1)
    episode_for_train = np.arange (0, nro_of_episodes_train, 1)

    label_input_type = input_type
    s008_data_copy = s008_data.copy ()

    labels_for_next_train = []
    samples_for_next_train = []
    all_score = []
    all_trainning_time = []
    all_test_time = []
    all_episodes = []
    all_samples_train = []
    all_samples_test = []

    label_train =[]
    input_train = []
    nonexistent_episodes = []
    index_beams =[]
    input = []

    for i in range(len (episode_for_train)):
        if i in s008_data['Episode'].tolist():
            index_beams.append(s008_data[s008_data['Episode'] == i]['index_beams'].tolist())
            #for j in range(len(index_beams)):
            #    label_train.append(index_beams[j])
            if label_input_type == 'coord':
                input.append(s008_data[s008_data['Episode'] == i]['encoding_coord'].tolist())
            #    for m in range(len(input)):
            #        input_train.append(input[m])
            elif label_input_type == 'lidar':
                input.append(s008_data[s008_data['Episode'] == i]['lidar'].tolist())
            #    for m in range(len(input)):
            #        input_train.append(input[m])
        else:
            nonexistent_episodes.append(i)
            continue

    for i in range(len(index_beams)):
        for j in range(len(index_beams[i])):
            label_train.append(index_beams[i][j])
            input_train.append(input[i][j])


    for i in range(len(episode_for_test)):
        if i in s009_data ['Episode'].tolist():
            label_test = s009_data [s009_data['Episode'] == i]['index_beams'].tolist ()
            if label_input_type == 'coord':
                input_test = s009_data [s009_data ['Episode'] == i] ['encoding_coord'].tolist ()
            elif label_input_type == 'lidar':
                input_test = s009_data [s009_data ['Episode'] == i] ['lidar'].tolist ()

                # print(i, len(label_train), len(label_test))

            index_predict, trainning_time, test_time = beam_selection_wisard (data_train=input_train,
                                                                              data_validation=input_test,
                                                                              label_train=label_train,
                                                                              addressSize=44)

            score = accuracy_score (label_test, index_predict)
            all_score.append (score)
            all_trainning_time.append (trainning_time)
            all_test_time.append (test_time)
            all_episodes.append (i)
            all_samples_train.append (len (input_train))
            all_samples_test.append (len (input_test))
        else:
            continue

    average_score = []
    for i in range (len (all_score)):
        i = i + 1
        average_score.append (np.mean (all_score [0:i]))

    path_result = '../results/score/Wisard/online/' + label_input_type + '/fixed_window/'
    # plt.plot (episode_for_test, average_score, 'o-', label='Cumulative average accuracy')
    plt.plot (all_episodes, all_score, 'o--', color='red', label='Accuracy per episode')
    plt.plot (all_episodes, average_score, 'o-', color='blue', label='Cumulative average accuracy')

    plt.xlabel ('Episode')
    plt.ylabel ('Accuracy')
    plt.legend (loc='lower right', bbox_to_anchor=(1.04, 0))
    plt.title ('Beam selection using WiSARD with ' + label_input_type + ' in fixed window')
    plt.savefig (path_result + str(rodada) +'score_fixed_window.png')
    # plt.show ()
    plt.close ()

    plt.plot (all_episodes, all_trainning_time, 'o-', color='green')
    plt.xlabel ('Episode')
    plt.ylabel ('Trainning Time')
    plt.title ('Trainning Time using fit with fixed window')
    plt.savefig (path_result +str(rodada) + 'time_train_fixed_window.png')
    plt.close ()

    plt.plot(all_episodes, all_test_time, 'o-', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Test Time')
    plt.title('Test Time using fit with fixed window')
    plt.savefig(path_result +str(rodada) + 'time_test_fixed_window.png')
    plt.close()

    headerList = ['Episode', 'Score', 'Trainning Time', 'Test Time', 'Samples Train', 'Samples Test']

    with open (path_result + str(rodada) +'all_results_fixed_window.csv', 'w') as f:
        writer_results = csv.writer (f, delimiter=',')
        writer_results.writerow (headerList)
        writer_results.writerows (zip (all_episodes,
                                       all_score,
                                       all_trainning_time, all_test_time,
                                       all_samples_train, all_samples_test))
def plot_score_comparation(input_type):
    path_result = '../results/score/Wisard/online/'
    #path_result_traditional = path_result + input_type+'/traditional_fit/'
    path_result_sliding_window = path_result + input_type+'/sliding_window/'
    path_result_incremental_window = path_result + input_type+'/incremental_window/'
    path_result_fixed_window = path_result + input_type+'/fixed_window/'

    #all_results_traditional = pd.read_csv(path_result_traditional + 'all_results_traditional_fit.csv')
    all_results_sliding_window = pd.read_csv(path_result_sliding_window + 'all_results_sliding_window.csv')
    all_results_incremental_window = pd.read_csv(path_result_incremental_window + 'all_results_incremental_window.csv')
    all_results_fixed_window = pd.read_csv(path_result_fixed_window + 'all_results_fixed_window.csv')

    #all_score_traditional = all_results_traditional['Score'].tolist()
    #average_score_traditional = []
    #for i in range (len (all_score_traditional)):
    #    i = i + 1
    #    average_score_traditional.append (np.mean (all_score_traditional[0:i]))

    all_score_sliding_window = all_results_sliding_window['Score'].tolist()
    average_score_sliding_window = []
    for i in range (len (all_score_sliding_window)):
        i = i + 1
        average_score_sliding_window.append (np.mean (all_score_sliding_window[0:i]))

    all_score_incremental_window = all_results_incremental_window['Score'].tolist()
    average_score_incremental_window = []
    for i in range (len (all_score_incremental_window)):
        i = i + 1
        average_score_incremental_window.append (np.mean (all_score_incremental_window [0:i]))

    all_score_fixed_window = all_results_fixed_window['Score'].tolist()
    average_score_fixed_window = []
    for i in range (len (all_score_fixed_window)):
        i = i + 1
        average_score_fixed_window.append(np.mean (all_score_fixed_window [0:i]))


    mean_sliding_window = np.round(np.mean(average_score_sliding_window),3)
    mean_incremental_window = np.round(np.mean(average_score_incremental_window),3)
    mean_fixed_window = np.round(np.mean(average_score_fixed_window),3)

    text_sliding_window = 'Mean: '+str(mean_sliding_window)
    text_incremental_window = 'Mean: '+str(mean_incremental_window)
    text_fixed_window = 'Mean: '+str(mean_fixed_window)


    #plt.plot(all_results_traditional['Episode'], average_score_traditional, 'o-', color='blue', label='Fixed window')
    plt.plot(all_results_sliding_window['Episode'], average_score_sliding_window, 'o-', color='red', label='Sliding window')
    plt.text(1800, mean_sliding_window+0.03, text_sliding_window, fontsize=8, color='red', fontname='Myanmar Sangam MN', fontweight='bold')
    plt.plot(all_results_incremental_window['Episode'], average_score_incremental_window, 'o-', color='green', label='Incremental window')
    plt.text(1800, mean_incremental_window, text_incremental_window, fontsize=8, color='green', fontname='Myanmar Sangam MN', fontweight='bold')
    plt.plot(all_results_fixed_window['Episode'], average_score_fixed_window, 'o-', color='purple', label='Fixed window')
    plt.text (1800, mean_fixed_window-0.02, text_fixed_window, fontsize=8, color='purple', fontname='Myanmar Sangam MN', fontweight='bold')

    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')#, bbox_to_anchor=(1.04, 0))
    plt.title('Beam selection using WiSARD with \n'+input_type+' in online learning')
    plt.savefig(path_result + input_type + '/score_comparation.png', dpi=300)
    plt.close()

def plot_time_comparition(input_type):
    path_result = '../results/score/Wisard/online/'
    path_result_traditional = path_result + input_type+'/traditional_fit/'
    path_result_sliding_window = path_result + input_type+'/sliding_window/'
    path_result_incremental_window = path_result + input_type+'/incremental_window/'
    path_result_fixed_window = path_result + input_type+'/fixed_window/'

    all_results_traditional = pd.read_csv(path_result_traditional + 'all_results_traditional_fit.csv')
    all_results_sliding_window = pd.read_csv(path_result_sliding_window + 'all_results_sliding_window.csv')
    all_results_incremental_window = pd.read_csv(path_result_incremental_window + 'all_results_incremental_window.csv')
    all_results_fixed_window = pd.read_csv(path_result_fixed_window + 'all_results_fixed_window.csv')

    flag = 'training'
    sns.set_theme (style="darkgrid")
    fig, ax1 = plt.subplots (figsize=(15, 7))
    plt.plot(all_results_traditional['Episode'],
             all_results_traditional['Trainning Time'],
             color='blue')
    plt.plot(all_results_sliding_window['Episode'],
                all_results_sliding_window['Trainning Time'],
                color='red')
    plt.plot(all_results_incremental_window['Episode'],
                all_results_incremental_window['Trainning Time'],
                color='green')
    plt.plot(all_results_fixed_window['Episode'],
                all_results_fixed_window['Trainning Time'],
                color='purple')


    ax1.set_ylabel(flag + ' time' , fontsize=12, color='black', labelpad=10, fontweight='bold', fontname = 'Myanmar Sangam MN')
    ax1.set_xlabel('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold', fontname = 'Myanmar Sangam MN')


    # Criando um segundo eixo
    ax2 = ax1.twinx ()
    #plt.plot (all_results_traditional['Episode'],
    #          all_results_traditional['Samples Train'],
    #          marker=',', color='blue', alpha=0.3, label='fixed window')
    plt.plot (all_results_sliding_window['Episode'],
                all_results_sliding_window['Samples Train'],
                marker=',', color='red',  alpha=0.3,label='sliding window')
    plt.plot (all_results_incremental_window['Episode'],
                all_results_incremental_window['Samples Train'],
                marker=',', color='green',  alpha=0.3,label='incremental window')
    plt.plot (all_results_fixed_window['Episode'],
                all_results_fixed_window['Samples Train'],
                marker=',', color='purple', alpha=0.3, label='fixed window')


    ax2.set_ylabel ('training samples', fontsize=12, color='black', labelpad=12, fontweight='bold', fontname = 'Myanmar Sangam MN')#, color='red')

    # Adicionando título e legendas
    title = "Relationship between trained samples and training time \n usign data: " + input_type
    plt.title(title, fontsize=15, color='black',  fontweight='bold')
    #plt.xticks(all_results_traditional['Episode'])
    plt.xlabel('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')
    plt.legend(loc='best',  ncol=3)#loc=(0,-0.4), ncol=3)#loc='best')
    plt.savefig (path_result + input_type + '/time_and_samples_train_comparation.png', dpi=300)
    plt.close()


    #plt.plot(all_results_traditional['Episode'], all_results_traditional['Trainning Time'], 'o-', color='blue', label='Fixed window')
    plt.plot(all_results_sliding_window['Episode'], all_results_sliding_window['Trainning Time'], '.', color='red', label='Sliding window')
    plt.plot(all_results_incremental_window['Episode'], all_results_incremental_window['Trainning Time'], '.', color='green', label='Incremental window')
    plt.plot(all_results_fixed_window['Episode'], all_results_fixed_window['Trainning Time'], '.', color='purple', label='Fixed window')
    plt.xlabel('Episode', fontsize=12, color='black',  fontname = 'Myanmar Sangam MN', fontweight='bold')
    plt.ylabel('Trainning Time', fontsize=12, color='black', fontname = 'Myanmar Sangam MN', fontweight='bold')
    plt.legend(loc='best')#, bbox_to_anchor=(1.04, 0))
    plt.title('Trainning Time using WiSARD with \n'+input_type+' in online learning')
    plt.savefig(path_result + input_type + '/time_train_comparation.png', dpi=300)
    plt.close()

    #plt.plot(all_results_traditional['Episode'], all_results_traditional['Test Time'], 'o-', color='blue', label='Traditional fit')
    plt.plot(all_results_sliding_window['Episode'], all_results_sliding_window['Test Time'], '.', color='red', label='Sliding window')
    plt.plot(all_results_incremental_window['Episode'], all_results_incremental_window['Test Time'], '.', color='green', label='Incremental window')
    plt.plot(all_results_fixed_window['Episode'], all_results_fixed_window['Test Time'], '.', color='purple', label='Fixed window')
    plt.xlabel('Episode', fontsize=12, color='black', fontname = 'Myanmar Sangam MN', fontweight='bold')
    plt.ylabel('Test Time', fontsize=12, color='black', fontname = 'Myanmar Sangam MN', fontweight='bold')
    plt.legend(loc='best')#, bbox_to_anchor=(1.04, 0))
    plt.title('Test Time using WiSARD with \n'+input_type+' in online learning', fontsize=12, color='black')
    plt.savefig(path_result + input_type + '/time_test_comparation.png', dpi=300)
    plt.close()

def comparition_between_results_of_time_simulations():
    path_result = '../results/score/Wisard/online/'
    path_result_fixed_window = path_result + input_type+'/fixed_window/'

    results_fixed_window_1 = pd.read_csv (path_result_fixed_window + '1time_train_fixed_window.csv')


eposodies_for_test = 2000
episodes_for_train = 2086
input_type = 'coord'
#fit_traditional(eposodies_for_test, input_type)
rodada = 1
fit_fixed_window(eposodies_for_test, episodes_for_train, input_type, rodada)
rodada = 2
fit_fixed_window(eposodies_for_test, episodes_for_train, input_type, rodada)
#fit_sliding_window(eposodies_for_test, input_type)
#fit_incremental(eposodies_for_test, input_type)

#plot_score_comparation(input_type)
#plot_time_comparition(input_type)


# testar tempo de treinamento e teste com esta ferramenta: https://docs.python.org/3/library/time.html#time.perf_counter



