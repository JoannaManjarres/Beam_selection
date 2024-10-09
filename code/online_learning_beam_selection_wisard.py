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
import time
from operator import itemgetter
import argparse


def beam_selection_top_k_wisard(x_train, x_test,
                                y_train, y_test,
                                address_of_size,
                                name_of_conf_input):

    #print ("... Calculando os top-k com Wisard")
    addressSize = address_of_size
    ignoreZero = False
    verbose = False #True
    var = True
    wsd = wp.Wisard(addressSize,
                    ignoreZero=ignoreZero,
                    verbose=verbose,
                    returnConfidence=var,
                    returnActivationDegree=var,
                    returnClassesDegrees=var)

    star_trainning = time.process_time_ns ()
    wsd.train(x_train, y_train)
    end_trainning = time.process_time_ns ()
    trainning_process_time = (end_trainning - star_trainning)



    # the output is a list of string, this represent the classes attributed to each input
    star_test = time.process_time_ns ()
    out = wsd.classify(x_test)
    end_test = time.process_time_ns ()
    test_process_time = (end_test - star_test)

    #wsd_1 = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)
    #wsd_1.train(x_train, y_train)
    #out_1 = wsd_1.classify(x_test)


    content_index = 0
    ram_index = 0
    #print(wsd.getsizeof(ram_index,content_index))
    #print(wsd.json())
    #print(out)

    top_k = [1, 5, 10, 15, 20, 25, 30]
    #top_k = np.arange(1, 51, 1)


    acuracia = []
    score = []

    all_classes_order = []

    for sample in range(len(out)):

        classes_degree = out[sample]['classesDegrees']
        dict_classes_degree_order = sorted(classes_degree, key=itemgetter('degree'), reverse=True)

        classes_by_sample_in_order = []
        for x in range(len(dict_classes_degree_order)):
            classes_by_sample_in_order.append(dict_classes_degree_order[x]['class'])
        all_classes_order.append(classes_by_sample_in_order)

    path_index_predict = '../results/score/Wisard/online/top_k/' + name_of_conf_input + '/'
    for i in range (len(top_k)):
        acerto = 0
        nao_acerto = 0
        best_classes = []
        best_classes_int = []
        for sample in range (len(all_classes_order)):
            best_classes.append(all_classes_order[sample][:top_k[i]])
            best_classes_int.append([int(x) for x in best_classes[sample]])
            if (y_test[sample] in best_classes[sample]):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        score.append(acerto / len(out))

        #file_name = 'index_beams_predict_top_' + str(top_k[i]) + '.npz'
        #npz_index_predict = path_index_predict + file_name
        #np.savez(npz_index_predict, output_classification=best_classes_int)

    df_results_wisard_top_k = pd.DataFrame ({"Top-K": top_k,
                                           "Acuracia": score,
                                           "Trainning Time": trainning_process_time,
                                           "Test Time": test_process_time})
    #path_csv = '../results/accuracy/8X32/' + data_input + '/top_k/'
    path_csv = '../results/score/Wisard/online/top_k/'+name_of_conf_input+'/'
    #df_score_wisard_top_k.to_csv (path_csv + 'score_' + name_of_conf_input + '_top_k.csv', index=False)

    file_name = 'index_beams_predict_top_k.npz'
    npz_index_predict = path_index_predict + file_name
    #np.savez (npz_index_predict, output_classification=all_classes_order)


    plot =False
    if plot:
        print ('Enderecamento de memoria: ', addressSize)
        '''
        for i in range(len(top_k)):
            acerto = 0
            nao_acerto = 0
    
            if top_k[i] == 1:
                a=0
                #acuracia_tpo_1 = accuracy_score(y_test, out_1)
                #print('Acuracia top k =1: ', acuracia_tpo_1)
    
            for amostra_a_avaliar in range(len(out)):
    
                lista_das_classes = out[amostra_a_avaliar]['classesDegrees']
                dict_com_classes_na_ordem = sorted(lista_das_classes, key=itemgetter('degree'), reverse=True)
                #f.write(str(dict_com_classes_na_ordem))
    
                classes_na_ordem_descendente = []
                for x in range(len(dict_com_classes_na_ordem)):
                    classes_na_ordem_descendente.append(dict_com_classes_na_ordem[x]['class'])
    
                top_5 = classes_na_ordem_descendente[0:top_k[i]]
    
                if top_k[i] == 1:
                    index_predict_top_1.append(top_5)
                if top_k[i] == 2:
                    index_predict_top_2.append(top_5)
                if top_k[i] == 3:
                    index_predict_top_3.append(top_5)
                if top_k [i] == 4:
                    index_predict_top_4.append (top_5)
                elif top_k[i] == 5:
                    index_predict_top_5.append(top_5)
                elif top_k[i] == 6:
                    index_predict_top_6.append(top_5)
                elif top_k[i] == 7:
                    index_predict_top_7.append(top_5)
                elif top_k[i] == 8:
                    index_predict_top_8.append(top_5)
                elif top_k[i] == 9:
                    index_predict_top_9.append(top_5)
                elif top_k[i] == 10:
                    index_predict_top_10.append(top_5)
                elif top_k[i] == 20:
                    index_predict_top_20.append(top_5)
                elif top_k[i] == 30:
                    index_predict_top_30.append(top_5)
                elif top_k[i] == 40:
                    index_predict_top_40.append(top_5)
                elif top_k[i] == 50:
                    index_predict_top_50.append(top_5)
    
    
                if( y_test[amostra_a_avaliar] in top_5):
                    acerto = acerto + 1
                else:
                    nao_acerto = nao_acerto + 1
    
            acuracia.append(acerto/len(out))
    
        #print("len(out):", len(out))
        #print("TOP-K: ", top_k)
        #print("Acuracia: ",acuracia)
        #f.close()
        path_index_predict = '../results/index_beams_predict/WiSARD/top_k/'+name_of_conf_input+'/'
        for i in range(len(top_k)):
            file_name = 'index_beams_predict_top_'+str(top_k[i])+'.npz'
            npz_index_predict = path_index_predict + file_name
            if top_k[i] == 1:
                np.savez(npz_index_predict, output_classification=index_predict_top_1)
            if top_k[i] == 2:
                np.savez(npz_index_predict, output_classification=index_predict_top_2)
            if top_k[i] == 3:
                np.savez(npz_index_predict, output_classification=index_predict_top_3)
            if top_k[i] == 4:
                np.savez(npz_index_predict, output_classification=index_predict_top_4)
            elif top_k[i] == 5:
                np.savez(npz_index_predict, output_classification=index_predict_top_5)
            elif top_k[i] == 6:
                np.savez(npz_index_predict, output_classification=index_predict_top_6)
            elif top_k[i] == 7:
                np.savez(npz_index_predict, output_classification=index_predict_top_7)
            elif top_k[i] == 8:
                np.savez(npz_index_predict, output_classification=index_predict_top_8)
            elif top_k[i] == 9:
                np.savez(npz_index_predict, output_classification=index_predict_top_9)
            elif top_k[i] == 10:
                np.savez(npz_index_predict, output_classification=index_predict_top_10)
            elif top_k[i] == 20:
                np.savez(npz_index_predict, output_classification=index_predict_top_20)
            elif top_k[i] == 30:
                np.savez(npz_index_predict, output_classification=index_predict_top_30)
            elif top_k[i] == 40:
                np.savez(npz_index_predict, output_classification=index_predict_top_40)
            elif top_k[i] == 50:
                np.savez(npz_index_predict, output_classification=index_predict_top_50)
    
        #npz_index_predict = '../results/index_beams_predict/top_k/' + f'index_beams_predict_top_{top_k[i]}' + '.npz'
        #np.savez (npz_index_predict, output_classification=estimated_beams)
        '''
        print ("-----------------------------")
        print ("TOP-K \t\t|\t Acuracia")
        print("-----------------------------")
        for i in range(len(top_k)):
            if top_k[i] == 1:
                print('K = ', top_k[i], '\t\t|\t ', np.round(score[i],3)), '\t\t|'
            elif top_k[i] == 5:
                print ('K = ', top_k [i], '\t\t|\t ', np.round (score [i], 3)), '\t\t|'
            else:
                print('K = ', top_k[i], '\t|\t ', np.round(score[i],3)), '\t\t|'
        print ("-----------------------------")


    #df_acuracia_wisard_top_k = pd.DataFrame({"Top-K": top_k, "Acuracia": acuracia})
    #path_csv='../results/accuracy/8X32/'+data_input+'/top_k/'
    #print(path_csv+'acuracia_wisard_' + data_input + '_top_k.csv')
    #df_acuracia_wisard_top_k.to_csv(path_csv + 'acuracia_wisard_' + data_input + '_' + data_set + '_top_k.csv', index=False)
    #df_acuracia_wisard_top_k.to_csv (path_csv + 'acuracia_wisard_' + name_of_conf_input + '_top_k.csv',
    #                                 index=False)


    #plot_top_k(top_k, score, data_input, name_of_conf_input=name_of_conf_input)
    return top_k, df_results_wisard_top_k

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

    #tic ()
    #start_time = time.perf_counter_ns()
    star_ = time.process_time_ns()
    wsd.train (data_train, label_train)
    #trainning_time = toc ()
    end_ = time.process_time_ns()
    #end_time = time.perf_counter_ns ()
    #trainning_time_perf_counter = end_time - start_time
    trainning_time_process_time = (end_ - star_)

    #tic ()
    # classify some data
    star_classify_time_process_time = time.process_time_ns()
    #star_classify_perf_counter = time.perf_counter_ns ()
    wisard_result = wsd.classify (data_validation)
    end_classify_time_process_time = time.process_time_ns()
    #end_classify_perf_counter = time.perf_counter_ns ()
    #test_time = toc()

    test_time_process_time = (end_classify_time_process_time - star_classify_time_process_time)
    #test_time_perf_counter = end_classify_perf_counter - star_classify_perf_counter

    return wisard_result, trainning_time_process_time, test_time_process_time

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

def fit_incremental_window(nro_of_episodes, input_type):
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
    s008_data ['lidar_coord'] = np.concatenate ((encoding_coord_s008, data_lidar_s008), axis=1).tolist ()

    # info_of_episode = s008_data[s008_data['Episode'] == 0]

    s009_data = all_info_s009[['Episode']].copy ()
    s009_data ['index_beams'] = beams_s009
    s009_data ['encoding_coord'] = encoding_coord_s009.tolist ()
    s009_data ['lidar'] = data_lidar_s009.tolist ()
    s009_data ['lidar_coord'] = np.concatenate ((encoding_coord_s009, data_lidar_s009), axis=1).tolist ()

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
                elif label_input_type == 'lidar_coord':
                    input_train = s008_data_copy['lidar_coord'].tolist()

                label_test = s009_data[s009_data ['Episode'] == i]['index_beams'].tolist ()
                if label_input_type == 'coord':
                    input_test = s009_data[s009_data ['Episode'] == i]['encoding_coord'].tolist ()
                elif label_input_type == 'lidar':
                    input_test = s009_data[s009_data ['Episode'] == i]['lidar'].tolist ()
                elif label_input_type == 'lidar_coord':
                    input_test = s009_data[s009_data ['Episode'] == i]['lidar_coord'].tolist ()

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
                elif label_input_type == 'lidar_coord':
                    input_test = s009_data [s009_data['Episode'] == i]['lidar_coord'].tolist()

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

def fit_incremental_window_top_k(nro_of_episodes, input_type, s008_data, s009_data):
    print("  __________________________________________________")
    print('/ \t Fit a WiSARD with: INCREMENTAL window top-k /')
    print("  __________________________________________________")
    episode_for_test = np.arange(0, nro_of_episodes, 1)

    label_input_type = input_type
    s008_data_copy = s008_data.copy()

    all_score = []
    all_episodes = []
    all_samples_train = []
    all_samples_test = []


    all_score_top_1 = []
    all_score_top_5 = []
    all_score_top_10 = []
    all_score_top_15 = []
    all_score_top_20 = []
    all_score_top_25 = []
    all_score_top_30 = []

    std_score_top_1 = []
    std_score_top_5 = []
    std_score_top_10 = []
    std_score_top_15 = []
    std_score_top_20 = []
    std_score_top_25 = []
    std_score_top_30 = []

    trainning_time_top_1 = []
    trainning_time_top_5 = []
    trainning_time_top_10 = []
    trainning_time_top_15 = []
    trainning_time_top_20 = []
    trainning_time_top_25 = []
    trainning_time_top_30 = []

    std_trainning_time_top_1 = []
    std_trainning_time_top_5 = []
    std_trainning_time_top_10 = []
    std_trainning_time_top_15 = []
    std_trainning_time_top_20 = []
    std_trainning_time_top_25 = []
    std_trainning_time_top_30 = []


    for i in range (len (episode_for_test)):
        if i in s009_data['Episode'].tolist():
            if i == 0:
                label_train = s008_data_copy['index_beams'].tolist()
                if label_input_type == 'coord':
                    input_train = s008_data_copy['encoding_coord'].tolist()
                elif label_input_type == 'lidar':
                    input_train = s008_data_copy['lidar'].tolist()
                elif label_input_type == 'lidar_coord':
                    input_train = s008_data_copy['lidar_coord'].tolist()

                label_test = s009_data[s009_data['Episode'] == i]['index_beams'].tolist()
                if label_input_type == 'coord':
                    input_test = s009_data[s009_data['Episode'] == i]['encoding_coord'].tolist()
                elif label_input_type == 'lidar':
                    input_test = s009_data[s009_data['Episode'] == i]['lidar'].tolist()
                elif label_input_type == 'lidar_coord':
                    input_test = s009_data[s009_data['Episode'] == i]['lidar_coord'].tolist()

                labels_for_next_train = label_test
                samples_for_next_train = input_test
            else:
                for j in range(len(labels_for_next_train)):
                    label_train.append(labels_for_next_train[j])
                    input_train.append(samples_for_next_train[j])

                label_test = s009_data[s009_data['Episode'] == i]['index_beams'].tolist()
                if label_input_type == 'coord':
                    input_test = s009_data[s009_data['Episode'] == i]['encoding_coord'].tolist()
                elif label_input_type == 'lidar':
                    input_test = s009_data[s009_data['Episode'] == i]['lidar'].tolist()
                elif label_input_type == 'lidar_coord':
                    input_test = s009_data[s009_data['Episode'] == i]['lidar_coord'].tolist()

                labels_for_next_train = label_test
                samples_for_next_train = input_test

            '''
            top_k, all_metrics = beam_selection_top_k_wisard (x_train=input_train,
                                                           x_test=input_test,
                                                           y_train=label_train,
                                                           y_test=label_test,
                                                           address_of_size=44,
                                                           name_of_conf_input=label_input_type)

            all_score.append(np.array(all_metrics['Acuracia']))
            all_score_top_1.append(all_metrics['Acuracia'][0])
            all_score_top_5.append(all_metrics['Acuracia'][1])
            all_score_top_10.append(all_metrics['Acuracia'][2])
            all_score_top_15.append(all_metrics['Acuracia'][3])
            all_score_top_20.append(all_metrics['Acuracia'][4])
            all_score_top_25.append(all_metrics['Acuracia'][5])
            all_score_top_30.append(all_metrics['Acuracia'][6])

            trainning_time_top_1.append(all_metrics['Trainning Time'][0])
            trainning_time_top_5.append(all_metrics['Trainning Time'][1])
            trainning_time_top_10.append(all_metrics['Trainning Time'][2])
            trainning_time_top_15.append(all_metrics['Trainning Time'][3])
            trainning_time_top_20.append(all_metrics['Trainning Time'][4])
            trainning_time_top_25.append(all_metrics['Trainning Time'][5])
            trainning_time_top_30.append(all_metrics['Trainning Time'][6])
            '''
            df_results_wisard_top_k_with_std = beam_selection_with_confidence_interval (input_train,
                                                                                        input_test,
                                                                                        label_train,
                                                                                        label_test,
                                                                                        label_input_type)

            all_score_top_1.append (df_results_wisard_top_k_with_std ['score_mean'] [0])
            all_score_top_5.append (df_results_wisard_top_k_with_std ['score_mean'] [1])
            all_score_top_10.append (df_results_wisard_top_k_with_std ['score_mean'] [2])
            all_score_top_15.append (df_results_wisard_top_k_with_std ['score_mean'] [3])
            all_score_top_20.append (df_results_wisard_top_k_with_std ['score_mean'] [4])
            all_score_top_25.append (df_results_wisard_top_k_with_std ['score_mean'] [5])
            all_score_top_30.append (df_results_wisard_top_k_with_std ['score_mean'] [6])

            std_score_top_1.append (df_results_wisard_top_k_with_std ['score_std'] [0])
            std_score_top_5.append (df_results_wisard_top_k_with_std ['score_std'] [1])
            std_score_top_10.append (df_results_wisard_top_k_with_std ['score_std'] [2])
            std_score_top_15.append (df_results_wisard_top_k_with_std ['score_std'] [3])
            std_score_top_20.append (df_results_wisard_top_k_with_std ['score_std'] [4])
            std_score_top_25.append (df_results_wisard_top_k_with_std ['score_std'] [5])
            std_score_top_30.append (df_results_wisard_top_k_with_std ['score_std'] [6])

            trainning_time_top_1.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [0])
            trainning_time_top_5.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [1])
            trainning_time_top_10.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [2])
            trainning_time_top_15.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [3])
            trainning_time_top_20.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [4])
            trainning_time_top_25.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [5])
            trainning_time_top_30.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [6])

            std_trainning_time_top_1.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [0])
            std_trainning_time_top_5.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [1])
            std_trainning_time_top_10.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [2])
            std_trainning_time_top_15.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [3])
            std_trainning_time_top_20.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [4])
            std_trainning_time_top_25.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [5])
            std_trainning_time_top_30.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [6])

            all_episodes.append(i)
            all_samples_train.append(len(input_train))
            all_samples_test.append(len(input_test))
        else:
            continue

        ## SAVE RESULTS

        df_results_score = pd.DataFrame ({"episode": all_episodes,
                                          "score_mean_top_1": all_score_top_1, "score_std_top_1": std_score_top_1,
                                          "score_mean_top_5": all_score_top_5, "score_std_top_5": std_score_top_5,
                                          "score_mean_top_10": all_score_top_10, "score_std_top_10": std_score_top_10,
                                          "score_mean_top_15": all_score_top_15, "score_std_top_15": std_score_top_15,
                                          "score_mean_top_20": all_score_top_20, "score_std_top_20": std_score_top_20,
                                          "score_mean_top_25": all_score_top_25, "score_std_top_25": std_score_top_25,
                                          "score_mean_top_30": all_score_top_30, "score_std_top_30": std_score_top_30,
                                          "samples_train": all_samples_train, "samples_test": all_samples_test})

        df_results_trainning_time = pd.DataFrame ({"episode": all_episodes,
                                                   "tranning_time_mean_top_1": trainning_time_top_1,
                                                   "tranning_time_std_top_1": std_trainning_time_top_1,
                                                   "tranning_time_mean_top_5": trainning_time_top_5,
                                                   "tranning_time_std_top_5": std_trainning_time_top_5,
                                                   "tranning_time_mean_top_10": trainning_time_top_10,
                                                   "tranning_time_std_top_10": std_trainning_time_top_10,
                                                   "tranning_time_mean_top_15": trainning_time_top_15,
                                                   "tranning_time_std_top_15": std_trainning_time_top_15,
                                                   "tranning_time_mean_top_20": trainning_time_top_20,
                                                   "tranning_time_std_top_20": std_trainning_time_top_20,
                                                   "tranning_time_mean_top_25": trainning_time_top_25,
                                                   "tranning_time_std_top_25": std_trainning_time_top_25,
                                                   "tranning_time_mean_top_30": trainning_time_top_30,
                                                   "tranning_time_std_top_30": std_trainning_time_top_30,
                                                   "samples_train": all_samples_train,
                                                   "samples_test": all_samples_test})


        path_result = '../results/score/Wisard/online/top_k/' + label_input_type + '/incremental_window/results_with_std/'
        df_results_score.to_csv(path_result + 'scores_with_std_incremental_window_top_k.csv',
                                 index=False)
        df_results_trainning_time.to_csv(path_result + 'trainning_time_with_std_incremental_window_top_k.csv', index=False)



        '''
        path_result = '../results/score/Wisard/online/top_k/' + label_input_type + '/incremental_window/'
        score_results = [all_score_top_1, all_score_top_5, all_score_top_10,
                         all_score_top_15, all_score_top_20, all_score_top_25, all_score_top_30]
        trainning_time_results = [trainning_time_top_1, trainning_time_top_5, trainning_time_top_10,
                                  trainning_time_top_15, trainning_time_top_20, trainning_time_top_25,
                                  trainning_time_top_30]

        save_in_csv_all_metrics_top_k (path_result,
                                       'all_results_incremental_window_top_k.csv',
                                       all_episodes,
                                       score_results,
                                       trainning_time_results,
                                       all_samples_train, all_samples_test)
        '''


def save_in_csv_all_metrics_top_k(path_result, file_name, episodes, score, time, samples_train, samples_test):
    headerList = ['episode',
                  'score top-1',
                  'score top-5',
                  'score top-10',
                  'score top-15',
                  'score top-20',
                  'score top-25',
                  'score top-30',
                  'trainning Time top-1',
                  'trainning Time top-5',
                  'trainning Time top-10',
                  'trainning Time top-15',
                  'trainning Time top-20',
                  'trainning Time top-25',
                  'trainning Time top-30',
                  'samples train',
                  'samples test']

    with open(path_result + file_name, 'w') as f:
        writer_results = csv.writer(f, delimiter=',')
        writer_results.writerow(headerList)
        writer_results.writerows(zip(episodes,
                                     score[0], score[1], score[2], score[3], score[4], score[5], score[6],
                                     time[0], time[1], time[2], time[3], time[4], time[5], time[6],
                                     samples_train, samples_test))





def fit_sliding_window_with_size_variation_top_k(nro_of_episodes,
                                                 input_type,
                                                 window_size,
                                                 s008_data,
                                                 s009_data):
    print(" _____________________________________________")
    print('/ \t Fit a WiSARD with: SLIDING window top-k \t/')
    print(" --------------------------------------------")
    episode_for_test = np.arange(0, nro_of_episodes, 1)

    label_input_type = input_type

    all_score = []
    all_results_top_1 =[]
    all_results = []
    all_episodes = []
    all_samples_train = []
    all_samples_test = []

    all_score_top_1 = []
    all_score_top_5 = []
    all_score_top_10 = []
    all_score_top_15 = []
    all_score_top_20 = []
    all_score_top_25 = []
    all_score_top_30 = []

    std_score_top_1 = []
    std_score_top_5 = []
    std_score_top_10 = []
    std_score_top_15 = []
    std_score_top_20 = []
    std_score_top_25 = []
    std_score_top_30 = []

    trainning_time_top_1 = []
    trainning_time_top_5 = []
    trainning_time_top_10 = []
    trainning_time_top_15 = []
    trainning_time_top_20 = []
    trainning_time_top_25 = []
    trainning_time_top_30 = []

    std_trainning_time_top_1 = []
    std_trainning_time_top_5 = []
    std_trainning_time_top_10 = []
    std_trainning_time_top_15 = []
    std_trainning_time_top_20 = []
    std_trainning_time_top_25 = []
    std_trainning_time_top_30 = []

    start_index_s009 = 0

    nro_episodes_s008 = 2085
    for i in range(len(episode_for_test)):
    #for i in tqdm(range(len(episode_for_test))):
        if i in s009_data['Episode'].tolist():
            if i == 0:
                start_index_s008 = nro_episodes_s008 - window_size
                input_train, label_train = extract_training_data_from_s008(s008_data, start_index_s008, label_input_type)
                input_test, label_test = extract_test_data_from_s009(i, label_input_type, s009_data)
            else:
                start_index_s008 = (nro_episodes_s008 - window_size)+i
                if start_index_s008 < nro_episodes_s008:
                    start_index_s009 = 0
                    end_index_s009 = window_size - (nro_episodes_s008 - start_index_s008)

                    input_train_s008, label_train_s008 = extract_training_data_from_s008(s008_data,
                                                                                         start_index_s008,
                                                                                         label_input_type)
                    input_train_s009, label_train_s009 = extract_training_data_from_s009(s009_data,
                                                                                         start_index_s009,
                                                                                         end_index_s009,
                                                                                         label_input_type)
                    input_train = input_train_s008 + input_train_s009
                    label_train = label_train_s008 + label_train_s009

                    input_test, label_test = extract_test_data_from_s009(i, label_input_type, s009_data)

                else:
                    end_index_s009 = start_index_s009 + window_size
                    input_train, label_train = extract_training_data_from_s009(s009_data,
                                                                               start_index_s009,
                                                                               end_index_s009,
                                                                               label_input_type)
                    input_test, label_test = extract_test_data_from_s009(i, label_input_type, s009_data)
                    start_index_s009 += 1

            df_results_wisard_top_k_with_std = beam_selection_with_confidence_interval (input_train,
                                                                                             input_test,
                                                                                             label_train,
                                                                                             label_test,
                                                                                             label_input_type)

            '''
            top_k, all_metrics = beam_selection_top_k_wisard (x_train=input_train,
                                                              x_test=input_test,
                                                              y_train=label_train,
                                                              y_test=label_test,
                                                              address_of_size=44,
                                                              name_of_conf_input=label_input_type)


            '''
            all_score.append(np.array(df_results_wisard_top_k_with_std['score_mean']))

            all_score_top_1.append(df_results_wisard_top_k_with_std['score_mean'][0])
            all_score_top_5.append(df_results_wisard_top_k_with_std['score_mean'][1])
            all_score_top_10.append(df_results_wisard_top_k_with_std['score_mean'][2])
            all_score_top_15.append(df_results_wisard_top_k_with_std['score_mean'][3])
            all_score_top_20.append(df_results_wisard_top_k_with_std['score_mean'][4])
            all_score_top_25.append(df_results_wisard_top_k_with_std['score_mean'][5])
            all_score_top_30.append(df_results_wisard_top_k_with_std['score_mean'][6])

            std_score_top_1.append(df_results_wisard_top_k_with_std['score_std'][0])
            std_score_top_5.append(df_results_wisard_top_k_with_std['score_std'][1])
            std_score_top_10.append(df_results_wisard_top_k_with_std['score_std'][2])
            std_score_top_15.append(df_results_wisard_top_k_with_std['score_std'][3])
            std_score_top_20.append(df_results_wisard_top_k_with_std['score_std'][4])
            std_score_top_25.append(df_results_wisard_top_k_with_std['score_std'][5])
            std_score_top_30.append(df_results_wisard_top_k_with_std['score_std'][6])

            trainning_time_top_1.append(df_results_wisard_top_k_with_std['tranning_time_mean'][0])
            trainning_time_top_5.append(df_results_wisard_top_k_with_std['tranning_time_mean'][1])
            trainning_time_top_10.append(df_results_wisard_top_k_with_std['tranning_time_mean'][2])
            trainning_time_top_15.append(df_results_wisard_top_k_with_std['tranning_time_mean'][3])
            trainning_time_top_20.append(df_results_wisard_top_k_with_std['tranning_time_mean'][4])
            trainning_time_top_25.append(df_results_wisard_top_k_with_std['tranning_time_mean'][5])
            trainning_time_top_30.append(df_results_wisard_top_k_with_std['tranning_time_mean'][6])

            std_trainning_time_top_1.append(df_results_wisard_top_k_with_std['tranning_time_std'][0])
            std_trainning_time_top_5.append(df_results_wisard_top_k_with_std['tranning_time_std'][1])
            std_trainning_time_top_10.append(df_results_wisard_top_k_with_std['tranning_time_std'][2])
            std_trainning_time_top_15.append(df_results_wisard_top_k_with_std['tranning_time_std'][3])
            std_trainning_time_top_20.append(df_results_wisard_top_k_with_std['tranning_time_std'][4])
            std_trainning_time_top_25.append(df_results_wisard_top_k_with_std['tranning_time_std'][5])
            std_trainning_time_top_30.append(df_results_wisard_top_k_with_std['tranning_time_std'][6])

            all_episodes.append(i)
            all_samples_train.append(len(input_train))
            all_samples_test.append(len(input_test))
        else:
            continue


            ## SAVE RESULTS
        #df_results_wisard_top_k_with_std['Episode'] = all_episodes
        #df_results_wisard_top_k_with_std['Samples Train'] = all_samples_train
        #df_results_wisard_top_k_with_std['Samples Test'] = all_samples_test

        df_results_score = pd.DataFrame({"episode": all_episodes,
                                         "score_mean_top_1": all_score_top_1, "score_std_top_1": std_score_top_1,
                                         "score_mean_top_5": all_score_top_5, "score_std_top_5": std_score_top_5,
                                         "score_mean_top_10": all_score_top_10, "score_std_top_10": std_score_top_10,
                                         "score_mean_top_15": all_score_top_15, "score_std_top_15": std_score_top_15,
                                         "score_mean_top_20": all_score_top_20, "score_std_top_20": std_score_top_20,
                                         "score_mean_top_25": all_score_top_25, "score_std_top_25": std_score_top_25,
                                         "score_mean_top_30": all_score_top_30, "score_std_top_30": std_score_top_30,
                                         "samples_train": all_samples_train, "samples_test": all_samples_test})

        df_results_trainning_time = pd.DataFrame({"episode": all_episodes,
                                                  "tranning_time_mean_top_1": trainning_time_top_1, "tranning_time_std_top_1": std_trainning_time_top_1,
                                                  "tranning_time_mean_top_5": trainning_time_top_5, "tranning_time_std_top_5": std_trainning_time_top_5,
                                                  "tranning_time_mean_top_10": trainning_time_top_10, "tranning_time_std_top_10": std_trainning_time_top_10,
                                                  "tranning_time_mean_top_15": trainning_time_top_15, "tranning_time_std_top_15": std_trainning_time_top_15,
                                                  "tranning_time_mean_top_20": trainning_time_top_20, "tranning_time_std_top_20": std_trainning_time_top_20,
                                                  "tranning_time_mean_top_25": trainning_time_top_25, "tranning_time_std_top_25": std_trainning_time_top_25,
                                                  "tranning_time_mean_top_30": trainning_time_top_30, "tranning_time_std_top_30": std_trainning_time_top_30,
                                                  "samples_train": all_samples_train, "samples_test": all_samples_test})


        path_result = '../results/score/Wisard/online/top_k/' + label_input_type + '/sliding_window/window_size_var/results_with_std/'
        df_results_score.to_csv(path_result + 'scores_with_std_sliding_window_'+str(window_size)+'_top_k.csv', index=False)
        df_results_trainning_time.to_csv(path_result + 'trainning_time_with_std_sliding_window_'+str(window_size)+'_top_k.csv', index=False)



def fit_sliding_window_with_size_var(nro_of_episodes, input_type, window_size):
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
    s008_data['lidar_coord'] = np.concatenate((encoding_coord_s008, data_lidar_s008), axis=1).tolist()

    s009_data = all_info_s009[['Episode']].copy()
    s009_data['index_beams'] = beams_s009
    s009_data['encoding_coord'] = encoding_coord_s009.tolist()
    s009_data['lidar'] = data_lidar_s009.tolist()
    s009_data['lidar_coord'] = np.concatenate((encoding_coord_s009, data_lidar_s009), axis=1).tolist()

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

    start_index_s009 = 0

    nro_episodes_s008 = 2085
    for i in range(len(episode_for_test)):
        if i in s009_data['Episode'].tolist ():
            if i == 0:
                start_index_s008 = nro_episodes_s008 - window_size
                input_train, label_train = extract_training_data_from_s008(s008_data, start_index_s008, label_input_type)
                input_test, label_test = extract_test_data_from_s009(i, label_input_type, s009_data)
            else:
                start_index_s008 = (nro_episodes_s008 - window_size)+i
                if start_index_s008 < nro_episodes_s008:
                    start_index_s009 = 0
                    end_index_s009 = window_size - (nro_episodes_s008 - start_index_s008)

                    input_train_s008, label_train_s008 = extract_training_data_from_s008(s008_data,
                                                                               start_index_s008,
                                                                               label_input_type)
                    input_train_s009, label_train_s009 = extract_training_data_from_s009(s009_data,
                                                                               start_index_s009,
                                                                               end_index_s009,
                                                                               label_input_type)
                    input_train = input_train_s008 + input_train_s009
                    label_train = label_train_s008 + label_train_s009

                    input_test, label_test = extract_test_data_from_s009(i, label_input_type, s009_data)

                else:
                    end_index_s009 = start_index_s009 + window_size
                    input_train, label_train = extract_training_data_from_s009(s009_data,
                                                                               start_index_s009,
                                                                               end_index_s009,
                                                                               label_input_type)
                    input_test, label_test = extract_test_data_from_s009(i, label_input_type, s009_data)
                    start_index_s009 += 1




            index_predict, trainning_time, test_time = beam_selection_wisard (data_train=input_train,
                                                                              data_validation=input_test,
                                                                              label_train=label_train,
                                                                              addressSize=44)

            score = accuracy_score(label_test, index_predict)
            all_score.append (score)
            all_trainning_time.append (trainning_time)
            all_test_time.append (test_time)
            all_episodes.append(i)
            all_samples_train.append (len (input_train))
            all_samples_test.append (len (input_test))
        else:
            continue

    average_score = []
    for i in range(len(all_score)):
        i = i+1
        average_score.append(np.mean(all_score[0:i]))

    path_result = '../results/score/Wisard/online/' + label_input_type + '/sliding_window/window_size_var/'
    plt.plot (all_episodes, all_score, 'o--', color='red', label='Accuracy per episode')
    plt.plot(all_episodes, average_score, 'o-', color='blue', label='Cumulative average accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right', bbox_to_anchor=(1.04, 0))
    plt.title('Beam selection using WiSARD with ' + label_input_type + ' in sliding window with variable size')
    plt.savefig(path_result + 'score_sliding_window_size_'+str(window_size)+'.png')
    plt.close ()

    plt.plot (all_episodes, all_trainning_time, 'o-', color='green')
    plt.xlabel ('Episode')
    plt.ylabel ('Trainning Time')
    plt.title ('Trainning Time using fit with sliding window')
    plt.savefig (path_result +'time_train_sliding_window_size_'+str(window_size)+'.png')
    plt.close ()

    plt.plot (all_episodes, all_test_time, 'o-', color='blue')
    plt.xlabel ('Episode')
    plt.ylabel ('Test Time')
    plt.title ('Test Time using fit with sliding window')
    plt.savefig (path_result +'time_test_sliding_window_size_'+str(window_size)+'.png')
    plt.close ()

    headerList = ['Episode', 'Score', 'Trainning Time', 'Test Time', 'Samples Train', 'Samples Test']

    with open(path_result + 'all_results_sliding_window_size_'+str(window_size)+'.csv', 'w') as f:
        writer_results = csv.writer (f, delimiter=',')
        writer_results.writerow (headerList)
        writer_results.writerows(zip(all_episodes,
                                     all_score,
                                     all_trainning_time, all_test_time,
                                     all_samples_train, all_samples_test))
def extract_test_data_from_s009(episode, label_input_type, s009_data):
    label_test = s009_data [s009_data ['Episode'] == episode] ['index_beams'].tolist ()

    input_test = []

    if label_input_type == 'coord':
        input_test = s009_data [s009_data ['Episode'] == episode] ['encoding_coord'].tolist ()
    elif label_input_type == 'lidar':
        input_test = s009_data [s009_data ['Episode'] == episode] ['lidar'].tolist ()
    elif label_input_type == 'lidar_coord':
        input_test = s009_data [s009_data ['Episode'] == episode] ['lidar_coord'].tolist ()
    else:
        print ('error: deve especificar o tipo de entrada')

    return input_test, label_test
def extract_training_data_from_s008(s008_data, start_index, input_type):

    initial_data_for_trainning = s008_data [s008_data ['Episode'] > start_index]
    label_train = initial_data_for_trainning ['index_beams'].tolist ()
    input_train = []

    if input_type == 'coord':
        input_train = initial_data_for_trainning ['encoding_coord'].tolist ()
    elif input_type == 'lidar':
        input_train = initial_data_for_trainning ['lidar'].tolist ()
    elif input_type == 'lidar_coord':
        input_train = initial_data_for_trainning ['lidar_coord'].tolist ()
    else:
        print('error: deve especificar o tipo de entrada')

    return input_train, label_train
def extract_training_data_from_s009(s009_data, start_index, end_index, input_type):


    data_for_trainnig = s009_data.loc[(s009_data['Episode'] >= start_index) & (s009_data['Episode'] < end_index)]

    label_train = data_for_trainnig['index_beams'].tolist()

    input_train = []
    if input_type == 'coord':
        input_train = data_for_trainnig['encoding_coord'].tolist()
    elif input_type == 'lidar':
        input_train = data_for_trainnig['lidar'].tolist()
    elif input_type == 'lidar_coord':
        input_train = data_for_trainnig['lidar_coord'].tolist()

    return input_train, label_train

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
    s008_data['lidar_coord'] = np.concatenate((encoding_coord_s008, data_lidar_s008), axis=1).tolist()

    #info_of_episode = s008_data[s008_data['Episode'] == 0]

    s009_data = all_info_s009[['Episode']].copy()
    s009_data['index_beams'] = beams_s009
    s009_data['encoding_coord'] = encoding_coord_s009.tolist()
    s009_data['lidar'] = data_lidar_s009.tolist()
    s009_data['lidar_coord'] = np.concatenate((encoding_coord_s009, data_lidar_s009), axis=1).tolist()


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
                elif label_input_type == 'lidar_coord':
                    input_train = s008_data_copy['lidar_coord'].tolist()

                label_test = s009_data[s009_data['Episode'] == i]['index_beams'].tolist()
                if label_input_type == 'coord':
                    input_test = s009_data[s009_data['Episode'] == i]['encoding_coord'].tolist()
                elif label_input_type == 'lidar':
                    input_test = s009_data[s009_data['Episode'] == i]['lidar'].tolist()
                elif label_input_type == 'lidar_coord':
                    input_test = s009_data[s009_data['Episode'] == i]['lidar_coord'].tolist()

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
                elif label_input_type == 'lidar_coord':
                    input_train = s008_data_copy['lidar_coord'].tolist()
                for j in range(len(labels_for_next_train)):
                    label_train.append(labels_for_next_train[j])
                    input_train.append(samples_for_next_train[j])

                label_test = s009_data[s009_data ['Episode'] == i]['index_beams'].tolist()
                if label_input_type == 'coord':
                    input_test = s009_data[s009_data ['Episode'] == i]['encoding_coord'].tolist()
                elif label_input_type == 'lidar':
                    input_test = s009_data[s009_data ['Episode'] == i]['lidar'].tolist()
                elif label_input_type == 'lidar_coord':
                    input_test = s009_data[s009_data ['Episode'] == i]['lidar_coord'].tolist()
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
def fit_sliding_window_top_k(nro_of_episodes, input_type):
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
    s008_data['index_beams'] = beams_s008.tolist ()
    s008_data['encoding_coord'] = encoding_coord_s008.tolist ()
    s008_data['lidar'] = data_lidar_s008.tolist ()
    s008_data['lidar_coord'] = np.concatenate ((encoding_coord_s008, data_lidar_s008), axis=1).tolist ()

    # info_of_episode = s008_data[s008_data['Episode'] == 0]

    s009_data = all_info_s009 [['Episode']].copy ()
    s009_data ['index_beams'] = beams_s009
    s009_data ['encoding_coord'] = encoding_coord_s009.tolist ()
    s009_data ['lidar'] = data_lidar_s009.tolist ()
    s009_data ['lidar_coord'] = np.concatenate ((encoding_coord_s009, data_lidar_s009), axis=1).tolist ()

    # episode_for_test = np.arange(0, 2000, 1)
    episode_for_test = np.arange (0, nro_of_episodes, 1)

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

    all_score_top_5 = []
    all_score_top_10 = []
    all_score_top_15 = []
    all_score_top_20 = []
    all_score_top_25 = []
    all_score_top_30 = []

    for i in range (len (episode_for_test)):
        if i in s009_data ['Episode'].tolist ():

            if i == 0:
                label_train = s008_data_copy ['index_beams'].tolist ()
                if label_input_type == 'coord':
                    input_train = s008_data_copy ['encoding_coord'].tolist ()
                elif label_input_type == 'lidar':
                    input_train = s008_data_copy ['lidar'].tolist ()
                elif label_input_type == 'lidar_coord':
                    input_train = s008_data_copy ['lidar_coord'].tolist ()

                label_test = s009_data [s009_data ['Episode'] == i] ['index_beams'].tolist ()
                if label_input_type == 'coord':
                    input_test = s009_data [s009_data ['Episode'] == i] ['encoding_coord'].tolist ()
                elif label_input_type == 'lidar':
                    input_test = s009_data [s009_data ['Episode'] == i] ['lidar'].tolist ()
                elif label_input_type == 'lidar_coord':
                    input_test = s009_data [s009_data ['Episode'] == i] ['lidar_coord'].tolist ()

                for k in range (len (label_test)):
                    labels_for_next_train.append (label_test [k])
                    samples_for_next_train.append (input_test [k])

                # print(i, len(label_train), len(label_test))
            else:
                s008_data_copy = s008_data_copy.drop (s008_data_copy [s008_data_copy ['Episode'] == i - 1].index)
                label_train = s008_data_copy ['index_beams'].tolist ()
                if label_input_type == 'coord':
                    input_train = s008_data_copy ['encoding_coord'].tolist ()
                elif label_input_type == 'lidar':
                    input_train = s008_data_copy ['lidar'].tolist ()
                elif label_input_type == 'lidar_coord':
                    input_train = s008_data_copy ['lidar_coord'].tolist ()
                for j in range (len (labels_for_next_train)):
                    label_train.append (labels_for_next_train [j])
                    input_train.append (samples_for_next_train [j])

                label_test = s009_data [s009_data ['Episode'] == i] ['index_beams'].tolist ()
                if label_input_type == 'coord':
                    input_test = s009_data [s009_data ['Episode'] == i] ['encoding_coord'].tolist ()
                elif label_input_type == 'lidar':
                    input_test = s009_data [s009_data ['Episode'] == i] ['lidar'].tolist ()
                elif label_input_type == 'lidar_coord':
                    input_test = s009_data [s009_data ['Episode'] == i] ['lidar_coord'].tolist ()
                for k in range (len (label_test)):
                    labels_for_next_train.append (label_test [k])
                    samples_for_next_train.append (input_test [k])
            top_k, acuracia = beam_selection_top_k_wisard (x_train=input_train,
                                                           x_test=input_test,
                                                           y_train=label_train,
                                                           y_test=label_test,
                                                           address_of_size=44,
                                                           name_of_conf_input=label_input_type)

            # score = accuracy_score (label_test, acuracia)
            all_score.append (np.array (acuracia ['Acuracia']))
            all_score_top_5.append (acuracia ['Acuracia'] [0])
            all_score_top_10.append (acuracia ['Acuracia'] [1])
            all_score_top_15.append (acuracia ['Acuracia'] [2])
            all_score_top_20.append (acuracia ['Acuracia'] [3])
            all_score_top_25.append (acuracia ['Acuracia'] [4])
            all_score_top_30.append (acuracia ['Acuracia'] [5])

            # all_trainning_time.append (trainning_time)
            # all_test_time.append (test_time)

            all_episodes.append (i)
            all_samples_train.append (len (input_train))
            all_samples_test.append (len (input_test))
        else:
            continue

            ## SAVE RESULTS
        path_result = '../results/score/Wisard/online/top_k/' + label_input_type + '/sliding_window/'

        headerList = ['Episode', 'score top-5',
                      'score top-10',
                      'score top-15',
                      'score top-20',
                      'score top-25',
                      'score top-30',
                      'Samples Train',
                      'Samples Test']

        with open (path_result + 'all_results_sliding_window.csv', 'w') as f:
            writer_results = csv.writer (f, delimiter=',')
            writer_results.writerow (headerList)
            writer_results.writerows (zip (all_episodes, all_score_top_5,
                                           all_score_top_10,
                                           all_score_top_15,
                                           all_score_top_20,
                                           all_score_top_25,
                                           all_score_top_30,
                                           all_samples_train, all_samples_test))



def prepare_data_for_simulation():
    preprocess_resolution = 16
    th = 0.15
    all_info_s009, encoding_coord_s009, beams_s009 = read_s009_data (preprocess_resolution)
    all_info_s008, encoding_coord_s008, beams_s008 = read_s008_data (preprocess_resolution)

    data_lidar_2D_with_rx_s008, data_lidar_2D_with_rx_s009 = pre_process_lidar.process_all_data_2D_with_rx_like_thermometer()
    data_lidar_s008, data_lidar_s009 = pre_process_lidar.data_lidar_2D_binary_without_variance(
        data_lidar_2D_with_rx_s008,
        data_lidar_2D_with_rx_s009,
        th)

    s008_data = all_info_s008 [['Episode']].copy ()
    s008_data ['index_beams'] = beams_s008.tolist ()
    s008_data ['encoding_coord'] = encoding_coord_s008.tolist ()
    s008_data ['lidar'] = data_lidar_s008.tolist ()
    s008_data ['lidar_coord'] = np.concatenate ((encoding_coord_s008, data_lidar_s008), axis=1).tolist ()

    s009_data = all_info_s009 [['Episode']].copy ()
    s009_data ['index_beams'] = beams_s009
    s009_data ['encoding_coord'] = encoding_coord_s009.tolist ()
    s009_data ['lidar'] = data_lidar_s009.tolist ()
    s009_data ['lidar_coord'] = np.concatenate ((encoding_coord_s009, data_lidar_s009), axis=1).tolist ()

    return s008_data, s009_data

def beam_selection_with_confidence_interval(input_train, input_test,
                                            label_train, label_test,
                                            label_input_type):
    all_score_top_1 = []
    all_score_top_5 = []
    all_score_top_10 = []
    all_score_top_15 = []
    all_score_top_20 = []
    all_score_top_25 = []
    all_score_top_30 = []

    trainning_time_top_1 = []
    trainning_time_top_5 = []
    trainning_time_top_10 = []
    trainning_time_top_15 = []
    trainning_time_top_20 = []
    trainning_time_top_25 = []
    trainning_time_top_30 = []

    experiments = 2
    for i in range(experiments):
        top_k, all_metrics = beam_selection_top_k_wisard (x_train=input_train,
                                                          x_test=input_test,
                                                          y_train=label_train,
                                                          y_test=label_test,
                                                          address_of_size=44,
                                                          name_of_conf_input=label_input_type)


        all_score_top_1.append(all_metrics['Acuracia'][0])
        all_score_top_5.append(all_metrics['Acuracia'][1])
        all_score_top_10.append(all_metrics['Acuracia'][2])
        all_score_top_15.append(all_metrics['Acuracia'][3])
        all_score_top_20.append(all_metrics['Acuracia'][4])
        all_score_top_25.append(all_metrics['Acuracia'][5])
        all_score_top_30.append(all_metrics['Acuracia'][6])

        trainning_time_top_1.append(all_metrics['Trainning Time'][0])
        trainning_time_top_5.append(all_metrics['Trainning Time'][1])
        trainning_time_top_10.append(all_metrics['Trainning Time'][2])
        trainning_time_top_15.append(all_metrics['Trainning Time'][3])
        trainning_time_top_20.append(all_metrics['Trainning Time'][4])
        trainning_time_top_25.append(all_metrics['Trainning Time'][5])
        trainning_time_top_30.append(all_metrics['Trainning Time'][6])



    mean_of_score = [np.mean(all_score_top_1), np.mean(all_score_top_5),
                        np.mean(all_score_top_10), np.mean(all_score_top_15),
                        np.mean(all_score_top_20), np.mean(all_score_top_25),
                        np.mean(all_score_top_30)]

    std_of_score = [np.std(all_score_top_1), np.std(all_score_top_5),
                    np.std(all_score_top_10), np.std(all_score_top_15),
                    np.std(all_score_top_20), np.std(all_score_top_25),
                    np.std(all_score_top_30)]

    mean_of_trainning_time = [np.mean(trainning_time_top_1), np.mean(trainning_time_top_5),
                                np.mean(trainning_time_top_10), np.mean(trainning_time_top_15),
                                np.mean(trainning_time_top_20), np.mean(trainning_time_top_25),
                                np.mean(trainning_time_top_30)]

    std_of_trainning_time = [np.std(trainning_time_top_1), np.std(trainning_time_top_5),
                            np.std(trainning_time_top_10), np.std(trainning_time_top_15),
                            np.std(trainning_time_top_20), np.std(trainning_time_top_25),
                            np.std(trainning_time_top_30)]



    df_results_wisard_top_k = pd.DataFrame ({"Top-K": top_k,
                                             "score_mean": mean_of_score,
                                             "score_std": std_of_score,
                                             "tranning_time_mean": mean_of_trainning_time,
                                             "tranning_time_std": std_of_trainning_time})

    return df_results_wisard_top_k


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
    s008_data['lidar_coord'] = np.concatenate((encoding_coord_s008, data_lidar_s008), axis=1).tolist()

    s009_data = all_info_s009[['Episode']].copy()
    s009_data['index_beams'] = beams_s009
    s009_data['encoding_coord'] = encoding_coord_s009.tolist()
    s009_data['lidar'] = data_lidar_s009.tolist()
    s009_data['lidar_coord'] = np.concatenate((encoding_coord_s009, data_lidar_s009), axis=1).tolist()

    episode_for_test = np.arange (0, nro_of_episodes, 1)

    all_score = []
    all_trainning_time = []
    all_trainning_time_perf_counter = []
    all_trainning_time_process_time = []
    all_test_time = []
    all_test_time_perf_counter = []
    all_test_time_process_time = []
    all_episodes = []
    all_samples_train = []
    all_samples_test = []

    label_train = s008_data['index_beams'].tolist()
    if label_input_type == 'coord':
        input_train = s008_data['encoding_coord'].tolist()
    elif label_input_type == 'lidar':
        input_train = s008_data['lidar'].tolist()
    elif label_input_type == 'lidar_coord':
        input_train = s008_data['lidar_coord'].tolist()

    for i in range(len(episode_for_test)):
        if i in s009_data['Episode'].tolist():
            label_test = s009_data[s009_data['Episode'] == i]['index_beams'].tolist()
            if label_input_type == 'coord':
                input_test = s009_data[s009_data['Episode'] == i]['encoding_coord'].tolist()
            elif label_input_type == 'lidar':
                input_test = s009_data[s009_data['Episode'] == i]['lidar'].tolist()
            elif label_input_type == 'lidar_coord':
                input_test = s009_data[s009_data['Episode'] == i]['lidar_coord'].tolist()

            index_predict, trainning_time, test_time = beam_selection_wisard (data_train=input_train,
                                                                              data_validation=input_test,
                                                                              label_train=label_train,
                                                                              addressSize=44)

            #index_predict, trainning_time, trainning_time_perf_counter, trainning_time_process_time, test_time, test_time_perf_counter, test_time_process_time = beam_selection_wisard (data_train=input_train,
            #                                                                      data_validation=input_test,
            #                                                                      label_train=label_train,
            #                                                                      addressSize=44)

            score = accuracy_score (label_test, index_predict)
            all_score.append(score)
            all_trainning_time.append(trainning_time)
            #all_trainning_time_perf_counter.append(trainning_time_perf_counter)
            #all_trainning_time_process_time.append(trainning_time_process_time)

            all_test_time.append(test_time)
            #all_test_time_perf_counter.append(test_time_perf_counter)
            #all_test_time_process_time.append(test_time_process_time)

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
    plt.savefig(path_result + 'time_train_traditional_fit.png')
    plt.close()

    plt.plot(all_episodes, all_test_time, 'o-', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Test Time')
    plt.title('Test Time using Traditional fit ')
    plt.savefig(path_result + 'time_test_traditional_fit.png')
    plt.close()


    headerList = ['Episode', 'Score', 'Trainning Time', 'Test Time', 'Samples Train', 'Samples Test']

    with open (path_result + 'all_results_traditional_fit.csv', 'w') as f:
        writer_results = csv.writer(f, delimiter=',')
        writer_results.writerow(headerList)
        writer_results.writerows(zip(all_episodes, all_score, all_trainning_time, all_test_time, all_samples_train, all_samples_test))
def fit_fixed_window_top_k(nro_of_episodes, label_input_type, s008_data, s009_data):
    print("  __________________________________________________")
    print('/ Fit a WiSARD with: FIXED window top-k /')
    print("  __________________________________________________")
    print(label_input_type)
    episode_for_test = np.arange(0, nro_of_episodes, 1)

    all_score = []
    all_episodes = []
    all_samples_train = []
    all_samples_test = []


    all_score_top_1 = []
    all_score_top_5 = []
    all_score_top_10 = []
    all_score_top_15 = []
    all_score_top_20 = []
    all_score_top_25 = []
    all_score_top_30 = []

    std_score_top_1 = []
    std_score_top_5 = []
    std_score_top_10 = []
    std_score_top_15 = []
    std_score_top_20 = []
    std_score_top_25 = []
    std_score_top_30 = []

    trainning_time_top_1 = []
    trainning_time_top_5 = []
    trainning_time_top_10 = []
    trainning_time_top_15 = []
    trainning_time_top_20 = []
    trainning_time_top_25 = []
    trainning_time_top_30 = []

    std_trainning_time_top_1 = []
    std_trainning_time_top_5 = []
    std_trainning_time_top_10 = []
    std_trainning_time_top_15 = []
    std_trainning_time_top_20 = []
    std_trainning_time_top_25 = []
    std_trainning_time_top_30 = []


    label_train = s008_data['index_beams'].tolist ()
    if label_input_type == 'coord':
        input_train = s008_data['encoding_coord'].tolist()
    elif label_input_type == 'lidar':
        print('Beam selection using' + label_input_type)
        input_train = s008_data['lidar'].tolist()
    elif label_input_type == 'lidar_coord':
        input_train = s008_data['lidar_coord'].tolist()

    for i in range(len(episode_for_test)):
        if i in s009_data['Episode'].tolist():
            label_test = s009_data[s009_data['Episode'] == i]['index_beams'].tolist()
            if label_input_type == 'coord':
                input_test = s009_data[s009_data['Episode'] == i]['encoding_coord'].tolist()
            elif label_input_type == 'lidar':
                input_test = s009_data[s009_data['Episode'] == i]['lidar'].tolist()
            elif label_input_type == 'lidar_coord':
                input_test = s009_data[s009_data['Episode'] == i]['lidar_coord'].tolist()

            #index_predict, trainning_time, test_time = beam_selection_wisard (data_train=input_train,
            #                                                                  data_validation=input_test,
            #                                                                  label_train=label_train,
            #                                                                  addressSize=44)
            '''
            top_k, all_metrics = beam_selection_top_k_wisard (x_train=input_train,
                                                              x_test=input_test,
                                                              y_train=label_train,
                                                              y_test=label_test,
                                                              address_of_size=44,
                                                              name_of_conf_input=label_input_type)


            # score = accuracy_score (label_test, acuracia)
            all_score.append (np.array (all_metrics ['Acuracia']))
            all_score_top_1.append (all_metrics ['Acuracia'] [0])
            all_score_top_5.append (all_metrics ['Acuracia'] [1])
            all_score_top_10.append (all_metrics ['Acuracia'] [2])
            all_score_top_15.append (all_metrics ['Acuracia'] [3])
            all_score_top_20.append (all_metrics ['Acuracia'] [4])
            all_score_top_25.append (all_metrics ['Acuracia'] [5])
            all_score_top_30.append (all_metrics ['Acuracia'] [6])

            trainning_time_top_1.append (all_metrics ['Trainning Time'] [0])
            trainning_time_top_5.append (all_metrics ['Trainning Time'] [1])
            trainning_time_top_10.append (all_metrics ['Trainning Time'] [2])
            trainning_time_top_15.append (all_metrics ['Trainning Time'] [3])
            trainning_time_top_20.append (all_metrics ['Trainning Time'] [4])
            trainning_time_top_25.append (all_metrics ['Trainning Time'] [5])
            trainning_time_top_30.append (all_metrics ['Trainning Time'] [6])
            '''

            df_results_wisard_top_k_with_std = beam_selection_with_confidence_interval (input_train,
                                                                                        input_test,
                                                                                        label_train,
                                                                                        label_test,
                                                                                        label_input_type)

            all_score_top_1.append (df_results_wisard_top_k_with_std ['score_mean'] [0])
            all_score_top_5.append (df_results_wisard_top_k_with_std ['score_mean'] [1])
            all_score_top_10.append (df_results_wisard_top_k_with_std ['score_mean'] [2])
            all_score_top_15.append (df_results_wisard_top_k_with_std ['score_mean'] [3])
            all_score_top_20.append (df_results_wisard_top_k_with_std ['score_mean'] [4])
            all_score_top_25.append (df_results_wisard_top_k_with_std ['score_mean'] [5])
            all_score_top_30.append (df_results_wisard_top_k_with_std ['score_mean'] [6])

            std_score_top_1.append (df_results_wisard_top_k_with_std ['score_std'] [0])
            std_score_top_5.append (df_results_wisard_top_k_with_std ['score_std'] [1])
            std_score_top_10.append (df_results_wisard_top_k_with_std ['score_std'] [2])
            std_score_top_15.append (df_results_wisard_top_k_with_std ['score_std'] [3])
            std_score_top_20.append (df_results_wisard_top_k_with_std ['score_std'] [4])
            std_score_top_25.append (df_results_wisard_top_k_with_std ['score_std'] [5])
            std_score_top_30.append (df_results_wisard_top_k_with_std ['score_std'] [6])

            trainning_time_top_1.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [0])
            trainning_time_top_5.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [1])
            trainning_time_top_10.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [2])
            trainning_time_top_15.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [3])
            trainning_time_top_20.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [4])
            trainning_time_top_25.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [5])
            trainning_time_top_30.append (df_results_wisard_top_k_with_std ['tranning_time_mean'] [6])

            std_trainning_time_top_1.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [0])
            std_trainning_time_top_5.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [1])
            std_trainning_time_top_10.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [2])
            std_trainning_time_top_15.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [3])
            std_trainning_time_top_20.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [4])
            std_trainning_time_top_25.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [5])
            std_trainning_time_top_30.append (df_results_wisard_top_k_with_std ['tranning_time_std'] [6])


            all_episodes.append (i)
            all_samples_train.append (len (input_train))
            all_samples_test.append (len (input_test))
        else:
            continue

    ## SAVE RESULTS

    df_results_score = pd.DataFrame ({"episode": all_episodes,
                                      "score_mean_top_1": all_score_top_1, "score_std_top_1": std_score_top_1,
                                      "score_mean_top_5": all_score_top_5, "score_std_top_5": std_score_top_5,
                                      "score_mean_top_10": all_score_top_10, "score_std_top_10": std_score_top_10,
                                      "score_mean_top_15": all_score_top_15, "score_std_top_15": std_score_top_15,
                                      "score_mean_top_20": all_score_top_20, "score_std_top_20": std_score_top_20,
                                      "score_mean_top_25": all_score_top_25, "score_std_top_25": std_score_top_25,
                                      "score_mean_top_30": all_score_top_30, "score_std_top_30": std_score_top_30,
                                      "samples_train": all_samples_train, "samples_test": all_samples_test})

    df_results_trainning_time = pd.DataFrame ({"episode": all_episodes,
                                               "tranning_time_mean_top_1": trainning_time_top_1,
                                               "tranning_time_std_top_1": std_trainning_time_top_1,
                                               "tranning_time_mean_top_5": trainning_time_top_5,
                                               "tranning_time_std_top_5": std_trainning_time_top_5,
                                               "tranning_time_mean_top_10": trainning_time_top_10,
                                               "tranning_time_std_top_10": std_trainning_time_top_10,
                                               "tranning_time_mean_top_15": trainning_time_top_15,
                                               "tranning_time_std_top_15": std_trainning_time_top_15,
                                               "tranning_time_mean_top_20": trainning_time_top_20,
                                               "tranning_time_std_top_20": std_trainning_time_top_20,
                                               "tranning_time_mean_top_25": trainning_time_top_25,
                                               "tranning_time_std_top_25": std_trainning_time_top_25,
                                               "tranning_time_mean_top_30": trainning_time_top_30,
                                               "tranning_time_std_top_30": std_trainning_time_top_30,
                                               "samples_train": all_samples_train, "samples_test": all_samples_test})

    print ('save results')
    path_result = '../results/score/Wisard/online/top_k/' + label_input_type + '/fixed_window/results_with_std/'
    df_results_score.to_csv(path_result + 'scores_with_std_fixed_window_top_k.csv',
                             index=False)
    df_results_trainning_time.to_csv(
        path_result + 'trainning_time_with_std_fixed_window_top_k.csv', index=False)


    '''
    path_result = '../results/score/Wisard/online/top_k/' + label_input_type + '/fixed_window/'

    score_results = [all_score_top_1, all_score_top_5, all_score_top_10,
                     all_score_top_15, all_score_top_20, all_score_top_25, all_score_top_30]
    trainning_time_results = [trainning_time_top_1, trainning_time_top_5, trainning_time_top_10,
                              trainning_time_top_15, trainning_time_top_20, trainning_time_top_25,
                              trainning_time_top_30]
    

    save_results = False
    if save_results:
        save_in_csv_all_metrics_top_k (path_result,
                                       'all_results_fixed_window_top_k.csv',
                                       all_episodes,
                                       score_results,
                                       trainning_time_results,
                                       all_samples_train, all_samples_test)
    return score_results, trainning_time_results
    '''
def fit_fixed_window(nro_of_episodes_test, nro_of_episodes_train, input_type):
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
    s008_data ['lidar_coord'] = np.concatenate ((encoding_coord_s008, data_lidar_s008), axis=1).tolist ()

    # info_of_episode = s008_data[s008_data['Episode'] == 0]

    s009_data = all_info_s009 [['Episode']].copy ()
    s009_data ['index_beams'] = beams_s009
    s009_data ['encoding_coord'] = encoding_coord_s009.tolist ()
    s009_data ['lidar'] = data_lidar_s009.tolist ()
    s009_data ['lidar_coord'] = np.concatenate ((encoding_coord_s009, data_lidar_s009), axis=1).tolist ()

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

    all_score = []
    all_trainning_time = []
    #all_trainning_time_perf_counter = []
    #all_trainning_time_process_time = []
    all_test_time = []
    #all_test_time_perf_counter = []
    #all_test_time_process_time = []
    all_episodes = []
    all_samples_train = []
    all_samples_test = []


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
            elif label_input_type == 'lidar_coord':
                input.append(s008_data[s008_data['Episode'] == i]['lidar_coord'].tolist())
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
            elif label_input_type == 'lidar_coord':
                input_test = s009_data [s009_data ['Episode'] == i] ['lidar_coord'].tolist ()

                # print(i, len(label_train), len(label_test))

            index_predict, trainning_time, test_time = beam_selection_wisard (data_train=input_train,
                                                                              data_validation=input_test,
                                                                              label_train=label_train,
                                                                              addressSize=44)

            #index_predict, trainning_time, trainning_time_perf_counter, trainning_time_process_time, test_time, test_time_perf_counter, test_time_process_time = beam_selection_wisard (
            #    data_train=input_train,
            #    data_validation=input_test,
            #    label_train=label_train,
            #    addressSize=44)

            score = accuracy_score(label_test, index_predict)
            all_score.append(score)
            all_trainning_time.append(trainning_time)
            #all_trainning_time_perf_counter.append(trainning_time_perf_counter)
            #all_trainning_time_process_time.append(trainning_time_process_time)

            all_test_time.append(test_time)
            #all_test_time_perf_counter.append (test_time_perf_counter)
            #all_test_time_process_time.append (test_time_process_time)

            all_episodes.append (i)
            all_samples_train.append (len (input_train))
            all_samples_test.append (len (input_test))

            #score = accuracy_score (label_test, index_predict)
            #all_score.append (score)
            #all_trainning_time.append (trainning_time)
            #all_test_time.append (test_time)
            #all_episodes.append (i)
            #all_samples_train.append (len (input_train))
            #all_samples_test.append (len (input_test))
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
    #plt.savefig (path_result + str(rodada) +'score_fixed_window.png')
    plt.savefig (path_result +  'score_fixed_window.png')
    # plt.show ()
    plt.close ()

    plt.plot (all_episodes, all_trainning_time, 'o-', color='green')
    plt.xlabel ('Episode')
    plt.ylabel ('Trainning Time')
    plt.title ('Trainning Time using fit with fixed window')
    plt.savefig (path_result + 'time_train_fixed_window.png')
    plt.close ()

    plt.plot(all_episodes, all_test_time, 'o-', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Test Time')
    plt.title('Test Time using fit with fixed window')
    plt.savefig(path_result + 'time_test_fixed_window.png')
    plt.close()

    headerList = ['Episode', 'Score', 'Trainning Time', 'Test Time', 'Samples Train', 'Samples Test']
    #headerList = ['Episode', 'Score', 'Trainning Time', 'Trainning Time perf_counter', 'Trainning Time process_time',
    #              'Test Time', 'Test Time perf_counter', 'Test Time process_time',  'Samples Train', 'Samples Test']

    with open (path_result  +'all_results_fixed_window.csv', 'w') as f:
        writer_results = csv.writer (f, delimiter=',')
        writer_results.writerow (headerList)
        writer_results.writerows (zip (all_episodes,all_score,
                                       all_trainning_time,
                                       all_test_time,
                                       all_samples_train, all_samples_test))

def plot_top_k_score_comparation_between_sliding_incremental_fixed_window(input_type, simulation_type):
    path_result = '../results/score/Wisard/online/top_k/' + input_type + '/' + simulation_type + '/'
    data = pd.read_csv(path_result + 'all_results_' + simulation_type + '_top_k.csv')

    mean_score_top_1 = calculate_mean_score (data ['score top-1'])
    mean_score_top_5 = calculate_mean_score (data ['score top-5'])
    mean_score_top_10 = calculate_mean_score (data ['score top-10'])
    mean_score_top_15 = calculate_mean_score (data ['score top-15'])
    mean_score_top_20 = calculate_mean_score (data ['score top-20'])
    mean_score_top_25 = calculate_mean_score (data ['score top-25'])
    mean_score_top_30 = calculate_mean_score (data ['score top-30'])

    mean_top_1 = np.round (np.mean (mean_score_top_1), 3)
    mean_top_5 = np.round (np.mean (mean_score_top_5), 3)
    mean_top_10 = np.round (np.mean (mean_score_top_10), 3)
    mean_top_15 = np.round (np.mean (mean_score_top_15), 3)
    mean_top_20 = np.round (np.mean (mean_score_top_20), 3)
    mean_top_25 = np.round (np.mean (mean_score_top_25), 3)
    mean_top_30 = np.round (np.mean (mean_score_top_30), 3)

    key_color = ['blue', 'red', 'green', 'purple', 'orange', 'magenta', 'olive']
    key_text = [mean_top_1, mean_top_5, mean_top_10, mean_top_15, mean_top_20, mean_top_25, mean_top_30]
    key_plot = [mean_score_top_1, mean_score_top_5, mean_score_top_10, mean_score_top_15, mean_score_top_20, mean_score_top_25, mean_score_top_30]
    key_label = ['Top-1', 'Top-5', 'Top-10', 'Top-15', 'Top-20', 'Top-25', 'Top-30']

    sns.set_style("darkgrid")  # whitegrid")
    key_acess_data = 'episode'
    for i in range(len(key_plot)):
        plt.plot(data[key_acess_data], key_plot[i], '.', color=key_color[i], label=key_label[i])

    '''
    if input_type == 'coord':
        pos_x = [1800, 1800, 1800, 1800, 1800, 1800]
        pos_y = [mean_top_1 + 0.06, mean_top_5 + 0.003, mean_top_10 + 0.001,
                 mean_top_15 + 0.001, mean_top_30 + 0.001, mean_top_25 + 0.007,
                 mean_top_30 + 0.013]

    if input_type == 'lidar':
        pos_x = [1800, 1800, 1800, 1800, 1800, 1800]
        pos_y = [mean_top_1 + 0.06, mean_top_5 - 0.03, mean_top_10 - 0.03,
                 mean_top_5 - 0.06, mean_top_5 - 0.09, mean_top_5 - 0.13,
                 mean_top_5 - 0.16]

    if input_type == 'lidar_coord':
        pos_x = [1800]
        pos_y = [mean_top_1 + 0.03, mean_top_1 + 0.06,
                 mean_top_1 + 0.09, mean_top_1 + 0.12,
                 mean_top_1 + 0.15, mean_top_1 + 0.18,
                 mean_top_1 + 0.21]
    '''

    pos_x = [1800]
    pos_y = [mean_top_1 + 0.03, mean_top_1 + 0.06,
             mean_top_1 + 0.09, mean_top_1 + 0.12,
             mean_top_1 + 0.15, mean_top_1 + 0.18,
             mean_top_1 + 0.21]


    for i in range(len(pos_y)):
        plt.text(pos_x[0], pos_y[i],
                 'Mean: ' + str(key_text[i]),
                 fontsize=8, fontweight='bold',
                 color=key_color[i]) #fontname='Myanmar Sangam MN',

    plt.title('Beam selection WiSARD using ' + input_type +
              ' \n in online learning with ' + simulation_type + ' top-k',
              fontsize=12, color='black', fontweight='bold')
    plt.xlabel ('Episode', fontweight='bold')
    plt.ylabel ('Accuracy', fontweight='bold')
    plt.legend (loc='best', ncol=3, fontsize=8)  # bbox_to_anchor=(1, 1.15)

    plt.savefig (path_result + 'score_comparation_top_k.png', dpi=300)
    plt.show ()
def read_csv_data(path, filename):
    data = pd.read_csv(path + filename)
    return data

def calculate_mean_score(data):
    #all_score = data ['Score'].tolist ()
    average_score = []
    for i in range (len (data)):
        i = i + 1
        average_score.append (np.mean (data [0:i]))
    return average_score

def plot_compare_windows_size_in_window_sliding(input_name):
    path_result = '../results/score/Wisard/online/'+input_name+'/sliding_window/window_size_var/'
    window_size = [100,  500, 1000, 1500, 2000]
    color = ['blue', 'red', 'green', 'purple', 'orange', 'maroon', 'teal', 'black', 'gray', 'brown', 'cyan', 'magenta', 'yellow', 'olive', 'navy', 'lime', 'aqua', 'fuchsia', 'silver', 'white']

    text_pos_y = 1800
    for i in range(len(window_size)):

        file_name = 'all_results_sliding_window_size_' + str(window_size[i]) + '.csv'
        data = read_csv_data(path_result, file_name)
        mean_cumulative_score = calculate_mean_score(data['Score'].tolist())
        window_size_label = 'Window size: ' + str(window_size[i])
        mean_label = 'Mean: ' + str(np.round(np.mean(mean_cumulative_score),3))
        plt.plot(data['Episode'], mean_cumulative_score, '.', label=window_size_label, color=color[i])
        plt.text(text_pos_y, 0.4, mean_label,
                 color=color[i], fontname='Myanmar Sangam MN', fontweight='bold',
                 fontdict=dict(fontsize=6, fontweight='bold'), bbox=dict(facecolor='white', edgecolor='white'))

        text_pos_y = text_pos_y - 300

    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend(loc='best', ncol=2, fontsize=6)
    plt.title('Beam selection using WiSARD with '+ input_name + '\n in online learning with sliding window varying the window size')
    plt.savefig(path_result + 'score_comparation_window_size.png', dpi=300)
    plt.show()
def plot_score_comparation_between_sliding_incremental_fixed_window(input_type):
    path_result = '../results/score/Wisard/online/'
    #path_result_traditional = path_result + input_type+'/traditional_fit/'
    #all_results_traditional = pd.read_csv(path_result_traditional + 'all_results_traditional_fit.csv')
    path_result_sliding_window = path_result + input_type+'/sliding_window/'
    all_results_sliding_window = pd.read_csv(path_result_sliding_window + 'all_results_sliding_window.csv')
    path_result_incremental_window = path_result + input_type+'/incremental_window/'
    all_results_incremental_window = pd.read_csv(path_result_incremental_window + 'all_results_incremental_window.csv')
    path_result_fixed_window = path_result + input_type+'/fixed_window/'
    all_results_fixed_window = pd.read_csv(path_result_fixed_window + 'all_results_fixed_window.csv')

    #mean_cumulative_score_traditional = calculate_mean_score(all_results_traditional)
    mean_cumulative_sliding_window = calculate_mean_score(all_results_sliding_window)
    mean_cumulative_incremental_window = calculate_mean_score(all_results_incremental_window)
    mean_cumulative_score_fixed_window = calculate_mean_score(all_results_fixed_window)

    mean_sliding_window = np.round(np.mean(mean_cumulative_sliding_window),3)
    #mean_traditional = np.round(np.mean(mean_cumulative_score_traditional),3)
    mean_incremental_window = np.round(np.mean(mean_cumulative_incremental_window),3)
    mean_fixed_window = np.round(np.mean(mean_cumulative_score_fixed_window),3)

    text_sliding_window = 'Mean: '+str(mean_sliding_window)
    text_incremental_window = 'Mean: '+str(mean_incremental_window)
    text_fixed_window = 'Mean: '+str(mean_fixed_window)
    #text_traditional_window = 'Mean: '+str(mean_traditional)

    plt.plot(all_results_fixed_window['Episode'], mean_cumulative_score_fixed_window, 'o-', color='purple', label='Fixed window')
    plt.text(1800, mean_fixed_window-0.01, text_fixed_window, fontsize=8, color='purple', fontname='Myanmar Sangam MN', fontweight='bold')
    plt.plot(all_results_sliding_window['Episode'], mean_cumulative_sliding_window, 'o-', color='red', label='Sliding window')
    plt.text(1800, mean_sliding_window+0.03, text_sliding_window, fontsize=8, color='red', fontname='Myanmar Sangam MN', fontweight='bold')
    plt.plot(all_results_incremental_window['Episode'], mean_cumulative_incremental_window, 'o-', color='green', label='Incremental window')
    plt.text(1800, mean_incremental_window, text_incremental_window, fontsize=8, color='green', fontname='Myanmar Sangam MN', fontweight='bold')

    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Beam selection using WiSARD with \n'+input_type+' in online learning')
    plt.savefig(path_result + input_type + '/score_comparation.png', dpi=300)
    plt.close()
def plot_top_K_time_and_score_comparition_sliding_incremental_fixed_window(input_type, top_k):
    path_result = '../results/score/Wisard/online/top_k/'

    path_result_sliding_window = path_result + input_type + '/sliding_window/window_size_var/'
    path_result_incremental_window = path_result + input_type + '/incremental_window/'
    path_result_fixed_window = path_result + input_type + '/fixed_window/'

    all_results_sliding_window = pd.read_csv (path_result_sliding_window + 'all_results_sliding_window_100_top_k.csv')
    all_results_incremental_window = pd.read_csv(path_result_incremental_window + 'all_results_incremental_window_top_k.csv')
    all_results_fixed_window = pd.read_csv (path_result_fixed_window + 'all_results_fixed_window_top_k.csv')



    window_size = [500, 1000, 1500, 2000]
    color = ['blue', 'red', 'green', 'purple', 'orange']#, 'maroon', 'teal', 'black', 'gray', 'brown', 'cyan', 'magenta', 'yellow', 'olive', 'navy', 'lime', 'aqua', 'fuchsia', 'silver', 'white']
    flag = 'training'
    sns.set_theme (style="darkgrid")
    fig, ax1 = plt.subplots (figsize=(15, 7))
    for i in range(len(window_size)):
        all_results_sliding_window = pd.read_csv(path_result_sliding_window +
                                                 'all_results_sliding_window_' +
                                                 str(window_size[i])+'_top_k.csv')

        plt.plot(all_results_sliding_window['episode'],
                 all_results_sliding_window['trainning Time top-'+str(top_k)]*1e-9,
                 color=color[i], marker=',', alpha=0.3,
                 label='Sliding window top-' + str(top_k) + '_' + str(window_size[i]))



    plt.plot (all_results_fixed_window ['episode'],
              all_results_fixed_window['trainning Time top-' + str(top_k)] * 1e-9,
              color='olive', marker=',', alpha=0.3, label='Fixed window top-'+str(top_k))
    plt.plot (all_results_incremental_window['episode'],
              all_results_incremental_window ['trainning Time top-'+str(top_k)] * 1e-9,
              color='magenta', marker=',', alpha=0.3)

    ax1.set_ylabel(flag + ' time [s]', fontsize=12, color='black', labelpad=10, fontweight='bold',
                    fontname='Myanmar Sangam MN')
    ax1.set_xlabel('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold', fontname='Myanmar Sangam MN')

    # Criando um segundo eixo
    ax2 = ax1.twinx()
    size_font = 10
    x_pos = [1000, 1250, 1500, 1800]
    if top_k == 1:
        y_pos = 0.5
    else:
        y_pos = 0.8
    for i in range(len(window_size)):
        all_results_sliding_window = pd.read_csv(path_result_sliding_window +
                                                 'all_results_sliding_window_' +
                                                 str(window_size[i])+'_top_k.csv')
        mean_cumulative_sliding_window = calculate_mean_score(all_results_sliding_window['score top-'+str(top_k)])
        mean = (np.mean(mean_cumulative_sliding_window))
        plt.plot(all_results_sliding_window['episode'],
                  mean_cumulative_sliding_window,
                  marker=',', color=color[i], label='sliding window - '+str(window_size[i]))
        plt.text(x_pos[i], y_pos,
                 'Mean: ' + str(np.round(mean, 3)),
                 fontsize=size_font, color=color[i], fontname='Myanmar Sangam MN', fontweight='bold')

    mean_cumulative_fixed_window = calculate_mean_score(all_results_fixed_window['score top-'+str(top_k)])
    mea_fixed_window = (np.mean(mean_cumulative_fixed_window))
    plt.plot (all_results_fixed_window ['episode'],
              mean_cumulative_fixed_window,
              marker=',', color='olive', alpha=0.3, label='fixed window')
    plt.text (500, y_pos,
              'Mean: ' + str (np.round (mea_fixed_window, 3)),
              fontsize=size_font, color='olive', fontname='Myanmar Sangam MN', fontweight='bold')

    mean_cumulative_incremental_window = calculate_mean_score(all_results_incremental_window['score top-'+str(top_k)])
    mean_incremental_window = (np.mean(mean_cumulative_incremental_window))
    plt.plot(all_results_incremental_window['episode'],
              mean_cumulative_incremental_window,
              marker=',', color='magenta',  label='incremental window')
    plt.text(750, y_pos,
              'Mean: ' + str(np.round(mean_incremental_window, 3)),
              fontsize=size_font, color='magenta', fontname='Myanmar Sangam MN', fontweight='bold')

    ax2.set_ylabel('Score', fontsize=12, color='black', labelpad=12, fontweight='bold',
                    fontname='Myanmar Sangam MN')  # , color='red')

    # Adicionando título e legendas
    title = "Relationship between  training time and score TOP-" + str(top_k) +"\n using data: " + input_type
    plt.title(title, fontsize=15, color='black', fontweight='bold')
    plt.xlabel('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')
    plt.legend(loc='best', ncol=3)  # loc=(0,-0.4), ncol=3)#loc='best')
    plt.savefig(path_result + input_type + '/comparition_score_time_episode_top_'+str(top_k)+'.png', dpi=300)
    plt.close()
    #plt.show()

def plot_time_and_samples_comparition_sliding_incremental_fixed_window(input_type):
    path_result = '../results/score/Wisard/online/'
    #path_result_traditional = path_result + input_type+'/traditional_fit/'
    path_result_sliding_window = path_result + input_type+'/sliding_window/'
    path_result_incremental_window = path_result + input_type+'/incremental_window/'
    path_result_fixed_window = path_result + input_type+'/fixed_window/'

    #all_results_traditional = pd.read_csv(path_result_traditional + 'all_results_traditional_fit.csv')
    all_results_sliding_window = pd.read_csv(path_result_sliding_window + 'all_results_sliding_window.csv')
    all_results_incremental_window = pd.read_csv(path_result_incremental_window + 'all_results_incremental_window.csv')
    all_results_fixed_window = pd.read_csv(path_result_fixed_window + 'all_results_fixed_window.csv')

    flag = 'training'
    sns.set_theme(style="darkgrid")
    fig, ax1 = plt.subplots (figsize=(15, 7))
    plt.plot(all_results_fixed_window['Episode'],
             all_results_fixed_window['Trainning Time']*1e-9,
             color='purple')
    plt.plot(all_results_sliding_window['Episode'],
                all_results_sliding_window['Trainning Time']*1e-9,
                color='red')
    plt.plot(all_results_incremental_window['Episode'],
                all_results_incremental_window['Trainning Time']*1e-9,
                color='green')


    ax1.set_ylabel(flag + ' time [s]' , fontsize=12, color='black', labelpad=10, fontweight='bold', fontname = 'Myanmar Sangam MN')
    ax1.set_xlabel('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold', fontname = 'Myanmar Sangam MN')


    # Criando um segundo eixo
    ax2 = ax1.twinx ()
    plt.plot (all_results_fixed_window['Episode'],
             all_results_fixed_window['Samples Train'],
              marker=',', color='purple', alpha=0.3, label='fixed window')
    plt.plot (all_results_sliding_window['Episode'],
                all_results_sliding_window['Samples Train'],
                marker=',', color='red',  alpha=0.3,label='sliding window')
    plt.plot (all_results_incremental_window['Episode'],
                all_results_incremental_window['Samples Train'],
                marker=',', color='green',  alpha=0.3,label='incremental window')

    ax2.set_ylabel ('training samples', fontsize=12, color='black', labelpad=12, fontweight='bold', fontname = 'Myanmar Sangam MN')#, color='red')

    # Adicionando título e legendas
    title = "Relationship between trained samples and training time \n usign data: " + input_type
    plt.title(title, fontsize=15, color='black',  fontweight='bold')
    #plt.xticks(all_results_traditional['Episode'])
    plt.xlabel('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')
    plt.legend(loc='best',  ncol=3)#loc=(0,-0.4), ncol=3)#loc='best')
    plt.savefig (path_result + input_type + '/time_and_samples_train_comparation.png', dpi=300)
    plt.close()


    plt.plot(all_results_fixed_window['Episode'], all_results_fixed_window['Trainning Time']/1e9, '.', color='purple', label='Fixed window')
    plt.plot(all_results_sliding_window['Episode'], all_results_sliding_window['Trainning Time']/1e9, '.', color='red', label='Sliding window')
    plt.plot(all_results_incremental_window['Episode'], all_results_incremental_window['Trainning Time']/1e9, '.', color='green', label='Incremental window')
    plt.xlabel('Episode', fontsize=12, color='black',  fontname = 'Myanmar Sangam MN', fontweight='bold')
    plt.ylabel('Trainning Time', fontsize=12, color='black', fontname = 'Myanmar Sangam MN', fontweight='bold')
    plt.legend(loc='best')#, bbox_to_anchor=(1.04, 0))
    plt.title('Trainning Time using WiSARD with \n'+input_type+' in online learning')
    plt.savefig(path_result + input_type + '/time_train_comparation.png', dpi=300)
    plt.close()

    plt.plot(all_results_fixed_window['Episode'], all_results_fixed_window['Test Time']/1e9, '.', color='purple', label='Fixed window')
    plt.plot(all_results_sliding_window['Episode'], all_results_sliding_window['Test Time']/1e9, '.', color='red', label='Sliding window')
    plt.plot(all_results_incremental_window['Episode'], all_results_incremental_window['Test Time']/1e9, '.', color='green', label='Incremental window')
    plt.xlabel('Episode', fontsize=12, color='black', fontname = 'Myanmar Sangam MN', fontweight='bold')
    plt.ylabel('Test Time', fontsize=12, color='black', fontname = 'Myanmar Sangam MN', fontweight='bold')
    plt.legend(loc='best')#, bbox_to_anchor=(1.04, 0))
    plt.title('Test Time using WiSARD with \n'+input_type+' in online learning', fontsize=12, color='black')
    plt.savefig(path_result + input_type + '/time_test_comparation.png', dpi=300)
    plt.close()

def comparition_between_two_measurements_time():
    path_result = '../results/score/Wisard/online/'
    simulation_type = 'fixed_window' #'sliding_window'#'incremental_window'
    path_result_fixed_window = path_result + input_type +'/'+ simulation_type+'/'

    a = pd.read_csv(path_result_fixed_window + 'all_results_'+simulation_type+'.csv')
    b = pd.read_csv(path_result_fixed_window + 'all_results_'+simulation_type+'_ns.csv')

    plt.plot(a['Episode'], a['Trainning Time']/1e9, 'o-', color='blue', label='Trainning Time -a ')
    plt.plot(b['Episode'], b['Trainning Time']/1e9, 'o-', color='red', label='Trainning Time -b')
    plt.title('Trainning Time comparition between two measurements')
    plt.ylabel('Trainning Time [s]')
    plt.xlabel('Episode')
    plt.legend()
    plt.show()
    plt.savefig(path_result_fixed_window + 'time_trainning_comparation.png')

def comparition_between_types_time_measurement():
    path_result = '../results/score/Wisard/online/'
    path_result_fixed_window = path_result + input_type+'/fixed_window/'

    results_fixed_window_1 = pd.read_csv (path_result_fixed_window + '1all_results_fixed_window.csv')
    sns.set_theme (style="darkgrid")
    fig, ax1 = plt.subplots (figsize=(15, 7))

    plt.plot (results_fixed_window_1 ['Episode'],
                results_fixed_window_1 ['Trainning Time perf_counter']/ 1e+9,
                '--', color='red', label='perf_counter')
    plt.plot (results_fixed_window_1 ['Episode'],
              results_fixed_window_1 ['Trainning Time process_time'] / 1e+9,
              '--', color='blue', label='process_time')
    #plt.plot (results_fixed_window_1 ['Episode'],
    #            results_fixed_window_1 ['Trainning Time'],
    #            '.', color='green', label='time')
    plt.legend()
    plt.title('Trainning Time using fixed window \n comparition between time measurement: \n perf_counter, process_time and time it')
    plt.show()

    sns.set_theme (style="darkgrid")
    fig, ax1 = plt.subplots (figsize=(15, 7))
    plt.plot (results_fixed_window_1 ['Episode'],
              results_fixed_window_1 ['Test Time process_time'] / 1e+9,
              'o', color='blue', label='process_time')
    plt.plot (results_fixed_window_1 ['Episode'],
              results_fixed_window_1 ['Test Time perf_counter'],
              '--', color='red', label='perf_counter')
    plt.plot (results_fixed_window_1 ['Episode'],
              results_fixed_window_1 ['Test Time'],
              '.', color='green', label='time')
    plt.legend ()
    plt.title (
        'Test Time using fixed window \n comparition between time measurement: \n perf_counter, process_time and time it')
    plt.show ()

def plot_comparition_with_standar_desviation(input_type, top_k):
    path_result = '../results/score/Wisard/online/top_k/'

    folder_std_results = 'results_with_std/'
    type_of_window = ['fixed_window', 'incremental_window', 'sliding_window']
    path_result_fixed_window = path_result + input_type + '/' + type_of_window[0] + '/' + folder_std_results
    path_result_incremental_window = path_result + input_type + '/' + type_of_window[1] +'/' + folder_std_results
    path_result_sliding_window = path_result + input_type + '/' + type_of_window[2]+'/window_size_var/' + folder_std_results

    name_of_file_score = 'scores_with_std_'
    name_of_file_time = 'trainning_time_with_std_'
    type_of_file = '_top_k.csv'

    all_scores_fixed_window = pd.read_csv(path_result_fixed_window + name_of_file_score + type_of_window[0] + type_of_file)
    all_times_fixed_window = pd.read_csv(path_result_fixed_window + name_of_file_time + type_of_window[0] + type_of_file)

    all_scores_incremental_window = pd.read_csv(path_result_incremental_window + name_of_file_score + type_of_window[1] + type_of_file)
    all_times_incremental_window = pd.read_csv(path_result_incremental_window + name_of_file_time + type_of_window[1] + type_of_file)



    mean_score_fixed_window = calculate_mean_score (all_scores_fixed_window ['score_mean_top_'+str(top_k)])
    # std_score = np.std(all_results_fixed_window['Score'])
    mean_score_incremental_window = calculate_mean_score (all_scores_incremental_window ['score_mean_top_'+ str(top_k)])

    #window_size = [500, 1000, 1500, 2000]
    window_size = [100]
    color = ['blue', 'red', 'green', 'purple', 'orange']
    # , 'maroon', 'teal', 'black', 'gray', 'brown', 'cyan', 'magenta', 'yellow', 'olive', 'navy', 'lime', 'aqua', 'fuchsia', 'silver', 'white']
    flag = 'training'
    sns.set_theme (style="darkgrid")
    fig, ax1 = plt.subplots (figsize=(15, 7))

    for i in range(len(window_size)):
        all_times_sliding_window = pd.read_csv(path_result_sliding_window +
                                                name_of_file_time + type_of_window[2] + '_' +
                                                str(window_size[i]) + type_of_file)
        plt.plot (all_times_sliding_window['episode'],
                  all_times_sliding_window['tranning_time_mean_top_' + str(top_k)] * 1e-9,
                  #all_times_sliding_window['tranning_time_mean_top_1'],
                  color=color[i], marker=',',
                  label='Sliding window top-' + str(top_k) + '_' + str(window_size[i]))
        plt.fill_between (all_times_sliding_window ['episode'],
                          all_times_sliding_window ['tranning_time_mean_top_1'] * 1e-9 +
                          all_times_fixed_window ['tranning_time_std_top_1'] * 1e-9,
                          all_times_sliding_window ['tranning_time_mean_top_1'] * 1e-9 -
                          all_times_fixed_window ['tranning_time_std_top_1'] * 1e-9,
                          color=color[i], alpha=0.3)

    plt.plot(all_times_fixed_window['episode'],
              all_times_fixed_window['tranning_time_mean_top_1'] * 1e-9,
              marker=',', color='olive', label='Trainning Time')
    plt.fill_between(all_times_fixed_window['episode'],
                      all_times_fixed_window['tranning_time_mean_top_1'] * 1e-9 + all_times_fixed_window['tranning_time_std_top_1'] * 1e-9,
                      all_times_fixed_window['tranning_time_mean_top_1'] * 1e-9 - all_times_fixed_window['tranning_time_std_top_1'] * 1e-9,
                        color='olive', alpha=0.3)
    plt.plot(all_times_incremental_window['episode'],
              all_times_incremental_window['tranning_time_mean_top_1'] * 1e-9,
              marker=',', color='magenta', label='Trainning Time')
    plt.fill_between (all_times_incremental_window ['episode'],
                      all_times_incremental_window ['tranning_time_mean_top_1'] * 1e-9 + all_times_fixed_window [
                          'tranning_time_std_top_1'] * 1e-9,
                      all_times_incremental_window ['tranning_time_mean_top_1'] * 1e-9 - all_times_fixed_window [
                          'tranning_time_std_top_1'] * 1e-9,
                      color='magenta', alpha=0.3)


    ax1.set_ylabel('Trainning time [s]', fontsize=12, color='black', labelpad=10, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')

    # Criando um segundo eixo
    ax2 = ax1.twinx()
    plt.plot(all_scores_fixed_window['episode'], mean_score_fixed_window,
              '-', color='olive',  label='fixed window')
    #plt.fill_between (all_scores_fixed_window ['episode'],
    #                  mean_score_fixed_window + all_scores_fixed_window ['score_std_top_1'],
    #                  mean_score_fixed_window - all_scores_fixed_window ['score_std_top_1'],
    #                  color='grey', alpha=0.3)
    plt.plot (all_scores_incremental_window ['episode'], mean_score_incremental_window,
              '-', color='magenta', label='incremental window')

    for i in range (len (window_size)):
        all_score_sliding_window = pd.read_csv(path_result_sliding_window +
                                                name_of_file_score + type_of_window[2] + '_' +
                                                str(window_size[i]) + type_of_file)
        mean_score_sliding_window = calculate_mean_score (all_score_sliding_window ['score_mean_top_' + str(top_k)])
        plt.plot (all_score_sliding_window ['episode'],
                  mean_score_sliding_window,
                  color=color[i], marker=',',
                  label='Sliding window ' + str(window_size[i]))


    ax2.set_ylabel ('Accuracy', fontsize=12, color='black', labelpad=12, fontweight='bold')  # , color='red')

    plt.text(1800, 0.5, 'Mean: '+str(np.round(np.mean(mean_score_fixed_window),3)), fontsize=8, color='purple', fontname='Myanmar Sangam MN', fontweight='bold')
    plt.title('Beam selection using WiSARD with '+input_type+' in online learning Top-'+str(top_k),  fontsize=14, color='black',  fontweight='bold')
    plt.legend(loc='best')
    plt.savefig(path_result + input_type + '/comparition_score_time_episode_std_top_'+str(top_k)+'.png', dpi=300)
    plt.show()
    #plt.close()
def simulation_of_online_learning_top_k(input_type):
    eposodies_for_test = 2000
    episodes_for_train = 2086

    s008_data, s009_data = prepare_data_for_simulation()


    #fit_fixed_window_top_k(eposodies_for_test, input_type, s008_data, s009_data)
    #plot_top_k_score_comparation_between_sliding_incremental_fixed_window(input_type, simulation_type='fixed_window')
    fit_incremental_window_top_k(eposodies_for_test, input_type, s008_data, s009_data)
    #plot_top_k_score_comparation_between_sliding_incremental_fixed_window(input_type, simulation_type='incremental_window')

    window_size = [100, 500, 1000, 1500, 2000]
    for i in range(len(window_size)):
        print('Window size: ', window_size[i])
        fit_sliding_window_with_size_variation_top_k(nro_of_episodes=eposodies_for_test,
                                                     input_type=input_type,
                                                     window_size=window_size[i],
                                                     s008_data=s008_data,
                                                     s009_data=s009_data)
        a=0


    top_k = [1, 5, 10, 15, 20, 25, 30]
    for i in range(len(top_k)):
        plot_top_K_time_and_score_comparition_sliding_incremental_fixed_window(input_type, top_k[i])

eposodies_for_test = 2000
episodes_for_train = 2086

parser = argparse.ArgumentParser(description="define a type of input: coord, lidar or lidar_coord")
parser.add_argument('--input_type', type=str, default='coord', help='type of input: coord, lidar or lidar_coord')
parser.add_argument('--top_k', type=str, default='False', help='type of input: True or False')
args = parser.parse_args()

input_type = args.input_type
top_k = args.top_k
input_type = 'lidar'

plot_comparition_with_standar_desviation(input_type, '1')
#path = '../data/coord/'
#filename = 'CoordVehiclesRxPerScene_s009.csv'
#dados = pd.read_csv(path+filename)
#dados_validos = dados[dados['Val'] == 'V']

simulation_of_online_learning_top_k(input_type)

#fit_traditional(eposodies_for_test, input_type)
#fit_fixed_window(eposodies_for_test, episodes_for_train, input_type)
#fit_sliding_window(eposodies_for_test, input_type)
#fit_incremental_window(eposodies_for_test, input_type)

#plot_score_comparation_between_sliding_incremental_fixed_window(input_type)

#plot_compare_windows_size_in_window_sliding(input_type)




#comparition_between_two_measurements_time()
# testar tempo de treinamento e teste com esta ferramenta: https://docs.python.org/3/library/time.html#time.perf_counter



