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
    wsd.train(x_train, y_train)

    # the output is a list of string, this represent the classes attributed to each input
    out = wsd.classify(x_test)

    #wsd_1 = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)
    #wsd_1.train(x_train, y_train)
    #out_1 = wsd_1.classify(x_test)


    content_index = 0
    ram_index = 0
    #print(wsd.getsizeof(ram_index,content_index))
    #print(wsd.json())
    #print(out)

    top_k = [5, 10, 15, 20, 25, 30]
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

    df_score_wisard_top_k = pd.DataFrame ({"Top-K": top_k, "Acuracia": score})
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
    return top_k, df_score_wisard_top_k

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

def fit_sliding_window_with_size_var(nro_of_episodes, input_type):
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

    p = 0
    window_size = 100
    nro_episodes_s008 = 2085
    for i in range(len(episode_for_test)):
        if i in s009_data['Episode'].tolist ():
            if i == 0:
                initial_data_for_trainning = s008_data[s008_data['Episode'] > (nro_episodes_s008-window_size)]
                label_train = initial_data_for_trainning['index_beams'].tolist()
                label_test = s009_data [s009_data ['Episode'] == i] ['index_beams'].tolist ()
                if label_input_type == 'coord':
                    input_train = initial_data_for_trainning['encoding_coord'].tolist()
                    input_test = s009_data [s009_data ['Episode'] == i] ['encoding_coord'].tolist ()
                elif label_input_type == 'lidar':
                    input_train = initial_data_for_trainning['lidar'].tolist()
                    input_test = s009_data [s009_data ['Episode'] == i] ['lidar'].tolist ()
                elif label_input_type == 'lidar_coord':
                    input_train = initial_data_for_trainning['lidar_coord'].tolist()
                    input_test = s009_data [s009_data ['Episode'] == i] ['lidar_coord'].tolist ()
                for k in range(len(label_test)):
                    labels_for_next_train.append(label_test[k])
                    samples_for_next_train.append(input_test[k])
            else:
                var = (nro_episodes_s008 - window_size)+i
                if var <= nro_episodes_s008:
                    initial_data_for_trainning = s008_data [s008_data ['Episode'] > var]
                    label_train = initial_data_for_trainning['index_beams'].tolist()
                    label_test = s009_data [s009_data ['Episode'] == i] ['index_beams'].tolist ()
                    if label_input_type == 'coord':
                        input_train = initial_data_for_trainning['encoding_coord'].tolist()
                        input_test = s009_data [s009_data ['Episode'] == i] ['encoding_coord'].tolist ()
                    elif label_input_type == 'lidar':
                        input_train = initial_data_for_trainning['lidar'].tolist()
                        input_test = s009_data [s009_data ['Episode'] == i] ['lidar'].tolist ()
                    elif label_input_type == 'lidar_coord':
                        input_train = initial_data_for_trainning['lidar_coord'].tolist()
                        input_test = s009_data [s009_data ['Episode'] == i] ['lidar_coord'].tolist ()
                    for j in range(len(labels_for_next_train)):
                        label_train.append(labels_for_next_train[j])
                        input_train.append(samples_for_next_train[j])
                    for k in range(len(label_test)):
                        labels_for_next_train.append(label_test[k])
                        samples_for_next_train.append(input_test[k])
                elif var > nro_episodes_s008:
                    aux = s009_data[s009_data['Episode']==i]['Episode'].tolist()
                    if aux[0] < window_size+p:
                        data_for_trainnig_a = s009_data[s009_data['Episode'] <= window_size+p]
                        data_for_trainnig = data_for_trainnig_a[data_for_trainnig_a['Episode'] > p]

                        label_test = s009_data[s009_data['Episode'] == i]['index_beams'].tolist()
                        label_train = data_for_trainnig['index_beams'].tolist()

                        if label_input_type == 'coord':
                            input_test = s009_data [s009_data ['Episode'] == i] ['encoding_coord'].tolist ()
                            input_train = data_for_trainnig['encoding_coord'].tolist()
                        elif label_input_type == 'lidar':
                            input_test = s009_data [s009_data ['Episode'] == i] ['lidar'].tolist ()
                            input_train = data_for_trainnig['lidar'].tolist()

                        elif label_input_type == 'lidar_coord':
                            input_test = s009_data [s009_data ['Episode'] == i] ['lidar_coord'].tolist ()
                            input_train = data_for_trainnig['lidar_coord'].tolist()

                   # if p==0:
                   #     for k in range(len(label_test)):
                   #         labels_for_next_train.append(label_test[k])
                   #         samples_for_next_train.append(input_test[k])
                   # elif p > 0:
                   #     for j in range(len(labels_for_next_train)):
                   #         label_train.append (labels_for_next_train [j])
                   #         input_train.append (samples_for_next_train [j])
                    p = p+1




            #print('i = ',i, initial_data_for_trainning['Episode'].tolist(), len(label_train),
            #      s009_data[s009_data['Episode'] == i]['Episode'].tolist(), len(label_test))
            #print('i = ',i, '|',len(label_train),'|', len(label_test))
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


    '''
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
        '''

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
    #plt.show ()
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

def fit_traditional_top_k(nro_of_episodes, label_input_type):
    preprocess_resolution = 16
    th = 0.15
    # data of coordinates, episodes and beams from s009 and s008
    all_info_s009, encoding_coord_s009, beams_s009 = read_s009_data (preprocess_resolution)
    all_info_s008, encoding_coord_s008, beams_s008 = read_s008_data (preprocess_resolution)

    # data of lidar 2D with rx from s009 and s008
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

    s009_data = all_info_s009 [['Episode']].copy ()
    s009_data ['index_beams'] = beams_s009
    s009_data ['encoding_coord'] = encoding_coord_s009.tolist ()
    s009_data ['lidar'] = data_lidar_s009.tolist ()
    s009_data ['lidar_coord'] = np.concatenate ((encoding_coord_s009, data_lidar_s009), axis=1).tolist ()

    episode_for_test = np.arange (0, nro_of_episodes, 1)

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


    label_train = s008_data['index_beams'].tolist ()
    if label_input_type == 'coord':
        input_train = s008_data ['encoding_coord'].tolist ()
    elif label_input_type == 'lidar':
        input_train = s008_data ['lidar'].tolist ()
    elif label_input_type == 'lidar_coord':
        input_train = s008_data ['lidar_coord'].tolist ()

    for i in range (len (episode_for_test)):
        if i in s009_data ['Episode'].tolist ():
            label_test = s009_data [s009_data ['Episode'] == i] ['index_beams'].tolist ()
            if label_input_type == 'coord':
                input_test = s009_data [s009_data ['Episode'] == i] ['encoding_coord'].tolist ()
            elif label_input_type == 'lidar':
                input_test = s009_data [s009_data ['Episode'] == i] ['lidar'].tolist ()
            elif label_input_type == 'lidar_coord':
                input_test = s009_data [s009_data ['Episode'] == i] ['lidar_coord'].tolist ()

            #index_predict, trainning_time, test_time = beam_selection_wisard (data_train=input_train,
            #                                                                  data_validation=input_test,
            #                                                                  label_train=label_train,
            #                                                                  addressSize=44)
            top_k, acuracia = beam_selection_top_k_wisard(x_train=input_train,
                                        x_test=input_test,
                                        y_train=label_train,
                                        y_test=label_test,
                                        address_of_size=44,
                                        name_of_conf_input=label_input_type)



            #score = accuracy_score (label_test, acuracia)
            all_score.append(np.array(acuracia['Acuracia']))
            all_score_top_5.append(acuracia['Acuracia'][0])
            all_score_top_10.append(acuracia['Acuracia'][1])
            all_score_top_15.append(acuracia['Acuracia'][2])
            all_score_top_20.append(acuracia['Acuracia'][3])
            all_score_top_25.append(acuracia['Acuracia'][4])
            all_score_top_30.append(acuracia['Acuracia'][5])

            #all_trainning_time.append (trainning_time)
            #all_test_time.append (test_time)

            all_episodes.append(i)
            all_samples_train.append (len (input_train))
            all_samples_test.append (len (input_test))
        else:
            continue

    ## SAVE RESULTS
    path_result = '../results/score/Wisard/online/top_k/' + label_input_type + '/traditional_fit/'

    '''
    plt.plot (all_episodes, all_score, 'o--', color='red', label='Accuracy per episode')
    plt.plot (all_episodes, average_score, 'o-', color='blue', label='Cumulative average accuracy')
    plt.xlabel ('Episode')
    plt.ylabel ('Accuracy')
    plt.legend (loc='lower right', bbox_to_anchor=(1.04, 0))
    plt.title ('Beam selection using WiSARD with ' + label_input_type + ' in traditional fit')
    plt.savefig (path_result + 'score_traditional_train.png')
    plt.close ()

    plt.plot (all_episodes, all_trainning_time, 'o-', color='green')
    plt.xlabel ('Episode')
    plt.ylabel ('Trainning Time')
    plt.title ('Trainning Time using Traditional fit ')
    plt.savefig (path_result + 'time_train_traditional_fit.png')
    plt.close ()

    plt.plot (all_episodes, all_test_time, 'o-', color='blue')
    plt.xlabel ('Episode')
    plt.ylabel ('Test Time')
    plt.title ('Test Time using Traditional fit ')
    plt.savefig (path_result + 'time_test_traditional_fit.png')
    plt.close ()
    '''

    headerList = ['Episode', 'score top-5',
                                'score top-10',
                                'score top-15',
                                'score top-20',
                                'score top-25',
                                'score top-30',
                                'Samples Train',
                                'Samples Test']

    with open (path_result + 'all_results_traditional_fit.csv', 'w') as f:
        writer_results = csv.writer (f, delimiter=',')
        writer_results.writerow (headerList)
        writer_results.writerows(zip(all_episodes, all_score_top_5,
                                                    all_score_top_10,
                                                    all_score_top_15,
                                                    all_score_top_20,
                                                    all_score_top_25,
                                                    all_score_top_30,
                                                    all_samples_train, all_samples_test))

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
def plot_score_comparation(input_type):
    path_result = '../results/score/Wisard/online/'
    path_result_traditional = path_result + input_type+'/traditional_fit/'
    path_result_sliding_window = path_result + input_type+'/sliding_window/'
    path_result_incremental_window = path_result + input_type+'/incremental_window/'
    path_result_fixed_window = path_result + input_type+'/fixed_window/'

    all_results_traditional = pd.read_csv(path_result_traditional + 'all_results_traditional_fit.csv')
    all_results_sliding_window = pd.read_csv(path_result_sliding_window + 'all_results_sliding_window.csv')
    all_results_incremental_window = pd.read_csv(path_result_incremental_window + 'all_results_incremental_window.csv')
    #all_results_fixed_window = pd.read_csv(path_result_fixed_window + 'all_results_fixed_window.csv')

    all_score_traditional = all_results_traditional['Score'].tolist()
    average_score_traditional = []
    for i in range (len (all_score_traditional)):
        i = i + 1
        average_score_traditional.append (np.mean (all_score_traditional[0:i]))

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

    #all_score_fixed_window = all_results_fixed_window['Score'].tolist()
    #average_score_fixed_window = []
    #for i in range (len (all_score_fixed_window)):
    #    i = i + 1
    #    average_score_fixed_window.append(np.mean (all_score_fixed_window [0:i]))


    mean_sliding_window = np.round(np.mean(average_score_sliding_window),3)
    mean_traditional = np.round(np.mean(average_score_traditional),3)
    mean_incremental_window = np.round(np.mean(average_score_incremental_window),3)
    #mean_fixed_window = np.round(np.mean(average_score_fixed_window),3)

    text_sliding_window = 'Mean: '+str(mean_sliding_window)
    text_incremental_window = 'Mean: '+str(mean_incremental_window)
    #text_fixed_window = 'Mean: '+str(mean_fixed_window)
    text_traditional_window = 'Mean: '+str(mean_traditional)



    plt.plot(all_results_traditional['Episode'], average_score_traditional, 'o-', color='purple', label='Fixed window')
    plt.text(1800, mean_traditional-0.01, text_traditional_window, fontsize=8, color='purple', fontname='Myanmar Sangam MN', fontweight='bold')
    plt.plot(all_results_sliding_window['Episode'], average_score_sliding_window, 'o-', color='red', label='Sliding window')
    plt.text(1800, mean_sliding_window+0.03, text_sliding_window, fontsize=8, color='red', fontname='Myanmar Sangam MN', fontweight='bold')
    plt.plot(all_results_incremental_window['Episode'], average_score_incremental_window, 'o-', color='green', label='Incremental window')
    plt.text(1800, mean_incremental_window, text_incremental_window, fontsize=8, color='green', fontname='Myanmar Sangam MN', fontweight='bold')
    #plt.plot(all_results_fixed_window['Episode'], average_score_fixed_window, 'o-', color='purple', label='Fixed window')
    #plt.text (1800, mean_fixed_window-0.02, text_fixed_window, fontsize=8, color='purple', fontname='Myanmar Sangam MN', fontweight='bold')

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
    #path_result_fixed_window = path_result + input_type+'/fixed_window/'

    all_results_traditional = pd.read_csv(path_result_traditional + 'all_results_traditional_fit.csv')
    all_results_sliding_window = pd.read_csv(path_result_sliding_window + 'all_results_sliding_window.csv')
    all_results_incremental_window = pd.read_csv(path_result_incremental_window + 'all_results_incremental_window.csv')
    #all_results_fixed_window = pd.read_csv(path_result_fixed_window + 'all_results_fixed_window.csv')

    flag = 'training'
    sns.set_theme (style="darkgrid")
    fig, ax1 = plt.subplots (figsize=(15, 7))
    plt.plot(all_results_traditional['Episode'],
             all_results_traditional['Trainning Time']*1e-9,
             color='purple')
    plt.plot(all_results_sliding_window['Episode'],
                all_results_sliding_window['Trainning Time']*1e-9,
                color='red')
    plt.plot(all_results_incremental_window['Episode'],
                all_results_incremental_window['Trainning Time']*1e-9,
                color='green')
    #plt.plot(all_results_fixed_window['Episode'],
    #            all_results_fixed_window['Trainning Time']*1e-9,
    #            color='purple')


    ax1.set_ylabel(flag + ' time [s]' , fontsize=12, color='black', labelpad=10, fontweight='bold', fontname = 'Myanmar Sangam MN')
    ax1.set_xlabel('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold', fontname = 'Myanmar Sangam MN')


    # Criando um segundo eixo
    ax2 = ax1.twinx ()
    plt.plot (all_results_traditional['Episode'],
             all_results_traditional['Samples Train'],
              marker=',', color='purple', alpha=0.3, label='fixed window')
    plt.plot (all_results_sliding_window['Episode'],
                all_results_sliding_window['Samples Train'],
                marker=',', color='red',  alpha=0.3,label='sliding window')
    plt.plot (all_results_incremental_window['Episode'],
                all_results_incremental_window['Samples Train'],
                marker=',', color='green',  alpha=0.3,label='incremental window')
    #plt.plot (all_results_fixed_window['Episode'],
    #            all_results_fixed_window['Samples Train'],
    #            marker=',', color='purple', alpha=0.3, label='fixed window')


    ax2.set_ylabel ('training samples', fontsize=12, color='black', labelpad=12, fontweight='bold', fontname = 'Myanmar Sangam MN')#, color='red')

    # Adicionando título e legendas
    title = "Relationship between trained samples and training time \n usign data: " + input_type
    plt.title(title, fontsize=15, color='black',  fontweight='bold')
    #plt.xticks(all_results_traditional['Episode'])
    plt.xlabel('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')
    plt.legend(loc='best',  ncol=3)#loc=(0,-0.4), ncol=3)#loc='best')
    plt.savefig (path_result + input_type + '/time_and_samples_train_comparation.png', dpi=300)
    plt.close()


    plt.plot(all_results_traditional['Episode'], all_results_traditional['Trainning Time']/1e9, '.', color='purple', label='Fixed window')
    plt.plot(all_results_sliding_window['Episode'], all_results_sliding_window['Trainning Time']/1e9, '.', color='red', label='Sliding window')
    plt.plot(all_results_incremental_window['Episode'], all_results_incremental_window['Trainning Time']/1e9, '.', color='green', label='Incremental window')
    #plt.plot(all_results_fixed_window['Episode'], all_results_fixed_window['Trainning Time']/1e9, '.', color='purple', label='Fixed window')
    plt.xlabel('Episode', fontsize=12, color='black',  fontname = 'Myanmar Sangam MN', fontweight='bold')
    plt.ylabel('Trainning Time', fontsize=12, color='black', fontname = 'Myanmar Sangam MN', fontweight='bold')
    plt.legend(loc='best')#, bbox_to_anchor=(1.04, 0))
    plt.title('Trainning Time using WiSARD with \n'+input_type+' in online learning')
    plt.savefig(path_result + input_type + '/time_train_comparation.png', dpi=300)
    plt.close()

    plt.plot(all_results_traditional['Episode'], all_results_traditional['Test Time']/1e9, '.', color='purple', label='Fixed window')
    plt.plot(all_results_sliding_window['Episode'], all_results_sliding_window['Test Time']/1e9, '.', color='red', label='Sliding window')
    plt.plot(all_results_incremental_window['Episode'], all_results_incremental_window['Test Time']/1e9, '.', color='green', label='Incremental window')
    #plt.plot(all_results_fixed_window['Episode'], all_results_fixed_window['Test Time']/1e9, '.', color='purple', label='Fixed window')
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



eposodies_for_test = 2000
episodes_for_train = 2086
input_type = 'coord'
#fit_traditional(eposodies_for_test, input_type)
#fit_sliding_window_with_size_var(eposodies_for_test, input_type)
#rodada = 1
#fit_fixed_window(eposodies_for_test, episodes_for_train, input_type, rodada)
#rodada = 2
#fit_fixed_window(eposodies_for_test, episodes_for_train, input_type)
#fit_sliding_window(eposodies_for_test, input_type)
#fit_incremental(eposodies_for_test, input_type)

#plot_score_comparation(input_type)
#plot_time_comparition(input_type)

asdfasdf fit_traditional_top_k(eposodies_for_test, input_type)
#comparition_between_two_measurements_time()
# testar tempo de treinamento e teste com esta ferramenta: https://docs.python.org/3/library/time.html#time.perf_counter


plot = False
if plot:
    path_result = '../results/score/Wisard/online/top_k/coord/traditional_fit/'
    simulation_type = 'traditional_fit' #'sliding_window'#'incremental_window'
    #path_result_fixed_window = path_result + input_type +'/'+ simulation_type+'/'

    a = pd.read_csv(path_result + 'all_results_traditional_fit.csv')
    #b = pd.read_csv(path_result + 'all_results_'+simulation_type+'_ns.csv')
    plt.plot(a['Episode'], a['score top-5'], 'o-', color='blue', label='Top-5')
    plt.plot(a['Episode'], a['score top-10'], 'o-', color='red', label='Top-10')
    plt.show()
