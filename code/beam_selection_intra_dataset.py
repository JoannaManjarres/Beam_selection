import pre_process_coord
import read_data
import beam_selection_LOS_NLOS
import numpy as np
import pandas as pd
import statistics

def beam_selection(input_type, connection_type, dataset):

    #input_type = 'coord'
    scale_to_coord = 16
    #dataset = 's008'
    #connection_type = 'ALL'
    if dataset == 's008':
        data_LOS, NLOS, valid_data = read_data.read_data_s008(scale_to_coord)
        episodes_for_train = 1564
    elif dataset == 's009':
        data_LOS, NLOS, valid_data = read_data.read_data_s009(scale_to_coord)
        episodes_for_train = 1599



    if connection_type == 'ALL':
        data_for_train = valid_data[valid_data['EpisodeID'] <= episodes_for_train]
        data_for_validation = valid_data[valid_data['EpisodeID'] > episodes_for_train]
    if connection_type == 'LOS':
        data_for_train = data_LOS[data_LOS['EpisodeID'] <= episodes_for_train]
        data_for_validation = data_LOS[data_LOS['EpisodeID'] > episodes_for_train]
    if connection_type == 'NLOS':
        data_for_train = NLOS[NLOS['EpisodeID'] <= episodes_for_train]
        data_for_validation = NLOS[NLOS['EpisodeID'] > episodes_for_train]

    # read beams
    index_beams_for_train = data_for_train ['index_beams'].tolist ()
    index_beams_for_validation = data_for_validation ['index_beams'].tolist ()
    label_train = [str (y) for y in index_beams_for_train]
    label_test = [str (y) for y in index_beams_for_validation]

    if input_type == 'coord':
        coord_for_train_encode = data_for_train['enconding_coord'].tolist()
        coord_for_validation_encode = data_for_validation['enconding_coord'].tolist()
        input_train = coord_for_train_encode
        input_validation = coord_for_validation_encode

    elif input_type == 'lidar':
        lidar_for_train = data_for_train ['lidar'].tolist ()
        lidar_for_validation = data_for_validation ['lidar'].tolist ()
        input_train = lidar_for_train
        input_validation = lidar_for_validation

    elif input_type == 'lidar_coord':
        lidar_coord_for_train = data_for_train ['lidar_coord'].tolist ()
        lidar_coord_for_validation = data_for_validation ['lidar_coord'].tolist ()
        input_train = lidar_coord_for_train
        input_validation = lidar_coord_for_validation


    vector_acuracia, address_size = beam_selection_LOS_NLOS.select_best_beam(input_train=input_train,
                                                                             input_validation=input_validation,
                                                                             label_train=label_train,
                                                                             label_validation=label_test,
                                                                             figure_name='',
                                                                             antenna_config='',
                                                                             type_of_input='',
                                                                             titulo_figura='',
                                                                             user='')

    return vector_acuracia, address_size

def calculate_beam_selection_mean_std(input_type, connection_type, dataset):
    #dataset = 's008'
    #connection_type = 'ALL'
    scale_to_coord = 16

    all_accuracy = pd.DataFrame()

    for i in range(10):
        accuracy_coord, size_memory = beam_selection(input_type, connection_type, dataset)
        all_accuracy[f'exper_{i}'] = accuracy_coord
    mean = all_accuracy.mean(axis=1)
    std = all_accuracy.std(axis=1)
    all_accuracy['size_memory'] = size_memory
    all_accuracy['mean'] = mean
    all_accuracy['std'] = std

    path_csv = '../results/score/Wisard/beam_selection_with_s008_or_s009/' + dataset + '/' + input_type + '/'  # + connection_type + '/'
    file_name = 'accuracy_' + input_type + '_res_' + str(scale_to_coord) + '_' + connection_type + '_addres_size.csv'
    all_accuracy.to_csv (path_csv + file_name, index=False)

def read_results_size_memory(dataset, input_type, connection_type):
    path_csv = '../results/score/Wisard/beam_selection_with_s008_or_s009/' + dataset + '/'+input_type+'/'
    file_name = 'accuracy_'+input_type+'_res_16_'+connection_type+'_addres_size.csv'
    df = pd.read_csv(path_csv + file_name)
    return df

def plot_results_size_memory(dataset, input_type):
    import matplotlib.pyplot as plt
    import seaborn as sns

    #dataset = 's009'
    #input_type = 'coord'
    #connection_type = 'ALL'
    results_ALL = read_results_size_memory(dataset=dataset, input_type=input_type, connection_type='ALL')
    results_LOS = read_results_size_memory(dataset=dataset, input_type=input_type, connection_type='LOS')
    results_NLOS = read_results_size_memory(dataset=dataset, input_type=input_type, connection_type='NLOS')

    #sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_ALL, x='size_memory', y='mean', label='Mean ALL', color='green', marker='o')
    plt.fill_between(results_ALL['size_memory'], results_ALL['mean'] - results_ALL['std'], results_ALL['mean'] + results_ALL['std'], alpha=0.2, color='green')

    sns.lineplot(data=results_LOS, x='size_memory', y='mean', label='Mean LOS', color='darkblue', marker='o')
    plt.fill_between(results_LOS['size_memory'], results_LOS['mean'] - results_LOS['std'], results_LOS['mean'] + results_LOS['std'], alpha=0.2, color='darkblue')

    sns.lineplot(data=results_NLOS, x='size_memory', y='mean', label='Mean NLOS', color='C1', marker='o')
    plt.fill_between(results_NLOS['size_memory'], results_NLOS['mean'] - results_NLOS['std'], results_NLOS['mean'] + results_NLOS['std'], alpha=0.2, color='C1')
    plt.title(f'Beam Selection Accuracy for {dataset} with {input_type}')
    plt.xlabel('Address Size')
    plt.ylabel('Accuracy')
    plt.xticks(results_ALL['size_memory'])
    plt.grid()
    plt.legend()
    #plt.show()

    path_to_save = '../results/score/Wisard/beam_selection_with_s008_or_s009/' + dataset + '/'+input_type+'/'
    file_name = input_type+ '_performance_accuracy_all_LOS_NLOS.png'
    plt.savefig (path_to_save + file_name, dpi=300, bbox_inches='tight')

def beam_selection_top_k(input_type, connection_type, dataset, address_size):

    #input_type = 'coord'
    scale_to_coord = 16
    #dataset = 's008'
    #connection_type = 'ALL'
    if dataset == 's008':
        data_LOS, NLOS, valid_data = read_data.read_data_s008(scale_to_coord)
        episodes_for_train = 1564
    elif dataset == 's009':
        data_LOS, NLOS, valid_data = read_data.read_data_s009(scale_to_coord)
        episodes_for_train = 1599



    if connection_type == 'ALL':
        data_for_train = valid_data[valid_data['EpisodeID'] <= episodes_for_train]
        data_for_validation = valid_data[valid_data['EpisodeID'] > episodes_for_train]
    if connection_type == 'LOS':
        data_for_train = data_LOS[data_LOS['EpisodeID'] <= episodes_for_train]
        data_for_validation = data_LOS[data_LOS['EpisodeID'] > episodes_for_train]
    if connection_type == 'NLOS':
        data_for_train = NLOS[NLOS['EpisodeID'] <= episodes_for_train]
        data_for_validation = NLOS[NLOS['EpisodeID'] > episodes_for_train]

    # read beams
    index_beams_for_train = data_for_train ['index_beams'].tolist ()
    index_beams_for_validation = data_for_validation ['index_beams'].tolist ()
    label_train = [str (y) for y in index_beams_for_train]
    label_test = [str (y) for y in index_beams_for_validation]

    if input_type == 'coord':
        coord_for_train_encode = data_for_train['enconding_coord'].tolist()
        coord_for_validation_encode = data_for_validation['enconding_coord'].tolist()
        input_train = coord_for_train_encode
        input_validation = coord_for_validation_encode

    elif input_type == 'lidar':
        lidar_for_train = data_for_train ['lidar'].tolist ()
        lidar_for_validation = data_for_validation ['lidar'].tolist ()
        input_train = lidar_for_train
        input_validation = lidar_for_validation

    elif input_type == 'lidar_coord':
        lidar_coord_for_train = data_for_train ['lidar_coord'].tolist ()
        lidar_coord_for_validation = data_for_validation ['lidar_coord'].tolist ()
        input_train = lidar_coord_for_train
        input_validation = lidar_coord_for_validation


    top_k, acuracia = beam_selection_LOS_NLOS.beam_selection_top_k_wisard(x_train= input_train,
                                                                            x_test=input_validation,
                                                                            y_train=label_train,
                                                                            y_test=label_test,
                                                                            data_input=input_type,
                                                                            data_set=connection_type,
                                                                            address_of_size=address_size,
                                                                            name_of_conf_input=connection_type)
    return top_k, acuracia

def calculate_beam_selection_top_k_mean_std(input_type, connection_type, dataset):
    #dataset = 's008'
    #connection_type = 'ALL'
    address_size = 44

    all_accuracy = pd.DataFrame()

    for i in range(10):
        top_k, acuracia = beam_selection_top_k(input_type, connection_type, dataset, address_size)
        all_accuracy[f'exper_{i}'] = acuracia
    mean = all_accuracy.mean(axis=1)
    std = all_accuracy.std(axis=1)
    all_accuracy['top_k'] = top_k
    all_accuracy['mean'] = mean
    all_accuracy['std'] = std

    path_csv = '../results/score/Wisard/beam_selection_with_s008_or_s009/' + dataset + '/' + input_type + '/'  # + connection_type + '/'
    file_name = 'accuracy_top_k' + input_type + '_addresSize_' + str(address_size) + '_' + connection_type + '.csv'
    all_accuracy.to_csv (path_csv + file_name, index=False)

def read_results_top_k(dataset, input_type, connection_type):
    path_csv = '../results/score/Wisard/beam_selection_with_s008_or_s009/' + dataset + '/'+input_type+'/'
    file_name = 'accuracy_top_k'+input_type+'_addresSize_44_'+connection_type+'.csv'
    df = pd.read_csv(path_csv + file_name)
    return df

def plot_results_TOP_K(dataset):
    import matplotlib.pyplot as plt
    import seaborn as sns

    results_ALL_coord = read_results_top_k(dataset=dataset, input_type='coord', connection_type='ALL')
    results_LOS_coord = read_results_top_k(dataset=dataset, input_type='coord', connection_type='LOS')
    results_NLOS_coord = read_results_top_k(dataset=dataset, input_type='coord', connection_type='NLOS')

    results_LOS_lidar = read_results_top_k(dataset=dataset, input_type='lidar', connection_type='LOS')
    results_NLOS_lidar = read_results_top_k(dataset=dataset, input_type='lidar', connection_type='NLOS')
    results_ALL_lidar = read_results_top_k(dataset=dataset, input_type='lidar', connection_type='ALL')

    results_LOS_lidar_coord = read_results_top_k(dataset=dataset, input_type='lidar_coord', connection_type='LOS')
    results_NLOS_lidar_coord = read_results_top_k(dataset=dataset, input_type='lidar_coord', connection_type='NLOS')
    results_ALL_lidar_coord = read_results_top_k(dataset=dataset, input_type='lidar_coord', connection_type='ALL')

    top_10 = results_ALL_lidar [results_ALL_lidar ['top_k'] <= 10] ['top_k']

    fig, ax = plt.subplots (1, 3, figsize=(14, 6), sharey=True)
    plt.subplots_adjust (left=0.08, right=0.98, bottom=0.1, top=0.9, hspace=0.12, wspace=0.05)
    size_of_font = 12
    marker_size = 5

    accuracy_top_10_coord_all = results_ALL_coord[results_ALL_coord['top_k']<=10]['mean']
    std_top_10_coord_all = results_ALL_coord[results_ALL_coord['top_k']<=10]['std']

    accuracy_top_10_coord_LOS = results_LOS_coord[results_LOS_coord['top_k']<=10]['mean']
    std_top_10_coord_LOS = results_LOS_coord[results_LOS_coord['top_k']<=10]['std']

    accuracy_top_10_coord_NLOS = results_NLOS_coord[results_NLOS_coord['top_k']<=10]['mean']
    std_top_10_coord_NLOS = results_NLOS_coord[results_NLOS_coord['top_k']<=10]['std']

    ax[0].plot(top_10, accuracy_top_10_coord_all, label='Mean ALL', color='green', marker='o', markersize=marker_size)
    ax[0].fill_between (top_10, accuracy_top_10_coord_all - std_top_10_coord_all,
                        accuracy_top_10_coord_all + std_top_10_coord_all, alpha=0.2, color='green')
    ax[0].plot(top_10, accuracy_top_10_coord_LOS, label='Mean LOS', color='darkblue', marker='o', markersize=marker_size)
    ax[0].fill_between (top_10, accuracy_top_10_coord_LOS - std_top_10_coord_LOS,
                        accuracy_top_10_coord_LOS + std_top_10_coord_LOS, alpha=0.2, color='darkblue')
    ax[0].plot(top_10, accuracy_top_10_coord_NLOS, label='Mean NLOS', color='C1', marker='o', markersize=marker_size)
    ax[0].fill_between (top_10, accuracy_top_10_coord_NLOS - std_top_10_coord_NLOS,
                        accuracy_top_10_coord_NLOS + std_top_10_coord_NLOS, alpha=0.2, color='C1')


    ax[0].set_xlabel('Coord \n Top-K', fontsize=size_of_font)
    ax[0].grid()
    ax[0].set_xticks(top_10)


    accuracy_top_10_lidar_all = results_ALL_lidar[results_ALL_lidar['top_k']<=10]['mean']
    std_top_10_lidar_all = results_ALL_lidar[results_ALL_lidar['top_k']<=10]['std']

    accuracy_top_10_lidar_LOS = results_LOS_lidar[results_LOS_lidar['top_k']<=10]['mean']
    std_top_10_lidar_LOS = results_LOS_lidar[results_LOS_lidar['top_k']<=10]['std']

    accuracy_top_10_lidar_NLOS = results_NLOS_lidar[results_NLOS_lidar['top_k']<=10]['mean']
    std_top_10_lidar_NLOS = results_NLOS_lidar[results_NLOS_lidar['top_k']<=10]['std']

    ax[1].plot(top_10, accuracy_top_10_lidar_all, label='Mean ALL', color='green',marker='o', markersize=marker_size)
    ax[1].fill_between (top_10, accuracy_top_10_lidar_all - std_top_10_lidar_all,
                        accuracy_top_10_lidar_all + std_top_10_lidar_all, alpha=0.2, color='green')

    ax[1].plot(top_10, accuracy_top_10_lidar_LOS, label='Mean LOS', color='darkblue',marker='o', markersize=marker_size)
    ax[1].fill_between (top_10, accuracy_top_10_lidar_LOS - std_top_10_lidar_LOS,
                        accuracy_top_10_lidar_LOS + std_top_10_lidar_LOS, alpha=0.2, color='darkblue')

    ax[1].plot(top_10, accuracy_top_10_lidar_NLOS, label='Mean NLOS', color='C1',marker='o', markersize=marker_size)
    ax[1].fill_between (top_10, accuracy_top_10_lidar_NLOS - std_top_10_lidar_NLOS,
                        accuracy_top_10_lidar_NLOS + std_top_10_lidar_NLOS, alpha=0.2, color='C1')
    ax[1].set_xlabel ('LiDAR \n Top-K', fontsize=size_of_font)
    ax[1].grid()
    ax[1].set_xticks(top_10)

    accuracy_top_10_lidar_coord_all = results_ALL_lidar_coord[results_ALL_lidar_coord['top_k']<=10]['mean']
    std_top_10_lidar_coord_all = results_ALL_lidar_coord[results_ALL_lidar_coord['top_k']<=10]['std']
    accuracy_top_10_lidar_coord_LOS = results_LOS_lidar_coord[results_LOS_lidar_coord['top_k']<=10]['mean']
    std_top_10_lidar_coord_LOS = results_LOS_lidar_coord[results_LOS_lidar_coord['top_k']<=10]['std']
    accuracy_top_10_lidar_coord_NLOS = results_NLOS_lidar_coord[results_NLOS_lidar_coord['top_k']<=10]['mean']
    std_top_10_lidar_coord_NLOS = results_NLOS_lidar_coord[results_NLOS_lidar_coord['top_k']<=10]['std']





    ax[2].plot(top_10, accuracy_top_10_lidar_all,
               label='Mean ALL', color='green',marker='o', markersize=marker_size)
    ax[2].fill_between (top_10, accuracy_top_10_lidar_all - std_top_10_lidar_coord_all,
                        accuracy_top_10_lidar_all + std_top_10_lidar_coord_all, alpha=0.2, color='green')
    ax[2].plot(top_10, accuracy_top_10_lidar_LOS,
                label='Mean LOS', color='darkblue', marker='o', markersize=marker_size)
    ax[2].fill_between (top_10, accuracy_top_10_lidar_LOS - std_top_10_lidar_coord_LOS,
                        accuracy_top_10_lidar_LOS + std_top_10_lidar_coord_LOS, alpha=0.2, color='darkblue')
    ax[2].plot(top_10, accuracy_top_10_lidar_NLOS,
                label='Mean NLOS', color='C1', marker='o', markersize=marker_size)
    ax[2].fill_between (top_10, accuracy_top_10_lidar_NLOS - std_top_10_lidar_coord_NLOS,
                        accuracy_top_10_lidar_NLOS + std_top_10_lidar_coord_NLOS, alpha=0.2, color='C1')
    ax[2].set_xlabel ('LiDAR + Coord \n Top-K', fontsize=size_of_font)
    ax[2].grid ()

    ax[1].legend()
    ax[0].set_ylabel('Accuracy', fontsize=size_of_font)

    fig.suptitle(f'Selecao de feixe intra-dataset {dataset}', fontsize=14)

    path_to_save = '../results/score/Wisard/beam_selection_with_s008_or_s009/' + dataset + '/'
    file_name = 'Top-k_performance_accuracy_all_LOS_NLOS.png'
    plt.savefig(path_to_save + file_name, dpi=300, bbox_inches='tight')


dataset = 's008'
#calculate_beam_selection_top_k_mean_std(input_type='coord', connection_type='LOS', dataset=dataset)

plot_results_TOP_K(dataset=dataset)
dataset = 's009'
plot_results_TOP_K(dataset=dataset)
#calculate_beam_selection_mean_std(input_type='lidar', connection_type='LOS', dataset=dataset)
#calculate_beam_selection_mean_std(input_type='lidar_coord', connection_type='LOS', dataset=dataset)


