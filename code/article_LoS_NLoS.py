import pandas as pd

import beam_selection_LOS_NLOS as bm_wisard
import read_data as read_data

def read_data_to_predict_beam(connection_type, scale_to_coord):
    s008_data_LOS, s008_data_NLOS, s008_valid_data = read_data.read_data_s008(scale_to_coord)
    s009_data_LOS, s009_data_NLOS, s009_valid_data = read_data.read_data_s009(scale_to_coord)

    if connection_type == 'LOS':
        return s008_data_LOS, s009_data_LOS
    elif connection_type == 'NLOS':
        return s008_data_NLOS, s009_data_NLOS
    elif connection_type == 'ALL':
        return s008_valid_data, s009_valid_data

def define_type_of_input(input_type,s008_data, s009_data):

    if input_type == 'lidar':
        s008_input_data = s008_data['lidar'].tolist()
        s009_input_data = s009_data['lidar'].tolist()
    elif input_type == 'coord':
        s008_input_data = s008_data['enconding_coord'].tolist()
        s009_input_data = s009_data['enconding_coord'].tolist()
    elif input_type == 'lidar_coord':
        s008_input_data = s008_data['lidar_coord'].tolist()
        s009_input_data = s009_data['lidar_coord'].tolist()

    s009_label_data = [str (y) for y in s009_data ['index_beams']]
    s008_label_data = [str (y) for y in s008_data ['index_beams']]

    return s008_input_data, s008_label_data, s009_input_data, s009_label_data

def select_beam(connection_type, input_type, dataset_for_train, scale_to_coord, if_top_k):


    s008_data, s009_data = read_data_to_predict_beam (connection_type, scale_to_coord)
    s008_input_data, s008_label_data, s009_input_data, s009_label_data = define_type_of_input (input_type,
                                                                                               s008_data,
                                                                                               s009_data)
    if dataset_for_train == 's008':
        data_train = s008_input_data
        data_test = s009_input_data
        label_train = s008_label_data
        label_test = s009_label_data
    elif dataset_for_train == 's009':
        data_train = s009_input_data
        data_test = s008_input_data
        label_train = s009_label_data
        label_test = s008_label_data

    if if_top_k:
        top_k, score = bm_wisard.beam_selection_top_k_wisard (x_train=data_train,
                                                    x_test=data_test,
                                                    y_train=label_train,
                                                    y_test=label_test,
                                                    data_input='', data_set='',
                                                    address_of_size=48,
                                                    name_of_conf_input='')
        results = pd.DataFrame ({'score': score,
                                 'top_k': top_k})

        return results


    else:
        average_accuracy, standar_desviation_accuracy, address_size = bm_wisard.select_best_beam (input_train=data_train,
                                                                                       input_validation=data_test,
                                                                                       label_train=label_train,
                                                                                       label_validation=label_test,
                                                                                       figure_name='cc',
                                                                                       antenna_config='cc',
                                                                                       type_of_input=input_type,
                                                                                       titulo_figura='cc',
                                                                                       user='cc',
                                                                                       enableDebug=False, )

        results = pd.DataFrame ({'Average_accuracy': average_accuracy,
                                 'Standar_desviation_accuracy': standar_desviation_accuracy,
                                 'Address_size': address_size})

       # max_acuraccy = results ['Average_accuracy'].max ()
       # best_config = results [results ['Average_accuracy'] == max_acuraccy]

        return results

def generalization_wisard_test():
    input_type = ['lidar']
    dataset_for_train = 's008'
    scale_to_coord = 16
    if_top_k = True

    for input_type in input_type:
        results = select_beam ('ALL',
                               input_type,
                               dataset_for_train,
                               scale_to_coord,
                               if_top_k)

        print ('Top k results: ', results)
        path_to_save = ('../results/results_article_LoS_NLoS/generalization_test/dataset_train_' + dataset_for_train + '/'
                        + input_type + '/')
        file_name = input_type + '_train_with_' + dataset_for_train + '.csv'
        results.to_csv (path_to_save + file_name, index=False)

def LoS_NLoS_beam_selection():
    connection_type = ['ALL']#['LOS','NLOS', 'ALL']
    input_type = ['coord', 'lidar', 'lidar_coord']
    dataset_for_train = 's008'
    if_top_k = False



    tunning_parameters = False
    if tunning_parameters:
        scale_to_preprocess_coord = [8, 16, 32]
        best_config_vector = []
        for scale_to_coord in scale_to_preprocess_coord:
            best_config = select_beam(connection_type, input_type, dataset_for_train, scale_to_coord)
            print(f'Scale to preprocess coord: {scale_to_coord}, Best config: {best_config}')
            best_config_vector.append(best_config)
    else:
        scale_to_coord = [8, 16, 32, 64]


        for i in range(len(connection_type)):
            for j in range(len(input_type)):
                for k in range(len(scale_to_coord)):
                    print (f'Connection type: {connection_type[i]}, '
                           f'Input type: {input_type[j]}, '
                           f'Scale to coord: {scale_to_coord[k]}, '
                           f'Dataset for train: {dataset_for_train}')

                    results = select_beam(connection_type[i],
                                          input_type[j],
                                          dataset_for_train,
                                          scale_to_coord[k],
                                          if_top_k)

                    path_to_save = '../results/results_article_LoS_NLoS/beam_selection/train_with_'+dataset_for_train+'/'+connection_type[i]+'/'+input_type[j]+'/'
                    file_name = (input_type [j] + '_train_with_' + dataset_for_train +
                                 '_connection_' + connection_type [i] +
                                 '_res_' + str (scale_to_coord[k]) + '.csv')
                    print (path_to_save + file_name)

                    results.to_csv(path_to_save+file_name, index=False)



                    a=0

def read_results_LOS_NLOS():
    path = '../results/score/Wisard/coord/LOS'
    connection_types = ['LOS', 'NLOS', 'ALL']
    input_types = ['lidar', 'coord', 'lidar_coord']


def read_LoS_NLoS_results(dataset):
    import os

    path_to_read = '../results/results_article_LoS_NLoS/beam_selection/train_with_'+dataset+'/'
    connection_types = ['LOS', 'NLOS', 'ALL']
    input_types = ['coord','lidar', 'lidar_coord']

    all_results = []
    for connection_type in connection_types:
        for input_type in input_types:
            file_name = input_type+'_train_with_'+dataset+'_'+connection_type+'.csv'
            full_path = os.path.join(path_to_read,connection_type, file_name)
            if os.path.exists(full_path):
                result = pd.read_csv(full_path)
                result.insert(0, 'Connection_type', connection_type)
                result.insert(1, 'Input_type', input_type)
                all_results.append(result)
            else:
                print(f'File not found: {full_path}')

    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        return combined_results
        a=0
    else:
        print('No results found.')
        return None


def plot_results_LOS_NLOS():
    import os


    connection_types = ['LOS', 'NLOS', 'ALL']
    input_types = ['coord', 'lidar', 'lidar_coord']

    all_results = []
    for connection_type in connection_types:
        for input_type in input_types:
            path_to_read = '../results/score/Wisard/' + input_type + '/' + connection_type + '/'
            if input_type == 'coord':
                file_name = 'accuracy_' + input_type + '_res_8_' + connection_type + '.csv'
            if input_type == 'lidar':
                file_name = 'accuracy_'+input_type+'_'+connection_type+'_thr_01.csv'
            if input_type == 'lidar_coord':
                file_name='accuracy_'+input_type+'_res_8_'+connection_type+'_thr_01.csv'
            full_path = os.path.join (path_to_read, file_name)
            print(full_path)
            if os.path.exists (full_path):
                result = pd.read_csv (full_path)
                result.insert (0, 'Connection_type', connection_type)
                result.insert (1, 'Input_type', input_type)
                all_results.append (result)
            else:
                print (f'File not found: {full_path}')

    if all_results:
        combined_results = pd.concat (all_results, ignore_index=True)
    else:
        print ('No results found.')
        return None

    coord_results = combined_results[combined_results['Input_type']=='coord']
    coord_results_LOS = coord_results[coord_results['Connection_type']=='LOS']
    coord_accuracy_LOS = coord_results_LOS[coord_results_LOS['addres_size']==64]['accuracy'].values
    coord_results_NLOS = coord_results[coord_results['Connection_type']=='NLOS']
    coord_accuracy_NLOS = coord_results_NLOS[coord_results_NLOS['addres_size']==64]['accuracy'].values
    coord_results_ALL = coord_results[coord_results['Connection_type']=='ALL']
    coord_accuracy_ALL = coord_results_ALL[coord_results_ALL['addres_size']==64]['accuracy'].values

    lidar_results = combined_results[combined_results['Input_type']=='lidar']
    lidar_results_LOS = lidar_results[lidar_results['Connection_type']=='LOS']
    lidar_accuracy = lidar_results_LOS[lidar_results_LOS['addres_size']==64]['accuracy'].values
    lidar_results_NLOS = lidar_results[lidar_results['Connection_type']=='NLOS']
    lidar_accuracy_NLOS = lidar_results_NLOS[lidar_results_NLOS['addres_size']==64]['accuracy'].values
    lidar_results_ALL = lidar_results[lidar_results['Connection_type']=='ALL']
    lidar_accuracy_ALL = lidar_results_ALL[lidar_results_ALL['addres_size']==64]['accuracy'].values

    lidar_coord_results = combined_results[combined_results['Input_type']=='lidar_coord']
    lidar_coord_results_LOS = lidar_coord_results[lidar_coord_results['Connection_type']=='LOS']
    lidar_coord_accuracy = lidar_coord_results_LOS[lidar_coord_results_LOS['addres_size']==64]['accuracy'].values
    lidar_coord_results_NLOS = lidar_coord_results[lidar_coord_results['Connection_type']=='NLOS']
    lidar_coord_accuracy_NLOS = lidar_coord_results_NLOS[lidar_coord_results_NLOS['addres_size']==64]['accuracy'].values
    lidar_coord_results_ALL = lidar_coord_results[lidar_coord_results['Connection_type']=='ALL']
    lidar_coord_accuracy_ALL = lidar_coord_results_ALL[lidar_coord_results_ALL['addres_size']==64]['accuracy'].values


    df_coord_data = pd.DataFrame ({'connection_type': ['LOS', 'NLOS', 'ALL'],
                  'Accuracy': [coord_accuracy_LOS[0], coord_accuracy_NLOS[0], coord_accuracy_ALL[0]]})
    df_lidar_data = pd.DataFrame ({'connection_type': ['LOS', 'NLOS', 'ALL'],
                    'Accuracy': [lidar_accuracy[0], lidar_accuracy_NLOS[0], lidar_accuracy_ALL[0]]})
    df_lidar_coord = pd.DataFrame ({'connection_type': ['LOS', 'NLOS', 'ALL'],
                    'Accuracy': [lidar_coord_accuracy[0], lidar_coord_accuracy_NLOS[0], lidar_coord_accuracy_ALL[0]]})


    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure ()
    plt.rcParams ['font.family'] = 'Times New Roman'
    plt.rcParams ['font.size'] = 12
    plt.tick_params (axis='x', which='both', bottom=False, top=False, labelbottom=True)

    plt.text (0, 2.1, 'GPS data', color='black', fontsize=12)

    bars_colors = ['steelblue',  'sandybrown', 'green']

    y_pos_coord = [1, 1.4, 1.8]
    y_pos_lidar = [3, 3.4, 3.8]
    y_pos_lidar_coord = [5, 5.4, 5.8]

    h = plt.barh (y_pos_coord, df_coord_data ['Accuracy'], height=0.3, color=bars_colors, alpha=0.6)
    for i in range (len (y_pos_coord)):
        plt.text (df_coord_data ['Accuracy'][i] - 0.1, y_pos_coord [i] - 0.03,
                  str (np.round (df_coord_data ['Accuracy'] [i] * 100, 2)) + '%',
                  ha='left', va='center', color='white', weight='bold')

    plt.text (0, 4.1, 'LiDAR data', color='black', fontsize=12)
    for i in range (len (y_pos_lidar)):
        plt.text (df_lidar_data ['Accuracy'] [i] - 0.1, y_pos_lidar [i] - 0.03,
                  str (np.round (df_lidar_data ['Accuracy'] [i] * 100, 2)) + '%',
                  ha='left', va='center', color='white', weight='bold')
    plt.barh (y_pos_lidar, df_lidar_data ['Accuracy'], height=0.3, color=bars_colors, alpha=0.6)  # 'skyblue')

    plt.text (0, 6.1, 'LiDAR + GPS data', color='black', fontsize=12)
    plt.barh (y_pos_lidar_coord, df_lidar_coord['Accuracy'], height=0.3, color=bars_colors, alpha=0.6)  # 'skyblue')
    for i in range (len (y_pos_lidar_coord)):
        plt.text (df_lidar_coord['Accuracy'][i] - 0.1, y_pos_lidar_coord [i] - 0.03,
                  str (np.round (df_lidar_coord['Accuracy'][i] * 100, 2)) + '%',
                  ha='left', va='center', color='white', weight='bold')
    plt.legend (handles=h, labels=['LOS', 'NLOS', 'ALL'], loc='lower right', ncol=3, frameon=False,
                bbox_to_anchor=(1.05, 1))

    plt.xlabel ('Accuracy')
    plt.axis ('off')

    a=0




def plot_LoS_NLoS_results():
    import matplotlib.pyplot as plt
    import numpy as np

    dataset = 's009'
    results = read_LoS_NLoS_results (dataset)
    coord = results[results['Input_type']=='coord']['Average_accuracy']
    lidar = results[results['Input_type']=='lidar']['Average_accuracy']
    lidar_coord = results[results['Input_type']=='lidar_coord']['Average_accuracy']

    gps = results[results['Input_type']=='coord']

    data = np.array([coord.values, lidar.values, lidar_coord.values])

    coord_data = {'connection_type': ['LOS', 'NLOS', 'ALL'],
                  'Accuracy': coord.values}

    data = {'Category': ['A', 'B', 'C', 'D'],
            'Value': [20, 35, 15, 40]}
    df = pd.DataFrame (coord_data)
    df_lidar = pd.DataFrame ({'connection_type': ['LOS', 'NLOS', 'ALL'],
                              'Accuracy': lidar.values})
    bars_colors = ['green', 'steelblue', 'sandybrown']

    y_pos_coord = [1, 1.4, 1.8]
    y_pos_lidar = [3, 3.4, 3.8]
    y_pos_lidar_coord = [5, 5.4, 5.8]



    plt.figure()
    plt.rcParams ['font.family'] = 'Times New Roman'
    plt.rcParams ['font.size'] = 12
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

    plt.text (0, 2.1, 'GPS data', color='black', fontsize=12)

    h = plt.barh (y_pos_coord, df ['Accuracy'], height=0.3, color=bars_colors, alpha=0.6)
    for i in range(len(y_pos_coord)):
        plt.text (df['Accuracy'][i]-0.1, y_pos_coord[i]-0.03, str(np.round(df['Accuracy'][i]*100,2))+'%',
                  ha='left', va='center', color='white', weight='bold')

    plt.text (0, 4.1, 'LiDAR data', color='black', fontsize=12)
    for i in range(len(y_pos_lidar)):
        plt.text (df_lidar['Accuracy'][i]-0.1, y_pos_lidar[i]-0.03, str(np.round(df_lidar['Accuracy'][i]*100,2))+'%',
                  ha='left', va='center', color='white', weight='bold')
    plt.barh (y_pos_lidar, df_lidar['Accuracy'], height=0.3, color=bars_colors, alpha=0.6)#'skyblue')

    plt.text (0, 6.1, 'LiDAR + GPS data', color='black', fontsize=12)
    plt.barh (y_pos_lidar_coord, lidar_coord.values, height=0.3, color=bars_colors, alpha=0.6)#'skyblue')
    for i in range(len(y_pos_lidar_coord)):
        plt.text (lidar_coord.values[i]-0.1, y_pos_lidar_coord[i]-0.03, str(np.round(lidar_coord.values[i]*100,2))+'%',
                  ha='left', va='center', color='white', weight='bold')
    plt.legend(handles=h, labels=['LOS', 'NLOS', 'ALL'], loc='lower right', ncol=3, frameon=False, bbox_to_anchor=(1.05, 1))

    plt.xlabel('Accuracy')
    plt.axis('off')
    path_to_save = '../results/results_article_LoS_NLoS/beam_selection/train_with_'+dataset+'/'
    file_name = 'beam_selection_results_LoS_NLoS_dataset_train_'+dataset+'.png'
    plt.savefig(path_to_save+file_name, bbox_inches='tight', dpi=300)


    #df.plot.barh (x='Accuracy', y='connection_type', title='Horizontal Bar Chart Example', color='skyblue')
    a=0


def read_results_top_k(dataset, input_type, connection_type):
    path_csv = '../results/score/Wisard/beam_selection_with_s008_or_s009/' + dataset + '/'+input_type+'/'
    file_name = 'accuracy_top_k'+input_type+'_addresSize_44_'+connection_type+'.csv'
    df = pd.read_csv(path_csv + file_name)


    return df

#generalization_wisard_test()
#LoS_NLoS_beam_selection()
plot_LoS_NLoS_results()
#plot_results_LOS_NLOS()