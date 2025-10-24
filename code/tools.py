import numpy as np

def calculate_mean_score(data):
    #all_score = data ['Score'].tolist ()
    average_score = []
    for i in range(len(data)):
        i = i + 1
        average_score.append(np.mean(data[0:i]))
    return average_score

def extract_test_data_from_s009_sliding_window(episode, label_input_type, s009_data):
    label_test = s009_data [s009_data ['Episode'] == episode] ['index_beams'].tolist ()

    input_test = []

    if label_input_type == 'coord':
        #input_test = s009_data [s009_data ['Episode'] == episode] ['encoding_coord'].tolist ()
        input_test = s009_data [s009_data ['Episode'] == episode] ['coord'].tolist ()
    elif label_input_type == 'lidar':
        input_test = s009_data [s009_data ['Episode'] == episode] ['lidar'].tolist ()
    elif label_input_type == 'lidar_coord':
        input_test_coord = s009_data [s009_data ['Episode'] == episode] ['coord'].tolist ()
        input_test_lidar = s009_data [s009_data ['Episode'] == episode] ['lidar'].tolist ()
        input_test = [input_test_coord, input_test_lidar]

    else:
        print ('error: deve especificar o tipo de entrada')
    a=0

    return input_test, label_test
def extract_training_data_from_s008_sliding_window(s008_data, start_index, input_type):

    initial_data_for_trainning = s008_data [s008_data ['Episode'] >= start_index]
    label_train = initial_data_for_trainning ['index_beams'].tolist ()
    input_train = []

    if input_type == 'coord':
        #input_train = initial_data_for_trainning ['encoding_coord'].tolist ()
        input_train = initial_data_for_trainning ['coord'].tolist ()

    elif input_type == 'lidar':
        input_train = initial_data_for_trainning ['lidar'].tolist ()
    elif input_type == 'lidar_coord':
        input_coord_train = initial_data_for_trainning ['coord'].tolist ()
        input_lidar_train = initial_data_for_trainning ['lidar'].tolist ()
        input_train = [input_lidar_train, input_coord_train,]
    else:
        print('error: deve especificar o tipo de entrada')

    return input_train, label_train
def extract_training_data_from_s009_sliding_window(s009_data, start_index, end_index, input_type):


    data_for_trainnig = s009_data.loc[(s009_data['Episode'] >= start_index) & (s009_data['Episode'] < end_index)]

    label_train = data_for_trainnig['index_beams'].tolist()

    input_train = []
    if input_type == 'coord':
        #input_train = data_for_trainnig['encoding_coord'].tolist()
        input_train = data_for_trainnig ['coord'].tolist ()
    elif input_type == 'lidar':
        input_train = data_for_trainnig['lidar'].tolist()
    elif input_type == 'lidar_coord':
        input_coord_train = data_for_trainnig ['coord'].tolist ()
        input_lidar_train = data_for_trainnig ['lidar'].tolist ()
        input_train = [input_lidar_train, input_coord_train, ]

    return input_train, label_train