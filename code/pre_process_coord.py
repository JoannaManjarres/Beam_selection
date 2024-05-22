import csv
import numpy as np

def read_valid_coordinates_s008():

    #filename = '/Users/Joanna/git/Analise_de_dados/data/coordinates/CoordVehiclesRxPerScene_s008.csv'
    filename ='../data/coord/CoordVehiclesRxPerScene_s008.csv'
    limit_ep_train = 1564

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        number_of_rows = len(list(reader))

    all_info_coord_val = np.zeros([11194, 5], dtype=object)

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        cont = 0
        for row in reader:
            if row['Val'] == 'V':
                all_info_coord_val[cont] = int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS']
                cont += 1

    # all_info_coord = np.array(all_info_coord)

    coord_train = all_info_coord_val[(all_info_coord_val[:, 0] < limit_ep_train + 1)]
    coord_validation = all_info_coord_val[(all_info_coord_val[:, 0] > limit_ep_train)]

    return all_info_coord_val, coord_train, coord_validation
def read_valid_coordinates_s009():
        path = '../data/coord/'
        filename = 'CoordVehiclesRxPerScene_s009.csv'

        with open (path + filename) as csvfile:
            reader = csv.DictReader (csvfile)
            number_of_rows = len (list (reader))

        all_info_coord = np.zeros ((9638, 4))
        info_coord = []

        with open (path + filename) as csvfile:
            reader = csv.DictReader (csvfile)
            cont = 0
            for row in reader:
                if row ['Val'] == 'V':
                    all_info_coord [cont] = int(row ['EpisodeID']), float(row['x']), float(row['y']), float (row['z'])
                    info_coord.append([int(row['EpisodeID']), row['LOS'], float(row ['x']), float(row ['y']), float(row['z'])])
                    cont += 1

        coord = np.asarray(info_coord)

        return (all_info_coord)

def  Thermomether_coord_x_y_unbalanced_for_s008(escala):
    #int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates_s008()

    episodios = all_info_coord_val[:,0]

    all_x_coord_str = all_info_coord_val[:,1]
    all_x_coord = [int(x) for x in all_x_coord_str]
    all_y_coord_str = all_info_coord_val[:, 2]
    all_y_coord = [int(y) for y in all_y_coord_str]


    min_x_coord = np.min(all_x_coord)
    min_y_coord = np.min(all_y_coord)
    max_x_coord = np.max(all_x_coord)
    max_y_coord = np.max(all_y_coord)


    diff_all_x_coord = np.array((all_x_coord - min_x_coord))
    diff_all_y_coord = np.array(all_y_coord) - min_y_coord

    escala = escala
    size_of_data_x = (max_x_coord - min_x_coord) * escala
    size_of_data_y = (max_y_coord - min_y_coord) * escala
    enconding_x = np.array([len(all_info_coord_val), size_of_data_x], dtype=int)
    enconding_y = np.array([len(all_info_coord_val), size_of_data_y+escala], dtype=int) #+escala], dtype=int)

    encoding_x_vector = np.zeros(enconding_x, dtype=int)
    encoding_y_vector = np.zeros(enconding_y, dtype=int)

    n_x = 1 * escala

    sample = 0
    for i in diff_all_x_coord:
        for j in range(i * n_x):
            encoding_x_vector[sample, j] = 1
        sample = sample + 1

    sample = 0
    for i in diff_all_y_coord:
        for j in range(i * n_x+escala):#+escala):
            encoding_y_vector[sample, j] = 1
        sample = sample + 1


    encondig_coord = np.concatenate((encoding_x_vector, encoding_y_vector), axis=1)
    encoding_coord_and_episode = np.column_stack([episodios, encondig_coord])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] < limit_ep_train + 1)]
    encondign_coord_validation = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] > limit_ep_train)]

    size_of_input = encondign_coord_train.shape
    encondign_coord_train = encondign_coord_train[:,1:size_of_input[1]]
    encondign_coord_validation = encondign_coord_validation[:,1:size_of_input[1]]



    #print ("Tamanho do vetor de entrada Train: ", encondign_coord_train.shape)
    #print ("Tamanho do vetor de entrada Validation: ", encondign_coord_validation.shape)

    return encondign_coord_train, encondign_coord_validation, encondig_coord
def  Thermomether_coord_x_y_unbalanced_for_s009(escala):
    #int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
    all_info_coord_val = read_valid_coordinates_s009()

    episodios = all_info_coord_val[:,0]

    all_x_coord_str = all_info_coord_val[:,1]
    all_x_coord = [int(x) for x in all_x_coord_str]
    all_y_coord_str = all_info_coord_val[:, 2]
    all_y_coord = [int(y) for y in all_y_coord_str]

    min_x_coord = np.min(all_x_coord)
    min_y_coord = np.min(all_y_coord)

    max_x_coord = np.max(all_x_coord)
    max_y_coord = np.max(all_y_coord)

    diff_all_x_coord = np.array((all_x_coord - min_x_coord))
    diff_all_y_coord = np.array((all_y_coord - min_y_coord))

    escala = escala
    size_of_data_x = (max_x_coord-min_x_coord) * escala
    size_of_data_y = (max_y_coord-min_y_coord) * escala
    enconding_x = np.array([len(all_info_coord_val), size_of_data_x], dtype=int)
    enconding_y = np.array([len(all_info_coord_val), size_of_data_y], dtype=int)

    encoding_x_vector = np.zeros(enconding_x, dtype=int)
    encoding_y_vector = np.zeros(enconding_y, dtype=int)

    n_x = 1 * escala

    sample = 0
    for i in diff_all_x_coord:
        for j in range(i * n_x):
            encoding_x_vector[sample, j] = 1
        sample = sample + 1

    sample = 0
    for i in diff_all_y_coord:
        for j in range(i * n_x):
            encoding_y_vector[sample, j] = 1
        sample = sample + 1

    encondig_coord = np.concatenate((encoding_x_vector, encoding_y_vector), axis=1)

    return encondig_coord

def  Thermomether_from_BS_coord_x_y_unbalanced_for_s008(escala):
    #int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates_s008()

    episodios = all_info_coord_val[:,0]

    all_x_coord_float = all_info_coord_val[:,1]
    all_x_coord = [int(x) for x in all_x_coord_float]
    all_y_coord_float = all_info_coord_val[:, 2]
    all_y_coord = [int(y) for y in all_y_coord_float]
    all_z_coord_float = all_info_coord_val [:, 3]
    all_z_coord = [int(z) for z in all_z_coord_float]


    BS_position_x = 746
    BS_position_y = 560
    BS_position_z = 4

    distance_from_BS_x = [np.abs(x-BS_position_x) for x in all_x_coord]
    distance_from_BS_y = [np.abs(y-BS_position_y) for y in all_y_coord]
    distance_from_BS_z = [np.abs(z-BS_position_z) for z in all_z_coord]

    max_dist_x = np.max(distance_from_BS_x)
    max_dist_y = np.max(distance_from_BS_y)
    max_dist_z = np.max(distance_from_BS_z)

    escala = escala
    size_of_data_x = (max_dist_x) * escala
    size_of_data_y = (max_dist_y) * escala
    size_of_data_z = (max_dist_z) * escala

    enconding_x = np.array([len(all_info_coord_val), size_of_data_x], dtype=int)
    enconding_y = np.array([len(all_info_coord_val), size_of_data_y], dtype=int)
    enconding_z = np.array([len(all_info_coord_val), size_of_data_z], dtype=int)

    encoding_x_vector = np.zeros(enconding_x, dtype=int)
    encoding_y_vector = np.zeros(enconding_y, dtype=int)
    encoding_z_vector = np.zeros(enconding_z, dtype=int)

    n_x = 1 * escala

    sample = 0
    for i in distance_from_BS_x:
        for j in range(i * n_x):
            encoding_x_vector[sample, j] = 1
        sample = sample + 1

    sample = 0
    for i in distance_from_BS_y:
        for j in range(i * n_x):
            encoding_y_vector[sample, j] = 1
        sample = sample + 1

    sample = 0
    for i in distance_from_BS_z:
        for j in range (i * n_x):
            encoding_z_vector [sample, j] = 1
        sample = sample + 1


    encondig_coord = np.concatenate((encoding_x_vector, encoding_y_vector), axis=1)
    encoding_coord_and_episode = np.column_stack([episodios, encondig_coord])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] < limit_ep_train + 1)]
    encondign_coord_validation = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] > limit_ep_train)]

    size_of_input = encondign_coord_train.shape
    encondign_coord_train = encondign_coord_train[:,1:size_of_input[1]]
    encondign_coord_validation = encondign_coord_validation[:,1:size_of_input[1]]


    print("\t\t Pre processamento das coordenadas")
    print("-------------------------------------------")
    print("\tEscala \t\tTrain \t\tValidation")
    print("\t", escala, "\t\t", encondign_coord_train.shape, "\t", encondign_coord_validation.shape)
    #print ("Tamanho do vetor de entrada Train: ", encondign_coord_train.shape)
    #print ("Tamanho do vetor de entrada Validation: ", encondign_coord_validation.shape)

    return encondign_coord_train, encondign_coord_validation, encondig_coord

def  Thermomether_from_BS_coord_x_y_unbalanced_all_data_concatenate_for_s008(escala):
    #int(row['EpisodeID']), float(row['x']), float(row['y']), float(row['z']), row['LOS'], row['Val']
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates_s008()

    episodios = all_info_coord_val[:,0]

    all_x_coord_float = all_info_coord_val[:,1]
    all_x_coord = [int(x) for x in all_x_coord_float]
    all_y_coord_float = all_info_coord_val[:, 2]
    all_y_coord = [int(y) for y in all_y_coord_float]
    all_z_coord_float = all_info_coord_val [:, 3]
    all_z_coord = [int(z) for z in all_z_coord_float]


    BS_position_x = 746
    BS_position_y = 560
    BS_position_z = 4

    distance_from_BS_x = [np.abs(x-BS_position_x) for x in all_x_coord]
    distance_from_BS_y = [np.abs(y-BS_position_y) for y in all_y_coord]
    distance_from_BS_z = [np.abs(z-BS_position_z) for z in all_z_coord]

    max_dist_x = np.max(distance_from_BS_x)
    max_dist_y = np.max(distance_from_BS_y)
    max_dist_z = np.max(distance_from_BS_z)

    escala = escala
    size_of_data_x = (max_dist_x) * escala
    size_of_data_y = (max_dist_y) * escala
    size_of_data_z = (max_dist_z) * escala
    size_of_all_data = size_of_data_x + size_of_data_y + size_of_data_z

    enconding_x = np.zeros([len(all_info_coord_val), size_of_data_x], dtype=int)
    enconding_y = np.zeros([len(all_info_coord_val), size_of_data_y], dtype=int)
    enconding_z = np.zeros([len(all_info_coord_val), size_of_data_z], dtype=int)

    encoding_all = np.zeros([len(all_info_coord_val), size_of_all_data], dtype=int)



    n_x = 1 * escala

    sample = 0
    for i in range(len(all_info_coord_val)):
        x = distance_from_BS_x[i]
        y = distance_from_BS_y[i]
        z = distance_from_BS_z[i]
        distances = x+y+z
        for j in range(distances * n_x):
            encoding_all[i, j] = 1
        sample = sample + 1
    '''
    sample = 0
    for i in distance_from_BS_x:
        for j in range(i * n_x):
            encoding_x_vector[sample, j] = 1
        sample = sample + 1

    sample = 0
    for i in distance_from_BS_y:
        for j in range(i * n_x):
            encoding_y_vector[sample, j] = 1
        sample = sample + 1

    sample = 0
    for i in distance_from_BS_z:
        for j in range (i * n_x):
            encoding_z_vector [sample, j] = 1
        sample = sample + 1


    encondig_coord = np.concatenate((encoding_x_vector, encoding_y_vector), axis=1)
    '''
    encoding_coord_and_episode = np.column_stack([episodios, encoding_all])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] < limit_ep_train + 1)]
    encondign_coord_validation = encoding_coord_and_episode[(encoding_coord_and_episode[:, 0] > limit_ep_train)]

    size_of_input = encondign_coord_train.shape
    encondign_coord_train = encondign_coord_train[:,1:size_of_input[1]]
    encondign_coord_validation = encondign_coord_validation[:,1:size_of_input[1]]


    print("\t\t Pre processamento das coordenadas")
    print("-------------------------------------------")
    print("\tEscala \t\tTrain \t\tValidation")
    print("\t", escala, "\t\t", encondign_coord_train.shape, "\t", encondign_coord_validation.shape)
    #print ("Tamanho do vetor de entrada Train: ", encondign_coord_train.shape)
    #print ("Tamanho do vetor de entrada Validation: ", encondign_coord_validation.shape)

    return encondign_coord_train, encondign_coord_validation, encoding_all


def sum_digits_of_each_coord_x_y_for_s008(escala):
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates_s008()

    episodios = all_info_coord_val [:, 0]

    all_x_coord_float = all_info_coord_val [:, 1]
    all_x_coord_int = [int(x) for x in all_x_coord_float]
    all_x_coord_str = [str (x) for x in all_x_coord_int]


    all_y_coord_float = all_info_coord_val [:, 2]
    all_y_coord_int = [int(y) for y in all_y_coord_float]
    all_y_coord_str = [str(y) for y in all_y_coord_int]


    preprocess_coord_x = []
    preprocess_coord_y = []


    for i in range(len(all_x_coord_str)):
        num_x = [int(x) for x in all_x_coord_str[i]]
        sum_x = sum(num_x)
        preprocess_coord_x.append(sum_x)

    for i in range(len(all_x_coord_str)):
        num_y = [int(x) for x in all_y_coord_str[i]]
        sum_y = sum(num_y)
        preprocess_coord_y.append(sum_y)

    all_elements = [sum(x) for x in zip(preprocess_coord_x, preprocess_coord_y)]
    encoding_coord = np.zeros ((len (all_info_coord_val), np.max(all_elements)*escala), dtype=int)


    sample = 0
    for i in all_elements:
        for j in range(i * escala):
            encoding_coord[sample, j] = 1
        sample = sample + 1

    encoding_coord_and_episode = np.column_stack ([episodios, encoding_coord])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode [(encoding_coord_and_episode [:, 0] < limit_ep_train + 1)]
    encondign_coord_validation = encoding_coord_and_episode [(encoding_coord_and_episode [:, 0] > limit_ep_train)]

    size_of_input = encondign_coord_train.shape
    encodign_coord_train = encondign_coord_train [:, 1:size_of_input [1]]
    encodign_coord_validation = encondign_coord_validation [:, 1:size_of_input [1]]

    return encodign_coord_train, encodign_coord_validation, encoding_coord


def new_sum_digits_of_each_coord(escala):
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates_s008 ()

    episodios = all_info_coord_val [:, 0]

    all_x_coord_float = all_info_coord_val [:, 1]
    all_x_coord_int = [int (x) for x in all_x_coord_float]
    all_x_coord_str = [str (x) for x in all_x_coord_int]

    all_y_coord_float = all_info_coord_val [:, 2]
    all_y_coord_int = [int (y) for y in all_y_coord_float]
    all_y_coord_str = [str (y) for y in all_y_coord_int]

    preprocess_coord_x = []
    preprocess_coord_y = []

    for i in range (len (all_x_coord_str)):
        num_x = [int (x) for x in all_x_coord_str [i]]
        sum_x = sum (num_x)
        preprocess_coord_x.append (sum_x)

    for i in range (len (all_x_coord_str)):
        num_y = [int (x) for x in all_y_coord_str [i]]
        sum_y = sum (num_y)
        preprocess_coord_y.append (sum_y)

    encoding_coord_x = np.zeros ((len (all_info_coord_val), np.max(preprocess_coord_x) * escala), dtype=int)
    encoding_coord_y = np.zeros ((len (all_info_coord_val), np.max (preprocess_coord_y) * escala), dtype=int)

    sample = 0
    for i in preprocess_coord_x:
        for j in range (i * escala):
            encoding_coord_x[sample, j] = 1
        sample = sample + 1

    sample = 0
    for i in preprocess_coord_y:
        for j in range (i * escala):
            encoding_coord_y[sample, j] = 1
        sample = sample + 1

    encondig_coord = np.concatenate ((encoding_coord_x, encoding_coord_y), axis=1)
    encoding_coord_and_episode = np.column_stack ([episodios, encondig_coord])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode [(encoding_coord_and_episode [:, 0] < limit_ep_train + 1)]
    encondign_coord_validation = encoding_coord_and_episode [(encoding_coord_and_episode [:, 0] > limit_ep_train)]

    size_of_input = encondign_coord_train.shape
    encodign_coord_train = encondign_coord_train [:, 1:size_of_input [1]]
    encodign_coord_validation = encondign_coord_validation [:, 1:size_of_input [1]]

    return encodign_coord_train, encodign_coord_validation, encondig_coord


def new_themomether_coord_x_y_unbalanced_for_s008(escala):
    all_info_coord_val, coord_train, coord_test = read_valid_coordinates_s008 ()

    episodios = all_info_coord_val [:, 0]

    all_x_coord_str = all_info_coord_val [:, 1]
    all_x_coord = [int (x) for x in all_x_coord_str]
    all_y_coord_str = all_info_coord_val [:, 2]
    all_y_coord = [int (y) for y in all_y_coord_str]

    min_x_coord = np.min (all_x_coord)
    min_y_coord = np.min (all_y_coord)
    max_x_coord = np.max (all_x_coord)
    max_y_coord = np.max (all_y_coord)

    diff_all_x_coord = np.array ((all_x_coord - min_x_coord))
    diff_all_y_coord = np.array (all_y_coord) - min_y_coord

    escala = escala
    size_of_data_x = (max_x_coord - min_x_coord) * escala
    size_of_data_y = (max_y_coord - min_y_coord) * escala
    enconding_x = np.array ([len (all_info_coord_val), size_of_data_x], dtype=int)
    enconding_y = np.array ([len (all_info_coord_val), size_of_data_y], dtype=int)  # +escala], dtype=int)
    encoding_coord_vector = np.zeros ((len (all_info_coord_val), size_of_data_x + size_of_data_y), dtype=int)



    encoding_x_vector = np.zeros (enconding_x, dtype=int)
    encoding_y_vector = np.zeros (enconding_y, dtype=int)

    all_encoding_coord_x = np.zeros (len(all_x_coord), dtype=int)
    #all_encoding_coord_x = []
    n_x = 1 * escala

    x_y_diff = diff_all_x_coord + diff_all_y_coord
    sample = 0
    for i in x_y_diff:
        for j in range(i * n_x):  # +escala):
            encoding_coord_vector[sample, j] = 1
        sample = sample + 1

    '''
    sample = 0
    for i in diff_all_x_coord:
        for j in range (i * n_x):  # +escala):
            encoding_x_vector [sample, j] = 1
        sample = sample + 1

    sample = 0
    for i in diff_all_y_coord:
        for j in range (i * n_x):  # +escala):
            encoding_y_vector [sample, j] = 1
        sample = sample + 1

    encondig_coord = np.concatenate ((encoding_x_vector, encoding_y_vector), axis=1)
    '''
    encoding_coord_and_episode = np.column_stack ([episodios, encoding_coord_vector])

    limit_ep_train = 1564
    encondign_coord_train = encoding_coord_and_episode [(encoding_coord_and_episode [:, 0] < limit_ep_train + 1)]
    encondign_coord_validation = encoding_coord_and_episode [(encoding_coord_and_episode [:, 0] > limit_ep_train)]

    size_of_input = encondign_coord_train.shape
    encondign_coord_train = encondign_coord_train [:, 1:size_of_input [1]]
    encondign_coord_validation = encondign_coord_validation [:, 1:size_of_input [1]]

    print ("\t\t Pre processamento das coordenadas")
    print ("-------------------------------------------")
    print ("\tEscala \t\tTrain \t\tValidation")
    print ("\t", escala, "\t\t", encondign_coord_train.shape, "\t", encondign_coord_validation.shape)
    # print ("Tamanho do vetor de entrada Train: ", encondign_coord_train.shape)
    # print ("Tamanho do vetor de entrada Validation: ", encondign_coord_validation.shape)

    return encondign_coord_train, encondign_coord_validation, encoding_coord_vector


