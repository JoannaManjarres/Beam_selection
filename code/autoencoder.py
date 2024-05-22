import keras
import cv2
#import pydot
#import pydotplus
#from pydotplus import graphviz
#from keras.utils.vis_utils import plot_model
#from keras.utils.vis_utils import model_to_dot
#keras.utils.vis_utils.pydot = pydot

from keras import layers
#import pydot
import pre_process_lidar
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
#import graphviz
import numpy as np
#keras.utils.vis_utils.pydot = pydot

def binarize_features_by_thermometer(features):
    '''Este metodo recebe um conjunto de features e binariza as mesmas de acordo com a regra do termometro
    Args:
        features: conjunto de features a serem binarizadas 2 matrizes 25X16'''
    read_external_features = False
    if read_external_features:
        # read as features
        path = "../data/lidar/pre_process_data_2D/autoencoder/"
        input_cache_file = np.load (path + "autoencoderfeatures_lidar_2D_test.npz", allow_pickle=True)
        key = list(input_cache_file.keys ())
        features = input_cache_file [key [0]]

    binarized_by_axis_0 = True
    binarized_by_entire_matrix = False

    if binarized_by_axis_0:
        print("Binarized by arg max of axis 0")
        features_binarized = np.zeros((features.shape[0], (features[0,1].shape[0] * features[0,1].shape[1]*2)), dtype=np.int8)
        for i in range(len(features)):
            sample = features[i]
            matrix_1 = sample[0]
            matrix_2 = sample[1]
            arg_max_matrix_1 = np.argmax(matrix_1)
            arg_max_matrix_2 = np.argmax(matrix_2)
            arg_max_matrix_1_0 = np.argmax(matrix_1, axis=0)
            arg_max_matrix_1_1 = np.argmax(matrix_1, axis=1)
            arg_max_matrix_2_0 = np.argmax(matrix_2, axis=0)
            arg_max_matrix_2_1 = np.argmax(matrix_2, axis=1)

            matriz_1_binary = np.zeros((matrix_1.shape[1], matrix_1.shape[0]), dtype=np.int8)
            matriz_2_binary = np.zeros((matrix_2.shape[1], matrix_2.shape[0]), dtype=np.int8)
            for j in range(len(arg_max_matrix_1_0)):
                for a in range(arg_max_matrix_1_0[j]):
                    matriz_1_binary[j, a] = 1
                for b in range(arg_max_matrix_2_0[j]):
                    matriz_2_binary[j, b] = 1

            matriz_1_binary_in_vector = matriz_1_binary.flatten()
            matriz_2_binary_in_vector = matriz_2_binary.flatten()
            feature_binarized_by_sample = np.concatenate((matriz_1_binary_in_vector, matriz_2_binary_in_vector), axis=0)

            features_binarized[i] = feature_binarized_by_sample
    elif binarized_by_entire_matrix:
        print("Binarized by arg max of entire matrix")
        binarized_matrix_1 = np.zeros((features.shape[0], (features[0,1].shape[0]*features[0,1].shape[1])), dtype=np.int8)
        binarized_matrix_2 = np.zeros((features.shape[0], (features[0,1].shape[0]*features[0,1].shape[1])), dtype=np.int8)
        #feature_binary = np.zeros((features.shape[0], (features[0,1].shape[0]*features[0,1].shape[1]*2)), dtype=np.int8)

        for i in range(len(features)):
            sample = features[i]
            arg_max_matrix_1 = sample[0].argmax()
            arg_max_matrix_2 = sample[1].argmax()

            for j in range(arg_max_matrix_1):
                binarized_matrix_1[i, j] = 1

            for m in range(arg_max_matrix_2):
                binarized_matrix_2[i, m] = 1

            features_binarized = np.concatenate((binarized_matrix_1, binarized_matrix_2), axis=1)



    return features_binarized

def binarize_features_by_threshold(features_train):
    read_external_features = False
    threshold_by_matriz = True
    threshold_based_on_max_of_dataset = False
    model = 4
    threshold_nivel = 100

    if read_external_features:
        if model == 3:
            # read as features
            path = "../data/lidar/pre_process_data_2D/autoencoder/"
            input_cache_file = np.load(path + "features_lidar_2D_test_model_3.npz", allow_pickle=True)
            key = list(input_cache_file.keys())
            features_test = input_cache_file[key [0]]

            #path = "../data/lidar/pre_process_data_2D/autoencoder/"
            input_cache_file = np.load (path + "features_lidar_2D_train_model_3.npz", allow_pickle=True)
            key = list(input_cache_file.keys())
            features_train = input_cache_file[key[0]]

        if model == 2:
            print("Read external features: features_lidar_2D_train_model_2.npz")
            # read as features
            path = "../data/lidar/pre_process_data_2D/autoencoder/"
            input_cache_file = np.load(path + "features_lidar_2D_test_model_2.npz", allow_pickle=True)
            key = list(input_cache_file.keys())
            features_test = input_cache_file[key[0]]

            path = "../data/lidar/pre_process_data_2D/autoencoder/"
            input_cache_file = np.load (path + "features_lidar_2D_train_model_2.npz", allow_pickle=True)
            key = list (input_cache_file.keys ())
            features_train = input_cache_file [key [0]]






    if model == 3:
        feature_binarized_train = np.zeros ((len(features_train), (features_train [0, 1].shape [0] * features_train [0, 1].shape [1] * 2)),
                                      dtype=np.int8)
        feature_binarized_test = np.zeros((len(features_test), (features_test[0, 1].shape[0] * features_test [0, 1].shape [1] * 2)),
            dtype=np.int8)
        if threshold_by_matriz:
            print("Binarized by threshold of each matriz")
            for i in range(len(features_train)):
                sample = features_train[i]
                max_matriz_1 = np.max(sample[0])
                max_matriz_2 = np.max(sample[1])
                t, sample_binarized_0 = cv2.threshold(sample[0], thresh=max_matriz_1 / threshold_nivel, maxval=1,
                                                     type=cv2.THRESH_BINARY, dst=features_train)
                t, sample_binarized_1 = cv2.threshold(sample[1], thresh=max_matriz_2 / threshold_nivel, maxval=1,
                                                       type=cv2.THRESH_BINARY, dst=features_train)
                # t, sample_binarized = cv2.threshold(x_train_encode[i], thresh=max_train_set/2, maxval=1, type=cv2.THRESH_BINARY_INV, dst=x_train_encode)

                sample_binarized_in_vector = np.concatenate((sample_binarized_0.flatten (), sample_binarized_1.flatten ()), axis=0)
                sample_binarized_in_vector = sample_binarized_in_vector.astype('int8')
                feature_binarized_train[i] = sample_binarized_in_vector


            for i in range(len(features_test)):
                sample = features_test[i]
                max_matriz_1 = np.max(sample[0])
                max_matriz_2 = np.max(sample[1])
                t, sample_binarized_0 = cv2.threshold(sample[0], thresh=max_matriz_1 / threshold_nivel, maxval=1,
                                                     type=cv2.THRESH_BINARY, dst=features_test)
                t, sample_binarized_1 = cv2.threshold(sample[1], thresh=max_matriz_2 / threshold_nivel, maxval=1,
                                                       type=cv2.THRESH_BINARY, dst=features_test)
                # t, sample_binarized = cv2.threshold(x_train_encode[i], thresh=max_train_set/2, maxval=1, type=cv2.THRESH_BINARY_INV, dst=x_train_encode)

                sample_binarized_in_vector = np.concatenate((sample_binarized_0.flatten (), sample_binarized_1.flatten ()), axis=0)
                sample_binarized_in_vector = sample_binarized_in_vector.astype('int8')
                feature_binarized_test[i] = sample_binarized_in_vector

            x_train = feature_binarized_train
            x_test = feature_binarized_test
            print ("tamanho do x_train", x_train.shape)
            print ("tamanho do x_test", x_test.shape)

        elif threshold_based_on_max_of_dataset:
            print("Binarized by threshold based on max of dataset")
            max_of_all_features_train = np.max(features_train)

            for i in range(len(features_train)):
                t, sample_binarized = cv2.threshold(features_train[i],
                                                    thresh=max_of_all_features_train/threshold_nivel, maxval=1,
                                                    type=cv2.THRESH_BINARY, dst=features_train)
                # t, sample_binarized = cv2.threshold(x_train_encode[i], thresh=max_train_set/2, maxval=1, type=cv2.THRESH_BINARY_INV, dst=x_train_encode)

                sample_binarized_in_vector = np.concatenate((sample_binarized[0].flatten(), sample_binarized[1].flatten()), axis=0)
                sample_binarized_in_vector = sample_binarized_in_vector.astype('int8')
                feature_binarized_train[i] = sample_binarized_in_vector

            max_of_all_features_test = np.max(features_test)

            for i in range(len(features_test)):
                t, sample_binarized = cv2.threshold(features_test[i],
                                                    thresh=max_of_all_features_test / threshold_nivel, maxval=1,
                                                    type=cv2.THRESH_BINARY, dst=features_test)
                # t, sample_binarized = cv2.threshold(x_train_encode[i], thresh=max_train_set/2, maxval=1, type=cv2.THRESH_BINARY_INV, dst=x_train_encode)

                sample_binarized_in_vector = np.concatenate((sample_binarized[0].flatten(), sample_binarized[1].flatten()), axis=0)
                sample_binarized_in_vector = sample_binarized_in_vector.astype('int8')
                feature_binarized_test[i] = sample_binarized_in_vector

            x_train = feature_binarized_train
            x_test = feature_binarized_test
            print ("tamanho do x_train", x_train.shape)
            print ("tamanho do x_test", x_test.shape)


    elif model == 2:
        x_train_encode = features_train
        x_test_encode = features_test
        feature_binarized_train = np.zeros((len(x_train_encode), (x_train_encode.shape [2] * x_train_encode.shape [3])), dtype=np.int8)

        for i in range(len(x_train_encode)):
            sample = x_train_encode[i]
            max_matriz = np.max(sample)

            t, sample_binarized = cv2.threshold(sample, thresh=max_matriz / threshold_nivel, maxval=1, type=cv2.THRESH_BINARY, dst=x_train_encode)

            sample_binarized_in_vector = sample_binarized.flatten().astype('int8')
            feature_binarized_train[i] = sample_binarized_in_vector

        x_train = feature_binarized_train

        feature_binarized_test = np.zeros((len(x_test_encode), (x_test_encode.shape[2] * x_test_encode.shape[3])),
                                           dtype=np.int8)
        for i in range(len(x_test_encode)):
            sample_test = x_test_encode[i]
            max_matriz = np.max(sample_test)

            t, sample_binarized_test = cv2.threshold(sample_test, thresh=max_matriz / threshold_nivel, maxval=1, type=cv2.THRESH_BINARY, dst=x_test_encode)

            sample_binarized_in_vector = sample_binarized_test.flatten().astype('int8')
            feature_binarized_test[i] = sample_binarized_in_vector

        x_test = feature_binarized_test
        print("tamanho do x_train", x_train.shape)
        print("tamanho do x_test", x_test.shape)

    elif model == 4:
        x_train_encode = features_train
        #x_test_encode = features_test
        #feature_binarized_train = np.zeros((len (x_train_encode), (x_train_encode.shape [2] * x_train_encode.shape [3])), dtype=np.int8)


        feature_binarized_train = np.zeros ((len (x_train_encode), 496), dtype=np.int8)
        for i in range (len (x_train_encode)):
            sample = x_train_encode[i, 0, 0]
            max_matriz = np.max(sample)
            sample_norm = sample / max_matriz

            #t, sample_binarized = cv2.threshold (sample_norm, thresh=0.5, maxval=1,
            #                                     type=cv2.THRESH_BINARY, dst=x_train_encode)

            sample_binarized = np.where(sample_norm > 0.8, 1, 0).tolist()

            index = 0
            termomether = np.zeros((1, 496), dtype=np.int8)
            for j in range(len(sample_binarized)):
                if sample_binarized[j] == 1:
                    index = index + j

            for term in range (0, index):
                termomether[0, term] = 1

            feature_binarized_train[i] = termomether



        x_train = feature_binarized_train
    return x_train








def autoencoder():
    use_thermometer = False
    #data_lidar_2D_train, data_lidar_2D_test = pre_process_lidar.process_data_lidar_to_2D()
    _,_, data_lidar_2D_matrix_train, data_lidar_2D_matrix_test = pre_process_lidar.process_data_lidar_into_2D_matrix()
    data_lidar_2D_train = data_lidar_2D_matrix_train
    data_lidar_2D_test = data_lidar_2D_matrix_test

    input_shape = (data_lidar_2D_train.shape[1], data_lidar_2D_train.shape[2])

    #------ created autoencoder model
    # This is input image
    input_img = keras.Input(shape=(20, 200, 1))

    type_model = 4

    if type_model == 1:
        #Encoder
        conv1_1 = layers.Conv2D(4, (3,3), activation='relu', padding='same')(input_img)
        pool1 = layers.MaxPooling2D((2, 2), padding='same')(conv1_1)
        conv1_2 = layers.Conv2D(8, (3,3), activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPooling2D((2, 2), padding='same')(conv1_2)
        conv1_3 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(pool2)
        h = layers.MaxPooling2D((3, 2), padding='same')(conv1_3)

        #Decoder
        conv2_1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(h)
        up1 = layers.UpSampling2D((5, 2))(conv2_1)
        conv2_2 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
        up2 = layers.UpSampling2D((1, 2))(conv2_2)
        conv2_3 = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(up2)
        up3 = layers.UpSampling2D((2, 2))(conv2_3)
        r = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up3)

    elif type_model == 2:
        # Encoder
        conv1_1 = layers.Conv2D (16, (5, 5), activation='relu', padding='same') (input_img)
        pool1 = layers.MaxPooling2D ((2, 2), padding='same') (conv1_1)
        conv1_2 = layers.Conv2D (32, (5, 5), activation='relu', padding='same') (pool1)
        pool2 = layers.MaxPooling2D ((2, 2), padding='same') (conv1_2)
        conv1_3 = layers.Conv2D (64, (5, 5), activation='relu', padding='same') (pool2)
        h = layers.MaxPooling2D ((6, 2), padding='same') (conv1_3)

        # Decoder
        conv2_1 = layers.Conv2D (64, (5, 5), activation='relu', padding='same') (h)
        up1 = layers.UpSampling2D ((5, 2)) (conv2_1)
        conv2_2 = layers.Conv2D (32, (5, 5), activation='relu', padding='same') (up1)
        up2 = layers.UpSampling2D ((2, 2)) (conv2_2)
        conv2_3 = layers.Conv2D (16, (5, 5), activation='relu', padding='same') (up2)
        up3 = layers.UpSampling2D ((2, 2)) (conv2_3)
        r = layers.Conv2D (1, (5, 5), activation='sigmoid', padding='same') (up3)

    elif type_model == 3:
        # Encoder
        conv1_1 = layers.Conv2D (4, (5, 5), activation='relu', padding='same') (input_img)
        pool1 = layers.MaxPooling2D ((2, 2), padding='same') (conv1_1)
        conv1_2 = layers.Conv2D (8, (5, 5), activation='relu', padding='same') (pool1)
        pool2 = layers.MaxPooling2D ((2, 2), padding='same') (conv1_2)
        conv1_3 = layers.Conv2D (16, (5, 5), activation='relu', padding='same') (pool2)
        h = layers.MaxPooling2D ((3, 2), padding='same') (conv1_3)


        # Decoder
        conv2_1 = layers.Conv2D (16, (5, 5), activation='relu', padding='same') (h)
        up1 = layers.UpSampling2D ((5, 2)) (conv2_1)
        conv2_2 = layers.Conv2D (8, (5, 5), activation='relu', padding='same') (up1)
        up2 = layers.UpSampling2D ((1, 2)) (conv2_2)
        conv2_3 = layers.Conv2D (4, (5, 5), activation='relu', padding='same') (up2)
        up3 = layers.UpSampling2D ((2, 2)) (conv2_3)
        r = layers.Conv2D (1, (5, 5), activation='sigmoid', padding='same') (up3)

    elif type_model == 4:
        kernel = 5
        # Encoder
        #(20,200)
        conv1_1 = layers.Conv2D(8, (kernel, kernel), activation='relu', padding='same')(input_img)
        #(20,200,16)
        pool1 = layers.MaxPooling2D((2, 2), padding='same')(conv1_1)
        #(10,100,16)
        conv1_2 = layers.Conv2D(16, (kernel, kernel), activation='relu', padding='same')(pool1)
        #(10,100,32)
        pool2 = layers.MaxPooling2D((2, 2), padding='same')(conv1_2)
        #(5,50,32)
        conv1_3 = layers.Conv2D(32, (kernel, kernel), activation='relu', padding='same')(pool2)
        #(5,50,64)
        pool3 = layers.MaxPooling2D ((5, 2), padding='same')(conv1_3)
        #(1,25,64)
        conv1_4 = layers.Conv2D(32, (kernel, kernel), activation='relu', padding='same')(pool3)
        #(1,25,128)
        pool4 = layers.MaxPooling2D((1, 5), padding='same')(conv1_4)
        #(1,5,128)
        h = layers.MaxPooling2D((1, 5), padding='same')(pool4)
        #(1,1,128)

        # Decoder
        conv2_1 = layers.Conv2D(32, (kernel, kernel), activation='relu', padding='same') (h)
        #(1,1,128)
        up1 = layers.UpSampling2D((5, 5))(conv2_1)
        #(5,5,128)
        conv2_2 = layers.Conv2D(32, (kernel, kernel), activation='relu', padding='same')(up1)
        #(5,5,64)
        up2 = layers.UpSampling2D((2, 4))(conv2_2)
        #(10,20,64)
        conv2_3 = layers.Conv2D(16, (kernel, kernel), activation='relu', padding='same')(up2)
        #(10,20,16)
        up3 = layers.UpSampling2D((2, 5))(conv2_3)
        #(20,100,16)
        conv2_4 = layers.Conv2D(8, (kernel, kernel), activation='relu', padding='same')(up3)
        #(20,100,8)
        up4 = layers.UpSampling2D((1, 2))(conv2_4)
        #(20,200,8)
        #conv2_5 = layers.Conv2D(8, (kernel, kernel), activation='relu', padding='same') (up4)
        #(10,100,8)
        #up5 = layers.UpSampling2D ((2, 2)) (conv2_5)
        #(20,200,8)
        r = layers.Conv2D(1, (2, 2), activation='sigmoid', padding='same')(up4)


    # This model maps an input to its reconstruction
    model = keras.Model(inputs=input_img, outputs=r)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #optimizer=''adadelta''
    model.summary()
    #plot_model(model, 'autoencoder.png' )#, show_shapes=True, show_layer_names=True)
    #, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

    #model.fit (data_lidar_2D_train, data_lidar_2D_train, epochs=10)
    history = model.fit(data_lidar_2D_train, data_lidar_2D_train, epochs=10, batch_size=128, shuffle=True, validation_split=0.2)


    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()

    reconstruction_data = model.predict(data_lidar_2D_train)

    # This model maps an input to its encoded representation
    fig = plt.figure(figsize=(5, 4))
    rows = 2
    cols = 1
    sample_for_plot = 4
    fig.add_subplot(rows, cols, 1)
    plt.imshow(data_lidar_2D_train[sample_for_plot], cmap='Greys', origin='lower', extent=[0, 200, 0, 20])
    fig.suptitle('Original/Decoded Image')

    fig.add_subplot(rows, cols, 2)
    plt.imshow(reconstruction_data[sample_for_plot], cmap='Greys', origin='lower', extent=[0, 200, 0, 20])
    #fig.suptitle (' Image')

    encoder_model = keras.Model(inputs=input_img, outputs=h)
    x_train_encode = encoder_model.predict(data_lidar_2D_train)
    x_test_encode = encoder_model.predict(data_lidar_2D_test)

    #save the features in a file
    #saveDataPath = "../data/lidar/pre_process_data_2D/autoencoder/"
    #np.savez(saveDataPath + 'features_lidar_2D_train_model_2' + '.npz', data_lidar=x_train_encode)
    #np.savez(saveDataPath + 'features_lidar_2D_test_model_2' + '.npz', data_lidar=x_test_encode)



    if type_model == 2:
        if use_thermometer:
            print("Binarized by thermometer")


        else:
            print("Binarized by threshold")
            threshold_nivel = 4
            feature_binarized_train = np.zeros((len(x_train_encode), (x_train_encode.shape[2] * x_train_encode.shape[3])),
                                          dtype=np.int8)
            for i in range(len(x_train_encode)):
                sample = x_train_encode[i]
                max_matriz = np.max(sample)

                t, sample_binarized = cv2.threshold(sample, thresh=max_matriz / threshold_nivel, maxval=1,
                                                       type=cv2.THRESH_BINARY, dst=x_train_encode)

                sample_binarized_in_vector = sample_binarized.flatten().astype('int8')
                #sample_binarized_in_vector = sample_binarized_in_vector.astype('int8')
                feature_binarized_train[i] = sample_binarized_in_vector

            x_train = feature_binarized_train

            feature_binarized_test = np.zeros((len(x_test_encode), (x_test_encode.shape[2] * x_test_encode.shape[3])), dtype=np.int8)
            for i in range(len(x_test_encode)):
                sample_test = x_test_encode[i]
                max_matriz = np.max(sample_test)

                t, sample_binarized_test = cv2.threshold(sample_test, thresh=max_matriz / threshold_nivel, maxval=1, type=cv2.THRESH_BINARY, dst=x_test_encode)

                sample_binarized_in_vector = sample_binarized_test.flatten().astype('int8')
                feature_binarized_test[i] = sample_binarized_in_vector

            x_test = feature_binarized_test

    if type_model == 3:

        if use_thermometer:
            print("Binarized by thermometer")
            feature_binarized_train = binarize_features_by_thermometer(x_train_encode)
            feature_binarized_test = binarize_features_by_thermometer(x_test_encode)
            x_train = feature_binarized_train
            x_test = feature_binarized_test


        else:
            print("Binarized by threshold")
            feature_binarized_train = binarize_features_by_threshold(x_train_encode)
            feature_binarized_test = binarize_features_by_threshold(x_test_encode)
            x_train = feature_binarized_train
            x_test = feature_binarized_test

    if type_model == 4:
        x_train = binarize_features_by_threshold (x_train_encode)
        x_test = binarize_features_by_threshold (x_test_encode)


    #train_loss = keras.losses.mae(reconstruction_data, data_lidar_2D_train)

    #plt.hist (train_loss [None, :], bins=50)
    #plt.xlabel ("Train loss")
    #plt.ylabel ("No of examples")
    #plt.show ()

    return x_train, x_test

#autoencoder()
#binarize_features_by_thermometer()
#binarize_features_by_threshold()
