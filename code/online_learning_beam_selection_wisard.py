import analyse_s009 as s009
import analyse_s008 as s008
import numpy as np
import pandas as pd
import pre_process_coord
import wisardpkg as wp
import timeit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


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

    _, _, input_train = pre_process_coord.Thermomether_coord_x_y_unbalanced_for_s008 (escala=preprocess_resolution)
    index_beams_train, index_beam_validation, _, _ = s008.read_beams_output_from_baseline ()
    label_train = np.concatenate ((index_beams_train, index_beam_validation), axis=0)

    return input_train, label_train

def main():
    preprocess_resolution = 16
    info_s009, encoding_coord_test, beams_s009 = read_s009_data(preprocess_resolution)
    episode_for_test = np.arange(0, 40, 1)
    #episode_for_test =[0,1]#,2,3,4,5]#,6,7,8,9]
    input_train, label_train = read_s008_data (preprocess_resolution)
    input_train = input_train.tolist()
    label_train = label_train.tolist()


    all_score=[]
    time_of_train = []
    time_of_test = []
    for i in range(len(episode_for_test)):
        all_info_from_episode = info_s009[info_s009['Episode'] == episode_for_test[i]]
        index = all_info_from_episode.index
        #labels = all_info_from_episode['beams_index']

        input_test = []
        label_test = []

        for j in range(len(index)):
                encode_coord = encoding_coord_test[index[j]]
                beams_index = beams_s009[index[j]]
                input_test.append((encode_coord))
                label_test.append(beams_index)

        #input_test = np.array(input_test, dtype='int8')
        #index_true = np.array(label_test)

        index_predict, trainning_time, test_time = beam_selection_wisard(data_train=input_train,
                                                                         data_validation=input_test,
                                                                         label_train=label_train,
                                                                         addressSize=44)
        score = accuracy_score(label_test, index_predict)
        all_score.append(score)
        time_of_train.append(trainning_time)
        time_of_test.append(test_time)

        for i in range(len(input_test)):
            input_train.append(input_test[i])
            label_train.append(label_test[i])

    average_score = []
    var = 0
    for i in range(len(all_score)):
        i = i+1
        average_score.append(np.mean(all_score[0:i]))

    plt.plot(episode_for_test, average_score, 'o-', label='Cumulative average accuracy')
    plt.plot(episode_for_test, all_score, 'o--', color='red', label='Accuracy per episode')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right', bbox_to_anchor=(1.04,0))
    plt.title('Beam selection using WiSARD')
    plt.savefig('score.png')
    plt.close()

    plt.plot(episode_for_test, time_of_train, 'o-', color='green')
    plt.xlabel ('Episode')
    plt.ylabel ('Trainning Time')
    plt.savefig ('time_train.png')
    plt.close ()

    plt.plot(episode_for_test, time_of_test, 'o-', color='blue')
    plt.xlabel ('Episode')
    plt.ylabel ('Test Time')
    plt.savefig ('time_test.png')
    plt.close ()

    a=var+1


main()





