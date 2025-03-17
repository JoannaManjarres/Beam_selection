import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import analyse_s009 as s009
import analyse_s008 as s008
import pre_process_coord as pre


def read_data():
    s009_data = s009.read_raw_coord ()
    s008_data, _, __ = pre.read_valid_coordinates_s008 ()

    dataset_s008 = pd.DataFrame({'episode': s008_data [:, 0], 'x': s008_data [:, 1], 'y': s008_data [:, 2], 'z': s008_data [:, 3], 'LOS': s008_data [:, 4]})
    data_of_s009 = np.array(s009_data)
    dataset_s009 = pd.DataFrame({'episode': data_of_s009 [:, 0], 'LOS': data_of_s009 [:, 1], 'x': data_of_s009 [:, 2], 'y': data_of_s009 [:, 3], 'z': data_of_s009 [:, 4]})


    return dataset_s008, dataset_s009

def plot_samples(dataset_s008, dataset_s009):
    s008 = dataset_s008 ['episode'].tolist()
    episode_s008, samples_for_episode_s008 = np.unique(s008, return_counts=True)

    s009 = np.array (dataset_s009 ['episode'])
    s009 = [int (numeric_string) for numeric_string in s009]
    episode_s009, samples_for_episode_s009 = np.unique (s009, return_counts=True)

    mean_s008=[]
    mean_s009=[]
    start= 0
    for i in range(21):
        end = start+100
        a = np.mean(samples_for_episode_s008[start:end])
        mean_s008.append(a)
        b = np.mean(samples_for_episode_s009[start:end])
        mean_s009.append(b)
        start = start+100


    mean_episodes = np.arange(0, 2079, 100)

    plt.figure(figsize=(14, 6))
    plt.plot(mean_episodes, mean_s008, '-o', label='s008', color='blue')
    plt.plot (mean_episodes, mean_s009, '-o', label='s009', color='orange')
    plt.xlabel ('episódios', fontsize=16, font='Times New Roman')
    plt.ylabel ('média das amostras', fontsize=16, font='Times New Roman')
    #plt.title('Mean of samples per 100 episodes')
    plt.grid (True)
    plt.xticks(mean_episodes)
    plt.legend ()
    plt.savefig( '../analyses/mean_samples_by_episode.png', dpi=300, bbox_inches='tight')
    plt.close ()

    plt.figure (figsize=(14, 6))
    #plt.plot(episode_s008, samples_for_episode_s008, 'o-', label='s008')
    plt.bar(episode_s008, samples_for_episode_s008, label='s008')
    plt.bar(episode_s009, samples_for_episode_s009,  label='s009')#, alpha=0.5)
    #plt.plot(episode_s009, samples_for_episode_s009, '--', label='s009', alpha=0.5)
    plt.text(0, np.max(samples_for_episode_s008),  'Mean: '+str(np.round(np.mean(samples_for_episode_s008),3)), fontsize=12, color='blue')
    plt.text(500, np.max(samples_for_episode_s009),  'Mean: '+str(np.round(np.mean(samples_for_episode_s009), 3)), fontsize=12, color='orange')
    plt.xlabel('episodes')
    plt.ylabel('samples')
    plt.title('Samples per episode')
    plt.legend()
    plt.savefig ('../analyses/samples_by_episode.png', dpi=300)
    plt.close ()

def plot_samples_NLOS():
    dataset_s008, dataset_s009 = read_data()

    s008_NLOS = dataset_s008[dataset_s008['LOS'] == 'LOS=0']
    s008_ep_NLOS = s008_NLOS ['episode'].tolist ()
    episode_NLOS_s008, samples_for_episode_s008_NLOS = np.unique (s008_ep_NLOS, return_counts=True)


    s008_LOS = dataset_s008[dataset_s008['LOS'] == 'LOS=1']
    s008_ep_LOS = s008_LOS['episode'].tolist()
    episode_s008_LOS, samples_for_episode_s008_LOS = np.unique(s008_ep_LOS, return_counts=True)

    s009_NLOS = dataset_s009[dataset_s009['LOS'] == 'LOS=0']
    s009_ep_NLOS = np.array (s009_NLOS ['episode']).tolist ()
    s009_ep_NLOS = [int (numeric_string) for numeric_string in s009_ep_NLOS]
    episode_NLOS_s009, samples_for_episode_s009_NLOS = np.unique (s009_ep_NLOS, return_counts=True)

    s009_LOS = dataset_s009[dataset_s009['LOS'] == 'LOS=1']
    s009_ep_LOS = np.array(s009_LOS['episode']).tolist()
    s009_ep_LOS = [int(numeric_string) for numeric_string in s009_ep_LOS]
    episode_s009_LOS, samples_for_episode_s009_LOS = np.unique(s009_ep_LOS, return_counts=True)

    samples_to_plot = 100
    ep_s008_NLOS, mean_s008_NLOS = calculate_mean_of_samples(episode_NLOS_s008, samples_for_episode_s008_NLOS, samples_to_plot)
    ep_s009_NLOS, mean_s009_NLOS = calculate_mean_of_samples(episode_NLOS_s009, samples_for_episode_s009_NLOS, samples_to_plot)
    ep_s008_LOS, mean_s008_LOS = calculate_mean_of_samples(episode_s008_LOS, samples_for_episode_s008_LOS, samples_to_plot)
    ep_s009_LOS, mean_s009_LOS = calculate_mean_of_samples(episode_s009_LOS, samples_for_episode_s009_LOS, samples_to_plot)

    plt.figure(figsize=(14, 6))
    #plt.plot(episode_s008, samples_for_episode_s008, 'o', label='s008', color='blue')
    plt.plot(ep_s008_NLOS, mean_s008_NLOS, '-o', label='s008 NLOS -> '+str(len(s008_NLOS)), color='blue')
    plt.plot (ep_s009_NLOS, mean_s009_NLOS, '-o', label='s009 NLOS -> '+str(len(s009_NLOS)), color='red')
    #plt.plot(ep, mean_s009_NLOS, '-o', label='s009 NLOS', color='orange')
    plt.plot(ep_s008_LOS, mean_s008_LOS, '--o', label='s008 LOS -> '+str(len(s008_LOS)), color='blue')
    plt.plot(ep_s009_LOS, mean_s009_LOS, '--o', label='s009 LOS -> '+str(len(s009_LOS)), color='red')
    #plt.bar(episode_s008[:100], samples_for_episode_s008[:100], label='s008')

    plt.xlabel('episodes')
    plt.ylabel('samples')
    plt.legend(ncol=2)
    plt.xticks(ep_s008_LOS, fontsize=5, rotation=90)
    plt.title('Média de amostras por cada '+str(samples_to_plot)+' episódios')
    a=0

def calculate_mean_of_samples(episodes, samples_for_episode, samples_to_plot):
    mean = []
    ep = []
    var = int(np.round (len (episodes) / samples_to_plot))

    start = 0
    for i in range (var):
        end = start + samples_to_plot
        a = np.mean(samples_for_episode[start:end])
        mean.append(a)
        ep.append(start)
        start = start + samples_to_plot

    return ep, mean


plot_samples_NLOS()
dataset_s008, dataset_s009 =read_data()
plot_samples(dataset_s008, dataset_s009)
