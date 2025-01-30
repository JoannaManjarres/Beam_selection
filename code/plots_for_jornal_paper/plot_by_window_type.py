import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_results(input_type, window_type, ref):
    # Read the data
    path ='../../results/score/'+ref+'/servidor_land/online/'+input_type+'/'+window_type+'/'
    filename = 'all_results_'+window_type+'_top_k.csv'
    if ref =='Batool':
        if window_type == 'sliding_window':
            filename = 'all_results_'+window_type+'_1000_top_k.csv'
    if ref =='Wisard':
        filename = '0_all_results_'+window_type+'_top_k.csv'
    data = pd.read_csv(path+filename)

    top_k = [1, 5, 10, 15, 20, 25, 30]

    all_mean_score_top_k = []
    all_std_score_top_k = []
    for i in range (len (top_k)):
        score_top_k = data [data ['top-k'] == top_k [i]]
        mean = score_top_k ['score'].mean ()
        std = score_top_k ['score'].std ()
        all_mean_score_top_k.append (mean)
        all_std_score_top_k.append (std)

    return data , all_mean_score_top_k, all_std_score_top_k, top_k


def plot_compare_refe_in_fixed_window():
    data_wisard, wisard_mean_score_top_k, wisard_std_score_top_k, top_k = read_results(input_type='lidar', window_type='fixed_window', ref='Wisard')
    data_batool, batool_mean_score_top_k, batool_std_score_top_k, _ = read_results(input_type='lidar', window_type='fixed_window', ref='Batool')

    plt.clf ()
    plt.errorbar (top_k, wisard_mean_score_top_k, yerr=wisard_std_score_top_k,
                  fmt='o-',  markersize=8, capsize=5, color='red', label='Wisard')
    plt.errorbar(top_k, batool_mean_score_top_k, yerr=batool_std_score_top_k,
                 fmt='o-', markersize=8, capsize=5, color='teal', label='Batool')
    for i in range(len(top_k)):
        plt.text(top_k[i]+0.2, wisard_mean_score_top_k[i] - 0.03,
                  str(np.round(wisard_mean_score_top_k[i], 2)), color='red')
        plt.text(top_k[i]+0.2, batool_mean_score_top_k[i] + 0.02,
                  str(np.round(batool_mean_score_top_k[i], 2)), color='teal')

    plt.xticks(top_k)
    plt.ylim([0, 1.1])
    plt.xlim([0, 35])
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('Top-K', fontsize=10)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.ylabel('Score', fontsize=10)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.title ('comparition of references in fixed window with LiDAR', fontsize=12)  # , fontweight='bold', fontname='Myanmar Sangam MN')
    plt.savefig ('../../results/score/plot_for_jornal/fixed_window/compare_ref_by_fixed_window.png',
                 dpi=300, bbox_inches='tight')

def plot_compare_inputs_in_fixed_wind_wisard():
    coord_data, coord_mean_score_top_k, coord_std_score_top_k, top_k = read_results(input_type='coord',
                                                                          window_type='fixed_window',
                                                                          ref='Wisard')
    lidar_data, lidar_mean_score_top_k, lidar_std_score_top_k, top_k = read_results (input_type='lidar',
                                                                               window_type='fixed_window',
                                                                               ref='Wisard')
    coordLidar_data, coordLidar_mean_score_top_k, coordLidar_std_score_top_k, top_k = read_results (input_type='lidar_coord',
                                                                                                  window_type='fixed_window',
                                                                                                  ref='Wisard')
    colors = ['peachpuff', 'lightsteelblue', 'gainsboro']
    plot_barras = True
    if plot_barras:
        inputs = ['coord', 'lidar', 'coord+Lidar']
        heigh_bar = 1

        offset = np.ones(len(top_k))

        fig, ax = plt.subplots (figsize=(6, 8))

        plt.barh(top_k-offset, coord_mean_score_top_k, height=heigh_bar,
                 color=colors[0], label='coord', xerr=coord_std_score_top_k, ecolor='gray')
        plt.barh(top_k, lidar_mean_score_top_k, height=heigh_bar,
                 color=colors[1], label='LiDAR', xerr=lidar_std_score_top_k, ecolor='gray')
        plt.barh(top_k+offset, coordLidar_mean_score_top_k, height=heigh_bar,
                 color=colors[2], label='coord+LiDAR', xerr=coordLidar_std_score_top_k, ecolor='gray')
        ax.spines ['top'].set_visible (False)
        ax.spines ['right'].set_visible (False)
        plt.yticks(top_k, top_k)
        plt.xlabel('Score')
        plt.ylabel('Top-K')
        plt.ylim([-1, 35])
        plt.legend(loc='upper right', ncols=3, frameon=False)
        plt.title('compare of inputs in fixed window - WiSARD')
        plt.savefig ('../../results/score/plot_for_jornal/fixed_window/compare_inputs_WiSARD_by_fixed_window.png',
                     dpi=300, bbox_inches='tight')

    # plot in lines
    else:
        plt.clf ()
        plt.plot(top_k, coord_mean_score_top_k, '--s', markersize=8, color=colors[0], label='coord')
        #plt.fill_between(top_k, np.array(coord_mean_score_top_k) - np.array(coord_std_score_top_k), np.array(coord_mean_score_top_k) + np.array(coord_std_score_top_k), alpha=0.1, color='red')
        plt.plot(top_k, lidar_mean_score_top_k, '--o', markersize=8, color=colors[1], label='LiDAR')
        #plt.fill_between(top_k, np.array(lidar_mean_score_top_k) - np.array(lidar_std_score_top_k), np.array(lidar_mean_score_top_k) + np.array(lidar_std_score_top_k), alpha=0.1, color='teal')
        plt.plot(top_k, coordLidar_mean_score_top_k, '--x', markersize=8, color=colors[2], label='coord+LiDAR')
        #plt.fill_between(top_k, np.array(coordLidar_mean_score_top_k) - np.array(coordLidar_std_score_top_k), np.array(coordLidar_mean_score_top_k) + np.array(coordLidar_std_score_top_k), alpha=0.1, color='blue')

        #plt.errorbar (top_k, coord_mean_score_top_k, yerr=coord_std_score_top_k,
        #              fmt='--s',  markersize=8, capsize=5, color=colors[0], label='coord')
        #plt.errorbar(top_k, lidar_mean_score_top_k, yerr=lidar_std_score_top_k,
        #                fmt='o-', markersize=8, capsize=5, color=colors[1], label='lidar')
        #plt.errorbar(top_k, coordLidar_mean_score_top_k, yerr=coordLidar_std_score_top_k,
        #                fmt='b--x', markersize=8, capsize=5, color=colors[2], label='coordLidar')
        values = False
        if values:
            for i in range(len(top_k)):
                plt.text(top_k[i]+0.2, coord_mean_score_top_k[i] - 0.03,
                          str(np.round(coord_mean_score_top_k[i], 2)), color='red')
                plt.text(top_k[i]+0.2, lidar_mean_score_top_k[i] + 0.02,
                          str(np.round(lidar_mean_score_top_k[i], 2)), color='teal')
                plt.text(top_k[i]+0.2, coordLidar_mean_score_top_k[i] - 0.03,
                          str(np.round(coordLidar_mean_score_top_k[i], 2)), color='blue')
        plt.xlabel('Top-K', fontsize=10)
        plt.ylabel('Score', fontsize=10)
        plt.title('compare of inputs in fixed window', fontsize=12)
        plt.grid()
        plt.legend(loc='lower right', frameon=False)
        plt.show ()

def plot_compare_inputs_in_incremental_wind_wisard():
    coord_data, coord_mean_score_top_k, coord_std_score_top_k, top_k = read_results(input_type='coord',
                                                                          window_type='incremental_window',
                                                                          ref='Wisard')
    lidar_data, lidar_mean_score_top_k, lidar_std_score_top_k, top_k = read_results (input_type='lidar',
                                                                               window_type='incremental_window',
                                                                               ref='Wisard')
    coordLidar_data, coordLidar_mean_score_top_k, coordLidar_std_score_top_k, top_k = read_results (input_type='lidar_coord',
                                                                                                  window_type='incremental_window',
                                                                                                  ref='Wisard')
    colors = ['peachpuff', 'lightsteelblue', 'gainsboro']
    plot_barras = True
    if plot_barras:
        inputs = ['coord', 'lidar', 'coord+Lidar']
        heigh_bar = 1

        offset = np.ones(len(top_k))

        fig, ax = plt.subplots (figsize=(6, 8))

        plt.barh(top_k-offset, coord_mean_score_top_k, height=heigh_bar,
                 color=colors[0], label='coord', xerr=coord_std_score_top_k, ecolor='gray')
        plt.barh(top_k, lidar_mean_score_top_k, height=heigh_bar,
                 color=colors[1], label='LiDAR', xerr=lidar_std_score_top_k, ecolor='gray')
        plt.barh(top_k+offset, coordLidar_mean_score_top_k, height=heigh_bar,
                 color=colors[2], label='coord+LiDAR', xerr=coordLidar_std_score_top_k, ecolor='gray')
        ax.spines ['top'].set_visible (False)
        ax.spines ['right'].set_visible (False)
        plt.yticks(top_k, top_k)
        plt.xlabel('Score')
        plt.ylabel('Top-K')
        plt.ylim([-1, 35])
        plt.legend(loc='upper right', ncols=3, frameon=False)
        plt.title('compare of inputs in incremental window - WiSARD')
        plt.savefig ('../../results/score/plot_for_jornal/incremental_window/compare_inputs_WiSARD_by_incremental_window.png',
                     dpi=300, bbox_inches='tight', pad_inches=0)

    # plot in lines
    else:
        plt.clf ()
        plt.plot(top_k, coord_mean_score_top_k, '--s', markersize=8, color=colors[0], label='coord')
        #plt.fill_between(top_k, np.array(coord_mean_score_top_k) - np.array(coord_std_score_top_k), np.array(coord_mean_score_top_k) + np.array(coord_std_score_top_k), alpha=0.1, color='red')
        plt.plot(top_k, lidar_mean_score_top_k, '--o', markersize=8, color=colors[1], label='LiDAR')
        #plt.fill_between(top_k, np.array(lidar_mean_score_top_k) - np.array(lidar_std_score_top_k), np.array(lidar_mean_score_top_k) + np.array(lidar_std_score_top_k), alpha=0.1, color='teal')
        plt.plot(top_k, coordLidar_mean_score_top_k, '--x', markersize=8, color=colors[2], label='coord+LiDAR')
        #plt.fill_between(top_k, np.array(coordLidar_mean_score_top_k) - np.array(coordLidar_std_score_top_k), np.array(coordLidar_mean_score_top_k) + np.array(coordLidar_std_score_top_k), alpha=0.1, color='blue')

        #plt.errorbar (top_k, coord_mean_score_top_k, yerr=coord_std_score_top_k,
        #              fmt='--s',  markersize=8, capsize=5, color=colors[0], label='coord')
        #plt.errorbar(top_k, lidar_mean_score_top_k, yerr=lidar_std_score_top_k,
        #                fmt='o-', markersize=8, capsize=5, color=colors[1], label='lidar')
        #plt.errorbar(top_k, coordLidar_mean_score_top_k, yerr=coordLidar_std_score_top_k,
        #                fmt='b--x', markersize=8, capsize=5, color=colors[2], label='coordLidar')
        values = False
        if values:
            for i in range(len(top_k)):
                plt.text(top_k[i]+0.2, coord_mean_score_top_k[i] - 0.03,
                          str(np.round(coord_mean_score_top_k[i], 2)), color='red')
                plt.text(top_k[i]+0.2, lidar_mean_score_top_k[i] + 0.02,
                          str(np.round(lidar_mean_score_top_k[i], 2)), color='teal')
                plt.text(top_k[i]+0.2, coordLidar_mean_score_top_k[i] - 0.03,
                          str(np.round(coordLidar_mean_score_top_k[i], 2)), color='blue')
        plt.xlabel('Top-K', fontsize=10)
        plt.ylabel('Score', fontsize=10)
        plt.title('compare of inputs in fixed window', fontsize=12)
        plt.grid()
        plt.legend(loc='lower right', frameon=False)
        plt.show ()

def plot_times_by_ref():
    data_wisard_lidar, _, _, top_k = read_results (input_type='lidar', window_type='fixed_window', ref='Wisard')
    data_wisard_coord, _, _, _ = read_results (input_type='coord', window_type='fixed_window', ref='Wisard')
    data_batool_lidar, _, _, _ = read_results (input_type='lidar', window_type='fixed_window', ref='Batool')

    data_wisard_top_1_lidar = data_wisard_lidar [data_wisard_lidar ['top-k'] == 1]
    data_wisard_top_1_coord = data_wisard_coord [data_wisard_coord ['top-k'] == 1]
    data_batool_top_1_lidar = data_batool_lidar [data_batool_lidar ['top-k'] == 1]

    time_train_wisard_lidar = data_wisard_top_1_lidar ['trainning_process_time'] * 1e-9
    time_train_wisard_coord = data_wisard_top_1_coord ['trainning_process_time'] * 1e-9
    time_train_batool_lidar = data_batool_top_1_lidar ['trainning_process_time'] * 1e-9

    data_wisard = pd.DataFrame()
    data_wisard['LiDAR'] = time_train_wisard_lidar
    data_wisard['Coord'] = time_train_wisard_coord

    data_batool = pd.DataFrame()
    data_batool['LiDAR'] = time_train_batool_lidar

    #fig, ax1 = plt.subplots (figsize=(15, 7))
    #sns.violinplot (data=[data_wisard ['Coord'], data_wisard ['LiDAR']], ax=ax1, palette=['tab:blue', 'tab:orange'])
    #ax1.set_xticklabels (['Coord', 'LiDAR'])

    violin_plot = True
    if violin_plot:
        sns.set_theme (style="darkgrid")
        fig, ax1 = plt.subplots (figsize=(6, 8))
        '''
        for i in range (len (window_size)):
            all_results_sliding_window = pd.read_csv (path_result_sliding_window +
                                                      'all_results_sliding_window_' +
                                                      str (window_size [i]) + '_top_k.csv')
    
            plt.plot (all_results_sliding_window ['episode'],
                      all_results_sliding_window ['trainning Time top-' + str (top_k)] * 1e-9,
                      color=color [i], marker=',', alpha=0.3,
                      label='Sliding window top-' + str (top_k) + '_' + str (window_size [i]))
        '''
        sequencial_color = sns.color_palette("Reds", 2)
        sns.violinplot(data=[data_wisard['Coord'], data_wisard['LiDAR']], ax=ax1,
                       palette=sequencial_color, label='WiSARD', fill=False)

        ax1.set_ylabel ( ' time [s]', fontsize=12, color='tab:red', labelpad=10, fontweight='bold')
        ax1.tick_params (axis='y', labelcolor='tab:red')
        ax1.set_xticklabels (['Coord', 'LiDAR'])
        ax1.legend()
        #ax1.set_ylim ([time_train_wisard.min()-0.001, time_train_wisard.max()+0.001])
        #ax1.set_xlabel ('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold', fontname='Myanmar Sangam MN')

        # Criando um segundo eixo
        ax2 = ax1.twinx ()
        size_font = 10
        #sns.violinplot(data=[data_batool['Batool']], ax=ax1, palette=['tab:blue', 'tab:orange'])
        sequencial_color = sns.color_palette ("Blues", 2)
        sns.violinplot (data=[data_batool['LiDAR'], data_batool['LiDAR']], ax=ax2,
                        palette=sequencial_color, label='Batool', fill=False)#, 'tab:purple', 'tab:brown'])
        ax2.set_ylabel ('time [s]', fontsize=12, color='tab:blue', labelpad=12, fontweight='bold',)  # , color='red')
        ax2.set_xticklabels ([ 'coord', 'LiDAR'])
        ax2.tick_params (axis='y', labelcolor='tab:blue')
        ax2.legend()

        fig.tight_layout ()

    else:
        sns.set_theme (style="darkgrid")
        fig, ax1 = plt.subplots (figsize=(6, 8))
        sequencial_color = sns.color_palette ("Reds", 2)
        sns.histplot(data=[data_wisard['Coord']], ax=ax1,
                     color=sequencial_color, label='WiSARD', bins=10, kde=True, fill=False)
        sns.histplot (data=[data_wisard['LiDAR']], ax=ax1,
                      color=sequencial_color, label='WiSARD', bins=10, kde=True, fill=False)
        #ax1.set_ylabel ('time [s]', fontsize=12, color='tab:red', labelpad=10, fontweight='bold')
        #ax1.tick_params (axis='y', labelcolor='tab:red')
        #ax1.set_xticklabels (['Coord', 'LiDAR'])
        ax1.legend()

        ax2 = ax1.twinx ()
        sequencial_color = sns.color_palette ("Blues", 2)
        sns.histplot (data=[data_batool['LiDAR']], ax=ax2,
                      color=sequencial_color, label='Batool', bins=10, kde=True, fill=False)
        #ax2.set_ylabel ('time [s]', fontsize=12, color='tab:blue', labelpad=12, fontweight='bold')
        #ax2.set_xticklabels (['coord', 'LiDAR'])
        #ax2.tick_params (axis='y', labelcolor='tab:blue')
        ax2.legend()


    #ax2.set_ylim ([time_train_batool.min()-0.001, time_train_batool.max()+0.001])
    a=0
    # Adicionando título e legendas
    #title = "Relationship between  training time and score TOP-" + str (top_k) + "\n using data: " + input_type
    #plt.title (title, fontsize=15, color='black', fontweight='bold')
    #plt.xlabel ('Episode', fontsize=12, color='black', labelpad=10, fontweight='bold')
    #plt.legend (loc='best', ncol=3)  # loc=(0,-0.4), ncol=3)#loc='best')

def plot_compare_times_by_ref():
    data_wisard, _, _, top_k = read_results(input_type='lidar', window_type='fixed_window', ref='Wisard')
    data_batool, _, _, _ = read_results(input_type='lidar', window_type='fixed_window', ref='Batool')



    data_wisard_top_1 = data_wisard[data_wisard['top-k'] == 1]
    data_batool_top_1 = data_batool[data_batool['top-k'] == 1]

    time_train_wisard = data_wisard_top_1['trainning_process_time']* 1e-9
    time_train_batool = data_batool_top_1['trainning_process_time']* 1e-9

    data_wisard_coord, _, _, top_k = read_results (input_type='coord', window_type='fixed_window', ref='Wisard')
    data_wisard_coord_top_1 = data_wisard_coord[data_wisard_coord['top-k'] == 1]
    time_train_wisard_coord = data_wisard_coord_top_1['trainning_process_time'] * 1e-9

    df_wisard = pd.DataFrame ()
    df_wisard['coord'] = time_train_wisard_coord
    df_wisard['lidar'] = time_train_wisard
    #df_wisard['batool'] = time_train_batool


    #sns.histplot(time_train_wisard, color='red', label='Wisard')
    #sns.histplot(time_train_batool, color='teal', label='Batool' )
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #sns.violinplot(df_wisard, color='red', label='Wisard')
    #sns.violinplot(time_train_wisard_coord, color='teal', label='Batool', log_scale=True )
    #ax.set_yscale('log')


    # Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2) = plt.subplots (2, 1, sharey=False, figsize=(4,6))

    sns.violinplot (data=time_train_wisard, split=True, ax=ax1, color='red', alpha=0.5)
    sns.violinplot(data=time_train_wisard_coord, split=True, ax=ax1, color='blue', alpha=0.5)
    #sns.violinplot (df_wisard, color='red', label='Wisard')
    #ax1.set_ylim ([time_train_wisard.min()-0.001, time_train_wisard.max()+0.001])
    ax1.set_ylabel ('Time (s)')
    ax1.set_xticks ([])
    #ax1.set_xlabel ('WiSARD')
    ax1.spines ['top'].set_visible (False)
    ax1.spines ['right'].set_visible (False)
    ax1.spines ['bottom'].set_visible (False)

    sns.violinplot (data=time_train_batool, split=True, ax=ax2, color='teal')  # , x='Deck', y='Num', hue='Transported', split=True)
    ax2.set_ylim ([time_train_batool.min()-5, time_train_batool.max()+5])
    #ax2.yscale('log')
    ax2.set_ylabel ('Time (s)')
    ax2.set_xticks ([])
    ax2.set_xlabel ('B. Salhei')
    ax2.spines ['top'].set_visible (False)
    ax2.spines ['right'].set_visible (False)
    ax2.spines ['bottom'].set_visible (False)


    plt.tight_layout ()
    plt.show()


    fig, axes = plt.subplots (2, 1, figsize=(10, 8), sharey='row')
    plt.subplot (1, 1, 1)
    sns.violinplot (data=time_train_wisard, split=True, ax = axes[0])#, x='Side', y='Num', hue='Transported', split=True)
    axes[0].set_xlim ([0, time_train_wisard.max()])
    #plt.xlabel ('Side')
    #plt.ylabel ('Num')
    #plt.title ('Transported Status by Side and Num')
    plt.grid (True)

    plt.subplot (1, 2, 1)
    sns.violinplot (data=time_train_batool, split=True, ax=axes[1])#, x='Deck', y='Num', hue='Transported', split=True)
    axes[1].set_xlim([0, time_train_batool.max ()])
    #plt.xlabel ('Deck')
    #plt.ylabel ('Num')
    #plt.title ('Transported Status by Deck and Num')
    plt.grid (True)

    plt.show()

    a =0

def calculate_statis(input_type, window_type, ref, flag_fast_experiment=False):
    data, _, _, top_k = read_results(input_type=input_type, window_type=window_type, ref=ref)
    data_top_1 = data[data['top-k'] == 1]
    time_train = data_top_1['trainning_process_time']* 1e-9

    statistics = time_train.describe()

    if flag_fast_experiment:
        data_time_with_valid_train_time = data [data ['trainning_process_time'] != 0]

        top_1 = data_time_with_valid_train_time [data_time_with_valid_train_time ['top-k'] == 1]
        time_train = top_1 ['trainning_process_time'] * 1e-9
        statistics = time_train.describe ()

    print('statistics of time trainning for ', input_type, ' with ', ref)
    print(statistics)
    print('---------------------------------')
    file = pd.DataFrame(data=statistics)
    file.to_csv('../../results/score/plot_for_jornal/'+window_type+'/stats/'+input_type+'_'+ref+'_'+window_type+'_statistics.csv')

    return statistics



input = 'lidar_coord'
window_type = 'fixed_window' # 'incremental_window' 'sliding_window'
ref = 'Batool' # 'Wisard'
calculate_statis(input, window_type, ref)





#plot_compare_inputs_in_incremental_wind_wisard()

