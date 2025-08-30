import read_data
import scipy.stats as st
import numpy as np
import pandas as pd

def read_classes(dataset):
    if dataset == 's008':
        data_LOS, data_NLOS, data_ALL = read_data.read_data_s008 (scale_to_coord=16)
        classes_LOS_s008 = data_LOS ['index_beams'].value_counts ()
        classes_NLOS_s008 = data_NLOS ['index_beams'].value_counts ()
        classes_ALL_s008 = data_ALL ['index_beams'].value_counts ()
        return classes_LOS_s008, classes_NLOS_s008, classes_ALL_s008

    elif dataset == 's009':
        data_LOS, data_NLOS, data_ALL = read_data.read_data_s009 (scale_to_coord=16)
        classes_LOS_s009 = data_LOS ['index_beams'].value_counts ()
        classes_NLOS_s009 = data_NLOS ['index_beams'].value_counts ()
        classes_ALL_s009 = data_ALL ['index_beams'].value_counts ()
        return classes_LOS_s009, classes_NLOS_s009, classes_ALL_s009





def plot_histogram_with_entropy(index_beam_all, index_beam_LOS, index_beam_NLOS,
                                All_entropy, LOS_entropy, NLOS_entropy,
                                path_to_save, dataset):
    #title = title
    import matplotlib.pyplot as plt
    import seaborn as sns

    percent = True
    if percent:
        labels_all, counts_all = np.unique (index_beam_all, return_counts=True)
        percent_all = [i / sum (counts_all) * 100 for i in counts_all]

        labels_LOS, counts_LOS = np.unique (index_beam_LOS, return_counts=True)
        percent_LOS = [i / sum (counts_LOS) * 100 for i in counts_LOS]

        labels_NLOS, counts_NLOS = np.unique (index_beam_NLOS, return_counts=True)
        percent_NLOS = [i / sum (counts_NLOS) * 100 for i in counts_NLOS]

    #sns.set(style='darkgrid')
    #sns.set(rc={'figure.figsize': (8, 4)})
    #fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig = plt.figure()
    plt.rcParams ['font.family'] = 'Times New Roman'
    #plt.rcParams ['font.size'] = 12
    gs = fig.add_gridspec (3, hspace=0)
    axs = gs.subplots (sharex=True)#, sharey=True)

    if percent:
        axs[0].bar (labels_all, percent_all, color='green', alpha=0.6)
        axs[1].bar (labels_LOS, percent_LOS, color='steelblue', alpha=0.6)
        axs[2].bar (labels_NLOS, percent_NLOS, color='sandybrown', alpha=0.6)
        if dataset =='s008':
            figure_name = 'histogram_percent_with_entropy_s008.png'
        else:
            figure_name = 'histogram_percent_with_entropy_s009.png'

    else:
        if dataset == 's008':
            figure_name = 'histogram_with_entropy_s008.png'
        else:
            figure_name = 'histogram_with_entropy_s009.png'
        axs[0].hist(index_beam_all, bins=256, density=False, alpha=0.6, color='green')
        axs[1].hist(index_beam_LOS, bins=256, density=False, alpha=0.6, color='steelblue')
        axs[2].hist(index_beam_NLOS, bins=256, density=False, alpha=0.6, color='sandybrown')

    axs[0].text(0.1, 0.9, 'All Data', horizontalalignment='center', fontsize=11, verticalalignment='center', transform=axs[0].transAxes)
    axs[0].text(0.9, 0.85, f'samples: {len(index_beam_all)}', fontsize=11, horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    t = axs[0].text(0.9, 0.65, f'Entropy: {All_entropy:.3f}', fontsize=11, horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    t.set_bbox(dict(facecolor='green', alpha=0.3, edgecolor='green', linewidth=0))

    axs[1].text(0.1, 0.9, 'LOS Data', fontsize=11, horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    axs[1].text(0.9, 0.85, f'samples: {len(index_beam_LOS)}', fontsize=11, horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    t = axs[1].text(0.9, 0.65, f'Entropy: {LOS_entropy:.3f}', fontsize=11, horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    t.set_bbox(dict(facecolor='steelblue', alpha=0.3, edgecolor='steelblue', linewidth=0))

    axs[2].text(0.1, 0.9, 'NLOS Data', fontsize=11, horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
    axs[2].text(0.9, 0.85, f'samples: {len(index_beam_NLOS)}', fontsize=11, horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
    t = axs[2].text(0.9, 0.65, f'Entropy: {NLOS_entropy:.3f}', fontsize=11, horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
    t.set_bbox(dict(facecolor='sandybrown', alpha=0.3, edgecolor='sandybrown', linewidth=0))

    plt.xlabel('Beam Index')
    #plt.ylabel('Frequency (%)' if percent else 'Frequency')
    fig.supylabel('Frequency (%)' if percent else 'Frequency')
    plt.savefig (path_to_save+figure_name, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_classes_sample_percentages_of_intersection_dataset(classes_of_dataset_s008, classes_of_dataset_S009):

    classes_in_both_dataset = set(classes_of_dataset_s008.index).intersection(set(classes_of_dataset_S009.index))

    samples_and_classes_into_intersection_in_s008 = classes_of_dataset_s008[classes_of_dataset_s008.index.isin(classes_in_both_dataset)]
    samples_into_intersection_in_s008 = samples_and_classes_into_intersection_in_s008.values.tolist()
    classes_into_intersection_in_s008 = samples_and_classes_into_intersection_in_s008.index.tolist()
    percentage_of_samples_in_intersection_LOS_s008 = np.round((samples_into_intersection_in_s008/classes_of_dataset_s008.sum()) * 100, 2)

    samples_and_classes_into_intersection_in_s009 = classes_of_dataset_S009[classes_of_dataset_S009.index.isin(classes_in_both_dataset)]
    samples_into_intersection_in_s009 = samples_and_classes_into_intersection_in_s009.values.tolist()
    classes_into_intersection_in_s009 = samples_and_classes_into_intersection_in_s009.index.tolist()
    percentage_of_samples_in_intersection_LOS_s009 = np.round((samples_into_intersection_in_s009 / classes_of_dataset_S009.sum()) * 100, 2)

    info_of_dataset = {'num_classes_in_s008': len(classes_of_dataset_s008),
                       'num_classes_in_s009': len(classes_of_dataset_S009),
                       'num_classes_in_both_datasets': len(classes_in_both_dataset)}

    data_about_intersection_in_s008 = {'classes': classes_into_intersection_in_s008,
                                       'samples': samples_into_intersection_in_s008,
                                       'percentage': percentage_of_samples_in_intersection_LOS_s008}

    data_about_intersection_in_s009 = {'classes': classes_into_intersection_in_s009,
                                        'samples': samples_into_intersection_in_s009,
                                        'percentage': percentage_of_samples_in_intersection_LOS_s009}

    return info_of_dataset, data_about_intersection_in_s008, data_about_intersection_in_s009
def classes_intersection( type_connection='LOS'):
    ''' This function calculates the intersection of classes between two datasets (s008 and s009).
    It returns the intersection of classes for LOS, NLOS, and ALL connections.
    It can return either the samples in the intersection or the percentages of samples in the intersection,
    depending on the type_data_analysis parameter. (samples or percentages)'''

    classes_LOS_s009, classes_NLOS_s009, classes_ALL_s009 = read_classes (dataset='s009')
    classes_LOS_s008, classes_NLOS_s008, classes_ALL_s008 = read_classes (dataset='s008')

    """
    Calculate the intersection of classes between two datasets.
    """
    # Calculate the intersection of classes
    print("Intersection of classes:")
    if type_connection == 'LOS':
        info_of_dataset, data_about_intersection_in_s008, data_about_intersection_in_s009 = calculate_classes_sample_percentages_of_intersection_dataset (classes_of_dataset_s008=classes_LOS_s008,
                                                                                                                                                           classes_of_dataset_S009=classes_LOS_s009)
        diff_s009_less_s008 = classes_LOS_s009 - classes_LOS_s008
        'class_with_nan: signifies that the class exists in one dataset but not in the other'
        'class_with_diff_equal_to_zero: signifies that the class exists in both datasets with the same number of samples'
        ('report_classes_in_one_of_datasets: a dictionary with the classes that exist in one of the datasets and '
                                            'the number of samples in each dataset')
        class_with_nan_s009_less_s008, class_with_diff_equal_to_zero_s009_less_s008, report_classes_in_one_of_datasets_s009_less_s008 = report_of_detalhes_distributions_differences (dataset_s008=classes_LOS_s008,
                                                      dataset_s009=classes_LOS_s009,
                                                      diff_dataset1_less_dataset2=diff_s009_less_s008)

        nan_and_diff_equal_zero = [class_with_nan_s009_less_s008.tolist(), class_with_diff_equal_to_zero_s009_less_s008.tolist()]
        diff_s008_less_s009 = classes_LOS_s008 - classes_LOS_s009
        class_with_nan_s008_less_s009, class_with_diff_equal_to_zero_s008_less_s009, report_classes_in_one_of_datasets_s008_less_s009 = report_of_detalhes_distributions_differences (dataset_s008=classes_LOS_s008,
                                                        dataset_s009=classes_LOS_s009,
                                                        diff_dataset1_less_dataset2=diff_s008_less_s009)

        a=0






    elif type_connection == 'NLOS':
        info_of_dataset, data_about_intersection_in_s008, data_about_intersection_in_s009 = calculate_classes_sample_percentages_of_intersection_dataset (classes_of_dataset_s008=classes_NLOS_s008,
                                                                                                                                                           classes_of_dataset_S009=classes_NLOS_s009)
        diff_s009_less_s008 = classes_NLOS_s009 - classes_NLOS_s008
        class_with_nan_s009_less_s008, class_with_diff_equal_to_zero_s009_less_s008, report_classes_in_one_of_datasets_s009_less_s008 = report_of_detalhes_distributions_differences (dataset_s008=classes_NLOS_s008,
                                                        dataset_s009=classes_NLOS_s009,
                                                        diff_dataset1_less_dataset2=diff_s009_less_s008)
        diff_s008_less_s009 = classes_NLOS_s008 - classes_NLOS_s009
        class_with_nan_s008_less_s009, class_with_diff_equal_to_zero_s008_less_s009, report_classes_in_one_of_datasets_s008_less_s009 = report_of_detalhes_distributions_differences (dataset_s008=classes_NLOS_s008,
                                                        dataset_s009=classes_NLOS_s009,
                                                        diff_dataset1_less_dataset2=diff_s008_less_s009)

    elif type_connection == 'ALL':
        info_of_dataset, data_about_intersection_in_s008, data_about_intersection_in_s009 = calculate_classes_sample_percentages_of_intersection_dataset (classes_of_dataset_s008=classes_ALL_s008,
                                                                                                                                                              classes_of_dataset_S009=classes_ALL_s009)
        diff_s009_less_s008 = classes_ALL_s009 - classes_ALL_s008
        class_with_nan_s009_less_s008, class_with_diff_equal_to_zero_s009_less_s008, report_classes_in_one_of_datasets_s009_less_s008 = report_of_detalhes_distributions_differences (dataset_s008=classes_ALL_s008,
                                                          dataset_s009=classes_ALL_s009,
                                                          diff_dataset1_less_dataset2=diff_s009_less_s008)
        diff_s008_less_s009 = classes_ALL_s008 - classes_ALL_s009
        class_with_nan_s008_less_s009, class_with_diff_equal_to_zero_s008_less_s009, report_classes_in_one_of_datasets_s008_less_s009 = report_of_detalhes_distributions_differences (dataset_s008=classes_ALL_s008,
                                                          dataset_s009=classes_ALL_s009,
                                                          diff_dataset1_less_dataset2=diff_s008_less_s009)


    return info_of_dataset, data_about_intersection_in_s008, data_about_intersection_in_s009, nan_and_diff_equal_zero, report_classes_in_one_of_datasets_s009_less_s008
    # calculate the number of samples in the intersection for each dataset



    '''
    plot_distribution = False
    if plot_distribution:
        # make a plot the intersection of classes
        import matplotlib.pyplot as plt
        plt.figure (figsize=(10, 6))
        plt.bar (intersection_classes_LOS_s008, sample_of_intersection_classes_LOS_s008, label='s008 LOS')
        plt.bar (intersection_classes_LOS_s008, sample_of_intersection_classes_LOS_s009, label='s009 LOS', alpha=0.5)
        plt.xlabel ('Classes')
        plt.ylabel ('Samples')
        plt.title ('Intersection of Classes for LOS')
        plt.legend ()
        plt.show ()

    if type_data_analysis =='samples':
        return intersection_classes_LOS_s008, sample_of_intersection_classes_LOS_s008, sample_of_intersection_classes_LOS_s009
    elif type_data_analysis =='percentages':
        return intersection_classes_LOS_s009, percentage_of_samples_in_intersection_LOS_s008, percentage_of_samples_in_intersection_LOS_s009
    '''
def shannon_entropy():
    data_LOS, data_NLOS, data_ALL = read_data.read_data_s008(scale_to_coord=16)
    classes_LOS_s008 = data_LOS['index_beams'].value_counts()
    classes_NLOS_s008 = data_NLOS['index_beams'].value_counts()
    classes_ALL_s008 = data_ALL['index_beams'].value_counts()

    samples_LOS_s008 = data_LOS.shape[0]
    samples_NLOS_s008 = data_NLOS.shape[0]
    samples_ALL_s008 = data_ALL.shape[0]


    n_classes = 256
    n_classes_of_s008_LOS = len(classes_LOS_s008)
    #norm_entropy_s008_LOS = st.entropy(classes_LOS_s008)/np.log(len(classes_LOS_s008))
    norm_entropy_s008_LOS = st.entropy (classes_LOS_s008) / np.log (n_classes)

    n_classes_of_s008_NLOS = len(classes_NLOS_s008)
    #norm_entropy_s008_NLOS = st.entropy(classes_NLOS_s008)/np.log(len(classes_NLOS_s008))
    norm_entropy_s008_NLOS = st.entropy (classes_NLOS_s008) / np.log (n_classes)

    n_classes_of_s008_ALL = len(classes_ALL_s008)
    #norm_entropy_s008_ALL = st.entropy(classes_ALL_s008)/np.log(len(classes_ALL_s008))
    norm_entropy_s008_ALL = st.entropy (classes_ALL_s008) / np.log (n_classes)

    plot_histogram_with_entropy (index_beam_all=data_ALL ['index_beams'],
                                 index_beam_LOS=data_LOS ['index_beams'],
                                 index_beam_NLOS=data_NLOS ['index_beams'],
                                 All_entropy=norm_entropy_s008_ALL,
                                 LOS_entropy=norm_entropy_s008_LOS,
                                 NLOS_entropy=norm_entropy_s008_NLOS,
                                 path_to_save='../analyses/divergence_analysis/',
                                 dataset = 's008')

    data_LOS, data_NLOS, data_ALL = read_data.read_data_s009 (scale_to_coord=16)
    classes_LOS_s009 = data_LOS['index_beams'].value_counts()
    classes_NLOS_s009 = data_NLOS['index_beams'].value_counts()
    classes_ALL_s009 = data_ALL['index_beams'].value_counts()

    samples_LOS_s009 = data_LOS.shape[0]
    samples_NLOS_s009 = data_NLOS.shape[0]
    samples_ALL_s009 = data_ALL.shape[0]

    n_classes_of_s009_LOS = len(classes_LOS_s009)
    #norm_entropy_s009_LOS = st.entropy(classes_LOS_s009)/np.log(len(classes_LOS_s009))
    norm_entropy_s009_LOS = st.entropy (classes_LOS_s009) / np.log (n_classes)
    n_classes_of_s009_NLOS = len(classes_NLOS_s009)
    #norm_entropy_s009_NLOS = st.entropy(classes_NLOS_s009)/np.log(len(classes_NLOS_s009))
    norm_entropy_s009_NLOS = st.entropy (classes_NLOS_s009) / np.log (n_classes)
    n_classes_of_s009_ALL = len(classes_ALL_s009)
    #norm_entropy_s009_ALL = st.entropy(classes_ALL_s009)/np.log(len(classes_ALL_s009))
    norm_entropy_s009_ALL = st.entropy(classes_ALL_s009)/np.log(n_classes)

    entropy = {'s008_LOS': norm_entropy_s008_LOS, 's009_LOS': norm_entropy_s009_LOS,
                's008_NLOS': norm_entropy_s008_NLOS, 's009_NLOS': norm_entropy_s009_NLOS,
                's008_ALL': norm_entropy_s008_ALL, 's009_ALL': norm_entropy_s009_ALL}

    plot_histogram_with_entropy(index_beam_all=data_ALL['index_beams'],
                                 index_beam_LOS=data_LOS['index_beams'],
                                 index_beam_NLOS=data_NLOS['index_beams'],
                                 All_entropy=norm_entropy_s009_ALL,
                                 LOS_entropy=norm_entropy_s009_LOS,
                                 NLOS_entropy=norm_entropy_s009_NLOS,
                                 path_to_save='../analyses/divergence_analysis/',
                                dataset = 's009')

    print("---------------------------------------------------------")
    print("|      Shannon Entropy Analysis for s008 and s009 ")
    print("---------------------------------------------------------")
    print("| Dataset \t| connection | n_Class \t| n_samples | Entropy")
    print("---------------------------------------------------------")
    print("|  s008 \t|\t LOS \t|\t", n_classes_of_s008_LOS ,"\t|\t", samples_LOS_s008, "\t| ", np.round(norm_entropy_s008_LOS,3))
    print("|  s009 \t|\t LOS \t|\t", n_classes_of_s009_LOS ,"\t|\t" , samples_LOS_s009, "\t| ", np.round(norm_entropy_s009_LOS,3))
    print("---------------------------------------------------------")
    print("|  s008 \t|\t NLOS \t|\t", n_classes_of_s008_NLOS ,"\t|\t", samples_NLOS_s008, "\t| ", np.round(norm_entropy_s008_NLOS,3))
    print("|  s009 \t|\t NLOS \t|\t", n_classes_of_s009_NLOS ,"\t|\t", samples_NLOS_s009, "\t| ", np.round(norm_entropy_s009_NLOS,3))
    print("---------------------------------------------------------")
    print("|  s008 \t|\t ALL \t|\t", n_classes_of_s008_ALL ,"\t|\t", samples_ALL_s008, "\t| ", np.round(norm_entropy_s008_ALL,3))
    print("|  s009 \t|\t ALL \t|\t", n_classes_of_s009_ALL ,"\t|\t", samples_ALL_s009, "\t| ", np.round(norm_entropy_s009_ALL,3))
    print("---------------------------------------------------------")

    return entropy
def report_of_detalhes_distributions_differences(dataset_s008, dataset_s009, diff_dataset1_less_dataset2):
    class_with_diff_equal_to_zero = diff_dataset1_less_dataset2 [diff_dataset1_less_dataset2.values == 0].keys ()
    print(" Classes with the same number of samples in both datasets :")
    for i in range (len (class_with_diff_equal_to_zero)):
        print (" ", class_with_diff_equal_to_zero [i])

    # Classes with NaN values in the difference
    # This means that the class exists in one dataset but not in the other
    class_with_nan = diff_dataset1_less_dataset2 [np.isnan (diff_dataset1_less_dataset2.values)].keys ()
    print ("Classes not present in one of the datasets (NaN values in the difference):", len (class_with_nan))
    # print(" Classes with NaN values in the difference :", len(class_with_nan))
    print ("\t class | Is in s008 | Is in s009")
    report_classes_in_one_of_datasets = {}
    for i in range (len (class_with_nan)):
        is_in_s008 = dataset_s008 [dataset_s008.keys () == class_with_nan [0]]
        is_in_s009 = dataset_s009 [dataset_s009.keys () == class_with_nan [0]]

        if is_in_s008.empty:
            samples_in_s008 = 0
        else:
            samples_in_s008 = is_in_s008.values [0]
        if is_in_s009.empty:
            samples_in_s009 = 0
        else:
            samples_in_s009 = is_in_s009.values [0]
        print ("\t ", class_with_nan [i], " |\t", samples_in_s008, " |\t", samples_in_s009)
        if i == 0:
            report_classes_in_one_of_datasets.update({'class': [class_with_nan[i]],
                                                      'samples_in_s008': [samples_in_s008],
                                                      'samples_in_s009': [samples_in_s009]})
        else:
            report_classes_in_one_of_datasets['class'].append(class_with_nan[i])
            report_classes_in_one_of_datasets['samples_in_s008'].append(samples_in_s008)
            report_classes_in_one_of_datasets['samples_in_s009'].append(samples_in_s009)
    a=0
    return class_with_nan, class_with_diff_equal_to_zero, report_classes_in_one_of_datasets

def calculate_the_diff_between_distributions():
    type_connection = 'LOS'
    classes_LOS_s009, classes_NLOS_s009, classes_ALL_s009 = read_classes (dataset='s009')
    classes_LOS_s008, classes_NLOS_s008, classes_ALL_s008 = read_classes(dataset='s008')

    '''
    if type_connection == 'LOS':
        class_s008 = classes_LOS_s008
        class_s009 = classes_LOS_s009

    elif type_connection == 'NLOS':
        class_s008 = classes_NLOS_s008
        class_s009 = classes_NLOS_s009

    elif type_connection == 'ALL':
        class_s008 = classes_ALL_s008
        class_s009 = classes_ALL_s009
    '''
    print("---------------------------------------------------------")
    print ("LOS - Relative values - Empirical Differences:")
    distributions_differences_empirical (classes_s009=classes_LOS_s009,
                                         classes_s008=classes_LOS_s008,
                                         type_connection='LOS',
                                         percentual=True)
    print ("LOS - Absolute values - Empirical Differences:")
    distributions_differences_empirical(classes_s009=classes_LOS_s009,
                                        classes_s008=classes_LOS_s008,
                                        type_connection='LOS',
                                        percentual=False)
    print("---------------------------------------------------------")
    print("NLOS - Relative values - Empirical Differences:")
    valores_relativos = distributions_differences_empirical(classes_s009=classes_NLOS_s009,
                                                            classes_s008=classes_NLOS_s008,
                                                            type_connection='NLOS',
                                                            percentual=True)
    print("NLOS - Absolute values - Empirical Differences:")
    valores_absolutos = distributions_differences_empirical(classes_s009=classes_NLOS_s009,
                                                            classes_s008=classes_NLOS_s008,
                                                            type_connection='NLOS',
                                                            percentual=False)
    print("---------------------------------------------------------")
    print("ALL - Relative values - Empirical Differences:")
    distributions_differences_empirical(classes_s009=classes_ALL_s009,
                                        classes_s008=classes_ALL_s008,
                                        type_connection='ALL',
                                        percentual=True)
    print("ALL - Absolute values - Empirical Differences:")
    distributions_differences_empirical(classes_s009=classes_ALL_s009,
                                        classes_s008=classes_ALL_s008,
                                        type_connection='ALL',
                                        percentual=False)
    print("---------------------------------------------------------")


    print ("---------------------------------------------------------")
    print ("LOS - Empirical Differences:")
    print ("                        Dc_max Relativo   Dc_max Absoluto  Dc_I Relativo   Dc_I Absoluto")
    print ("s008 -> s009: Dc_max=", valores_relativos[0], valores_absolutos[0], valores_relativos[1], valores_absolutos[1])

    print("---------------------------------------------------------")
    distributions_differences_empirical(classes_s009=classes_NLOS_s009,
                                        classes_s008=classes_NLOS_s008,
                                        type_connection='NLOS',
                                        percentual=True)
    print("---------------------------------------------------------")
    distributions_differences_empirical(classes_s009=classes_ALL_s009,
                                        classes_s008=classes_ALL_s008,
                                        type_connection='ALL',
                                        percentual=True)

def diff_of_distributions(type_connection='LOS' ):
    classes_LOS_s009, classes_NLOS_s009, classes_ALL_s009 = read_classes (dataset='s009')
    classes_LOS_s008, classes_NLOS_s008, classes_ALL_s008 = read_classes (dataset='s008')

    if type_connection == 'LOS':
        classes_s008 = classes_LOS_s008
        classes_s009 = classes_LOS_s009
    elif type_connection == 'NLOS':
        classes_s008 = classes_NLOS_s008
        classes_s009 = classes_NLOS_s009
    elif type_connection == 'ALL':
        classes_s008 = classes_ALL_s008
        classes_s009 = classes_ALL_s009

    # Suponha que classes_s008 e classes_s009 sejam Series com índices = classes e valores = contagens
    # 1. Calcula os percentuais normalizados
    percentual_s008 = (classes_s008 / classes_s008.sum ()) * 100
    percentual_s009 = (classes_s009 / classes_s009.sum ()) * 100

    # 2. Alinha os índices (classes) nos dois datasets
    all_classes = percentual_s008.index.union (percentual_s009.index)

    P = percentual_s008.reindex (all_classes, fill_value=0)
    Q = percentual_s009.reindex (all_classes, fill_value=0)

    # 3. Calcula as diferenças positivas
    diff_s009_menos_s008 = (Q - P).clip (lower=0)
    diff_s008_menos_s009 = (P - Q).clip (lower=0)

    # 4. Preenche NaNs com os percentuais originais (opcional, pois já garantimos 0 nas faltantes)
    p = diff_s008_menos_s009.fillna (P)
    q = diff_s009_menos_s008.fillna (Q)

    # 5. Soma total das diferenças
    soma_p = p.sum ()
    soma_q = q.sum ()

    D_Cmax_P = np.sum (np.maximum (P - Q, 0))
    D_Cmax_Q = np.sum (np.maximum (Q - P, 0))

    # 6. Função indicadora: onde P > Q
    I = np.where (P > Q, 1, 0)
    Dc_I = np.sum (I)

    # 6. Função indicadora: onde Q > P
    I_Q = np.where (P < Q, 1, 0)
    Dc_I_Q = np.sum (I_Q)

    # cria um grafico de p e q
    import matplotlib.pyplot as plt
    plt.figure (figsize=(10, 6))
    plt.bar (p.index, p.values, label='P (s008)', alpha=0.5)
    plt.bar (q.index, q.values, label='Q (s009)', alpha=0.5)
    plt.xlabel('Classes')
    plt.ylabel('Percentual')
    plt.title('Diferenças Percentuais entre Classes s008 e s009 ' + type_connection + ' Connection')
    #plt.xticks(p.index, rotation=45, fontsize=8)
    plt.legend()
    plt.text(0.5, 8, f'D_Cmax P -> Q: {D_Cmax_P:.2f}\nD_Cmax Q -> P: {D_Cmax_Q:.2f}\nDc_I P->Q: {Dc_I}\nDc_I Q->P: {Dc_I_Q}',)
    plt.grid (True, linestyle='--', alpha=0.5)

    plt.show ()

    # 7. Resultados
    print (f"Total diferença P > Q (soma_p): {soma_p:.2f}")
    print (f"Total diferença Q > P (soma_q): {soma_q:.2f}")
    print (f"Número de classes em que P > Q (D_c^I): {Dc_I}")
    print (f"Número de classes em que Q > P (D_c^I_Q): {Dc_I_Q}")
    print (f"D_Cmax P -> Q: {D_Cmax_P:.2f}")
    print (f"D_Cmax Q -> P: {D_Cmax_Q:.2f}")

def distributions_differences_empirical(classes_s009, classes_s008, percentual, type_connection):
    #percentual = True
    if percentual:
        percentual_s008 = (classes_s008 / classes_s008.sum ()) * 100
        percentual_s009 = (classes_s009 / classes_s009.sum ()) * 100
        diff_classes_s009_less_s008 = percentual_s009 - percentual_s008
        diff_classes_s008_less_s009 = percentual_s008 - percentual_s009

        # eliminating negative values
        diff_classes_s009_less_s008[diff_classes_s009_less_s008 < 0] = 0
        diff_classes_s008_less_s009[diff_classes_s008_less_s009 < 0] = 0

        # fillna with percentual_s008 and percentual_s009
        p = diff_classes_s008_less_s009.fillna(percentual_s008)
        q = diff_classes_s009_less_s008.fillna(percentual_s009)
        p.fillna (percentual_s008)
        q.fillna (percentual_s009)

        Dc_max_s008_s009 = Dc (p, q)
        Dc_max_s008_s009_ = np.sum (np.maximum (p - q, 0))
        Dc_max_s009_s008 = Dc (q, p)
        Dc_max_s009_s008_ = np.sum (np.maximum (q - p, 0))


        index_diff_classes_s008_less_s009_nan = diff_classes_s008_less_s009[diff_classes_s008_less_s009.isnull()==True]
        index_diff_classes_s009_less_s008_nan = diff_classes_s009_less_s008[diff_classes_s009_less_s008.isnull()==True]
        info_diff_classes_s009_less_s008_nan_with_values = index_diff_classes_s008_less_s009_nan.fillna (percentual_s009)
        info_diff_classes_s008_less_s009_nan_with_values = index_diff_classes_s009_less_s008_nan.fillna (percentual_s008)

        p_prima = diff_classes_s008_less_s009.fillna (info_diff_classes_s008_less_s009_nan_with_values)
        q_prima = diff_classes_s009_less_s008.fillna (info_diff_classes_s009_less_s008_nan_with_values)

        diff_classes_s009_less_s008 [diff_classes_s009_less_s008 < 0] = 0
        diff_classes_s008_less_s009 [diff_classes_s008_less_s009 < 0] = 0

        p = diff_classes_s008_less_s009.fillna (percentual_s008)
        q = diff_classes_s009_less_s008.fillna (percentual_s009)
        q = q.fillna (0)
        p = p.fillna (0)
        Dc_max_s008_s009 = Dc (p, q)
        Dc_max_s008_s009_ = np.sum (np.maximum (p - q, 0))
        Dc_max_s009_s008 = Dc (q, p)
        Dc_max_s009_s008_ = np.sum (np.maximum (q - p, 0))

        Dc_I_max_s008_s009 = Dc_I (p, q)
        Dc_I_max_s009_s008 = Dc_I (q, p)
        a=0

    else:
        diff_classes_s009_less_s008 = classes_s009 - classes_s008
        diff_classes_s008_less_s009 = classes_s008 - classes_s009

        diff_classes_s009_less_s008[diff_classes_s009_less_s008 < 0] = 0
        diff_classes_s008_less_s009[diff_classes_s008_less_s009 < 0] = 0

        p = diff_classes_s008_less_s009.fillna(classes_s008)
        q = diff_classes_s009_less_s008.fillna(classes_s009)
        q = q.fillna(0)
        p = p.fillna(0)



        Dc_max_s008_s009 = Dc(p, q)
        Dc_max_s009_s008 = Dc(q, p)

        Dc_I_max_s008_s009 = Dc_I(p, q)
        Dc_I_max_s009_s008 = Dc_I(q, p)

    print ("Empirical Differences:")
    print (type_connection, " s008 -> s009: Dc_max=", np.round(Dc_max_s008_s009,3), " Dc_I=", np.round(Dc_I_max_s008_s009,3))
    print (type_connection, " s009 -> s008: Dc_max=", np.round(Dc_max_s009_s008,3), " Dc_I=", np.round(Dc_I_max_s009_s008,3))


    return Dc_max_s008_s009, Dc_I_max_s008_s009, Dc_max_s009_s008, Dc_I_max_s009_s008



    '''

    classes_with_nan_in_diff_s008_less_s009 = diff_classes_s008_less_s009[np.isnan(diff_classes_s008_less_s009.values)].index.tolist()
    #get index of nan_class_in_s008
    #nan_class_in_s008 = nan_class_in_s008.index.tolist()
    #get values of classes_with_nan_in_diff_s008_less_s009 in classes_s008

    classes_in_s008_only = []
    samples_in_s008_only = []
    classes_in_s009_only = []
    samples_in_s009_only = []
    classes_and_samples_not_in_s008 = []
    for i in range(len(classes_with_nan_in_diff_s008_less_s009)):
        if classes_with_nan_in_diff_s008_less_s009[i] in classes_s008.index:
            classes_in_s008_only.append(classes_with_nan_in_diff_s008_less_s009[i])
            samples_in_s008_only.append(classes_s008[classes_with_nan_in_diff_s008_less_s009[i]])

        else:
            classes_and_samples_not_in_s008.append(classes_with_nan_in_diff_s008_less_s009[i])
            classes_in_s009_only.append(classes_with_nan_in_diff_s008_less_s009[i])
            samples_in_s009_only.append(classes_s009[classes_with_nan_in_diff_s008_less_s009[i]])

    classes_and_samples_in_s008_only = {'classes': classes_in_s008_only, 'samples': samples_in_s008_only}
    classes_and_samples_in_s009_only = {'classes': classes_in_s009_only, 'samples': samples_in_s009_only}

    #-----



    diff_s008_less_s009_with_s008_data = diff_classes_s008_less_s009.fillna(classes_s008)
    diff_s008_less_s009_with_s008_data[np.isnan(diff_s008_less_s009_with_s008_data.values)] = 0
    p = diff_s008_less_s009_with_s008_data

    diff_s008_less_s009_with_s009_data = diff_classes_s008_less_s009.fillna (classes_s009)
    diff_s008_less_s009_with_s009_data[np.isnan(diff_s008_less_s009_with_s009_data.values)] = 0
    q= diff_s008_less_s009_with_s009_data

    Dc_max_s008_s009 = Dc(p, q)
    Dc_max_s009_s008 = Dc(q, p)

    Dc_I_max_s008_s009 = Dc_I(p, q)
    Dc_I_max_s009_s008 = Dc_I(q, p)

    print ("Empirical Differences:")
    print (type_connection, " s009 -> s008: Dc_max=", Dc_max_s009_s008, " Dc_I=", Dc_I_max_s009_s008)
    print (type_connection, " s008 -> s009: Dc_max=", Dc_max_s008_s009, " Dc_I=", Dc_I_max_s008_s009)

    #diff_s008_less_s009_with_s009_data = diff_classes_s008_less_s009.fillna(classes_s009)
    #diff_s008_less_s009_with_s009_data[np.isnan(diff_s008_less_s009_with_s009_data.values)] = 0

    #diff_classes_s009_less_s008_with_s009_data = diff_classes_s009_less_s008.fillna(classes_s009)
    #diff_classes_s009_less_s008_with_s009_data[np.isnan(diff_classes_s009_less_s008_with_s009_data.values)] = 0
    # ---------------------------------------
    #Dc_max_s008_s009 = Dc (diff_s008_less_s009_with_s008_data, diff_classes_s009_less_s008_with_s009_data)
    #Dc_max_s009_s008 = Dc (diff_classes_s009_less_s008_with_s009_data, diff_s008_less_s009_with_s008_data)
    # ---------------------------------------
    #Dc_I_max_s008_s009 = Dc_I (diff_s008_less_s009_with_s008_data, diff_classes_s009_less_s008_with_s009_data)
    #Dc_I_max_s009_s008 = Dc_I (diff_classes_s009_less_s008_with_s009_data, diff_s008_less_s009_with_s008_data)
    # ---------------------------------------
    #print ("Empirical Differences:")
    #print (type_connection, " s009 -> s008: Dc_max=", Dc_max_s009_s008, " Dc_I=", Dc_I_max_s009_s008)
    #print (type_connection, " s008 -> s009: Dc_max=", Dc_max_s008_s009, " Dc_I=", Dc_I_max_s008_s009)
    '''
    '''
    show_details_diff = False
    if show_details_diff:
        report_of_detalhes_distributions_differences(classes_s008, classes_s009, diff_classes_s009_less_s008)

    #Assming as zero the classes are not in both datasets (nan)
    #assuming_diff_LOS_s009_less_s008 = diff_LOS_s009_less_s008.copy()
    #assuming_diff_LOS_s009_less_s008[np.isnan(assuming_diff_LOS_s009_less_s008.values)] = 0
    #assuming_diff_LOS_s008_less_s009 = diff_LOS_s008_less_s009.copy ()
    #assuming_diff_LOS_s008_less_s009 [np.isnan (assuming_diff_LOS_s008_less_s009.values)] = 0










    assuming_diff_s009_less_s008 = diff_classes_s009_less_s008.copy()
    assuming_diff_s009_less_s008[np.isnan(assuming_diff_s009_less_s008.values)] = 0
    for i in range(len(assuming_diff_s009_less_s008)):
        if assuming_diff_s009_less_s008.index [i] == 'nan' and assuming_diff_s009_less_s008.index [i] in classes_and_samples_in_s009_only:
            assuming_diff_s009_less_s008[i] = classes_and_samples_in_s009_only['samples'][classes_and_samples_in_s009_only['classes'].index(assuming_diff_s009_less_s008.index[i])]

    #trazer as quantidades das classes que nao estao no outro dataset
    #change the nan values by values in samples in s008 and s009


    assuming_diff_s008_less_s009 = diff_classes_s008_less_s009.copy()
    assuming_diff_s008_less_s009[np.isnan(assuming_diff_s008_less_s009.values)] = classes_and_samples_in_s008_only



    #---------------------------------------
    Dc_max_s009_s008 = Dc(assuming_diff_s009_less_s008, assuming_diff_s008_less_s009)
    Dc_max_s008_s009 = Dc(assuming_diff_s008_less_s009, assuming_diff_s009_less_s008)
    #---------------------------------------
    Dc_I_max_s009_s008 = Dc_I(assuming_diff_s009_less_s008, assuming_diff_s008_less_s009)
    Dc_I_max_s008_s009 = Dc_I(assuming_diff_s008_less_s009, assuming_diff_s009_less_s008)
    #---------------------------------------
    print("Empirical Differences:")
    print(type_connection, " s009 -> s008: Dc_max=", Dc_max_s009_s008, " Dc_I=", Dc_I_max_s009_s008)
    print(type_connection, " s008 -> s009: Dc_max=", Dc_max_s008_s009, " Dc_I=", Dc_I_max_s008_s009)
    '''


def Dc_I(P,Q):
    # Function indicating the difference between two distributions
    # This function calculates the difference between two distributions
    """
    Calculates the indicator function of difference between two distributions P and Q.
    The function returns 1 if P is greater than Q, 0 if P is less than Q.
    If P and Q are equal, the function returns 0.
    1. If P > Q, returns 1.
    2. If P < Q, returns 0.
    3. If P == Q, returns 0.
    4. sums the absolute differences between P and Q.
    """

    # Check if P and Q are numpy arrays, if not convert them
    if not isinstance(P, np.ndarray):
        P = np.array(P)
    if not isinstance(Q, np.ndarray):
        Q = np.array(Q)
    # Check if P and Q are 1D arrays, if not flatten them
    if P.ndim != 1:
        P = P.flatten()
    if Q.ndim != 1:
        Q = Q.flatten()
    # Check if P and Q have the same length
    if P.shape[0] != Q.shape[0]:
        raise ValueError("P and Q must have the same length.")

    # Create an array of the indicator function
    I = np.where(P > Q, 1, 0)
    Dc_I = np.sum(I)
    # Return the sum of the indicator function
    return Dc_I

def Dc(P, Q):
    # Function indicating the difference between two distributions
    # This function calculates the difference between two distributions
    """
    Calculates the difference between two distributions P and Q.
    """
    diff = np.sum(np.maximum(P - Q, 0))
    return diff





def distributions_differences():
    #statistical-distance
    # A python module with functions to calculate distance / dissimilarity
    # measures between two probability density functions (pdfs).
    # The module can be used to compare points in vector spaces.

    import distance
    #classes_LOS_s009, classes_NLOS_s009, classes_ALL_s009 = read_classes (dataset='s009')
    #classes_LOS_s008, classes_NLOS_s008, classes_ALL_s008 = read_classes (dataset='s008')

    # Calculate the differences between the distributions of classes
    a = np.array([1,2,3,4])
    b = np.array([2,3,4,5])
    c = distance.euclidean(a, b)

    a=0

def plot_radar():
    import matplotlib.pyplot as plt
    import plotly.express as px

    labels = ['DC_max', 'DC_I', 'DC_max_%', 'DC_I_%']

    all_vals = [3463, 96, 0.239, 95]
    los_vals = [5041, 75, 0.349, 44]
    nlos_vals = [1278, 85, 0.408, 107]

    # Normalizando para radar plot (valores em uma escala comum)
    def normalize(vals):
        max_vals = [max ([all_vals [i], los_vals [i], nlos_vals [i]]) for i in range (len (vals))]
        b= [v / max_v if max_v != 0 else 0 for v, max_v in zip (vals, max_vals)]
        return b

    all_norm = normalize (all_vals)
    los_norm = normalize (los_vals)
    nlos_norm = normalize (nlos_vals)

    df = pd.DataFrame (dict (
        r=[1, 5, 2, 2, 3],
        theta=['processing cost', 'mechanical properties', 'chemical stability',
               'thermal stability', 'device integration']))
    df = pd.DataFrame (dict (
        r = all_norm,
        theta = labels))

    df_1 = pd.DataFrame (dict ( r= los_norm, theta=labels))
    df_2 = pd.DataFrame (dict ( r= nlos_norm, theta=labels))


    fig = px.line_polar (df, r='r', theta='theta', line_close=True)
    fig.update_traces (line_color='blue', fillcolor='rgba(0, 0, 255, 0.2)', name='ALL', fill='toself')
    fig.add_scatterpolar (r=df_1['r'], theta=df_1['theta'], fill='toself', name='LOS')
    fig.add_scatterpolar (r=df_2['r'], theta=df_2['theta'], fill='toself', name='NLOS')

    fig.show ()


    a=0



    # Setup do radar chart
    labels.append (labels [0])  # loop para fechar o gráfico
    angles = np.linspace (0, 2 * np.pi, len (labels), endpoint=False).tolist ()
    angles += angles [:1]

    all_norm += all_norm [:1]
    los_norm += los_norm [:1]
    nlos_norm += nlos_norm [:1]

    # Plot
    fig, ax = plt.subplots (figsize=(8, 6), subplot_kw=dict (polar=True))
    ax.plot (angles, all_norm, label='ALL', linewidth=2, linestyle='-', marker='o')
    ax.fill (angles, all_norm, alpha=0.15)

    ax.plot (angles, los_norm, label='LOS', linewidth=2, linestyle='-', marker='s')
    ax.fill (angles, los_norm, alpha=0.15)

    ax.plot (angles, nlos_norm, label='NLOS', linewidth=2, linestyle='-', marker='^')
    ax.fill (angles, nlos_norm, alpha=0.15)

    ax.set_title ("Normalized Divergence Metrics per Connection Type", size=14, pad=20)
    ax.set_xticks (angles [:-1])
    ax.set_xticklabels (labels [:-1])
    ax.set_yticklabels ([])
    ax.legend (loc='upper right', bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout ()
    plt.show ()
    a=0


def plot_relation_between_accuracy_and_divergence():
    import matplotlib.pyplot as plt

    accuracy = {
        "ALL": [58, 53],
        "LOS": [65, 25],
        "NLOS": [50, 20]
    }

    ration_DC_max_DC_I = {
        "ALL": [36.07, 34.05],
        "LOS": [67.21, 4],
        "NLOS": [15.03, 72.78]
    }

    labels_acc = ['ALL', 'LOS', 'NLOS']
    x = np.arange (len (labels_acc))
    width = 0.35
    plt.rcParams ['font.family'] = 'Times New Roman'
    plt.rcParams ['font.size'] = 14
    ax = plt.subplot (1, 1, 1)

    # Barras de acurácia
    #ax.grid (True, linestyle='--', alpha=0.5)
    acc_bars1 = ax.bar (x - width / 2, [accuracy [s] [0] for s in labels_acc], width,
                        label='Accuracy (s008→s009)',
                        color='darkgray')#, alpha=0.4)
    acc_bars2 = ax.bar (x + width / 2, [accuracy [s] [1] for s in labels_acc], width,
                        label='Accuracy (s009→s008)',
                         color='teal', alpha=0.4)
    ax.set_ylabel ('Top-1 Accuracy (%)')
    ax.set_ylim (0, 100)
    ax.set_xticks (x)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 1), fontsize=10, frameon=False, ncol=2)
    ax.set_xticklabels (labels_acc)
    ax.spines ['top'].set_visible (False)

    #ax.set_title ("Accuracy vs DC_max")

    # Linha de DC_max
    ax1 = ax.twinx ()
    dc_bars1 = ax1.plot (x, [ration_DC_max_DC_I [s] [0] for s in labels_acc], label='DRC (s008→s009)', color='orangered', marker='o', linestyle='--')
    dc_bars2 = ax1.plot (x, [ration_DC_max_DC_I [s] [1] for s in labels_acc], label='DCR (s009→s008)', color='firebrick', marker='s')
    ax1.set_ylabel (r'$DCR=\frac{DC_{max}}{DC_{I}}$', fontsize=14)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 1.05), fontsize=10, frameon=False, ncol=2)
    #ax1.set_ylim (0, 6000)
    ax1.spines ['top'].set_visible (False)

    plt.savefig ('../analyses/divergence_analysis/accuracy_vs_divergence.png', dpi=300, bbox_inches='tight')

    a=0



def plot_radar_and_metrics():
    import matplotlib.pyplot as plt
    import numpy as np

    # === DADOS ===

    # Métricas DC por cenário
    all_vals = [3463, 96, 0.239, 95]
    los_vals = [5041, 75, 0.349, 44]
    nlos_vals = [1278, 85, 0.408, 107]

    # Acurácia top-1 WiSARD: [s8 → s9, s9 → s8]
    accuracy = {
        "ALL": [58, 53],
        "LOS": [65, 25],
        "NLOS": [50, 20]
    }

    # DC_max absolutos
    dc_max = {
        "ALL": [3463, 1907],
        "LOS": [5041, 32],
        "NLOS": [1278, 4731]
    }

    # === FUNÇÕES AUXILIARES ===

    def normalize_and_close(vals, all_vals, los_vals, nlos_vals):
        max_vals = [max ([all_vals [i], los_vals [i], nlos_vals [i]]) for i in range (len (vals))]
        norm = [v / max_v if max_v != 0 else 0 for v, max_v in zip (vals, max_vals)]
        return norm + norm [:1]

    # === PLOT ===

    fig = plt.figure (figsize=(18, 6))

    # --- 1. Radar das métricas de divergência ---
    labels = ['DC_max', 'DC_I', 'DC_max_%', 'DC_I_%']
    angles_metrics = np.linspace (0, 2 * np.pi, len (labels), endpoint=False).tolist ()
    angles_metrics += angles_metrics [:1]

    ax1 = plt.subplot (1, 3, 1, polar=True)
    all_norm = normalize_and_close (all_vals, all_vals, los_vals, nlos_vals)
    los_norm = normalize_and_close (los_vals, all_vals, los_vals, nlos_vals)
    nlos_norm = normalize_and_close (nlos_vals, all_vals, los_vals, nlos_vals)

    ax1.plot (angles_metrics, all_norm, label='ALL', marker='o')
    ax1.fill (angles_metrics, all_norm, alpha=0.15)
    ax1.plot (angles_metrics, los_norm, label='LOS', marker='s')
    ax1.fill (angles_metrics, los_norm, alpha=0.15)
    ax1.plot (angles_metrics, nlos_norm, label='NLOS', marker='^')
    ax1.fill (angles_metrics, nlos_norm, alpha=0.15)

    ax1.set_title ("Divergence Metrics per Connection", size=13)
    ax1.set_xticks (angles_metrics [:-1])
    ax1.set_xticklabels (labels)
    ax1.set_yticklabels ([])
    ax1.legend (loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # --- 2. Radar de acurácia top-1 ---
    labels_acc = ['ALL', 'LOS', 'NLOS']
    angles_acc = np.linspace (0, 2 * np.pi, len (labels_acc), endpoint=False).tolist ()
    angles_acc += angles_acc [:1]
    acc_s8_s9 = [accuracy [s] [0] for s in labels_acc] + [accuracy [labels_acc [0]] [0]]
    acc_s9_s8 = [accuracy [s] [1] for s in labels_acc] + [accuracy [labels_acc [0]] [1]]

    ax2 = plt.subplot (1, 3, 2, polar=True)
    ax2.plot (angles_acc, acc_s8_s9, label='s8→s9', color='teal', marker='o')
    ax2.fill (angles_acc, acc_s8_s9, alpha=0.15, color='teal')
    ax2.plot (angles_acc, acc_s9_s8, label='s9→s8', color='darkcyan', marker='s')
    ax2.fill (angles_acc, acc_s9_s8, alpha=0.15, color='darkcyan')

    ax2.set_title ("Top-1 Accuracy per Connection", size=13)
    ax2.set_xticks (angles_acc [:-1])
    ax2.set_xticklabels (labels_acc)
    ax2.set_yticks ([20, 40, 60, 80])
    ax2.set_yticklabels (['20%', '40%', '60%', '80%'])
    ax2.legend (loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # --- 3. Barras de acurácia + linha de DC_max ---
    x = np.arange (len (labels_acc))
    width = 0.35
    ax3 = plt.subplot (1, 3, 3)

    # Barras de acurácia
    acc_bars1 = ax3.bar (x - width / 2, [accuracy [s] [0] for s in labels_acc], width, label='Accuracy (s8→s9)',
                         color='skyblue')
    acc_bars2 = ax3.bar (x + width / 2, [accuracy [s] [1] for s in labels_acc], width, label='Accuracy (s9→s8)',
                         color='dodgerblue')
    ax3.set_ylabel ('Top-1 Accuracy (%)')
    ax3.set_ylim (0, 100)
    ax3.set_xticks (x)
    ax3.set_xticklabels (labels_acc)
    ax3.set_title ("Accuracy vs DC_max")

    # Linha de DC_max
    ax4 = ax3.twinx ()
    dc_bars1 = ax4.plot (x, [dc_max [s] [0] for s in labels_acc], label='DC_max (s8→s9)', color='orangered', marker='o')
    dc_bars2 = ax4.plot (x, [dc_max [s] [1] for s in labels_acc], label='DC_max (s9→s8)', color='firebrick', marker='s')
    ax4.set_ylabel ('DC_max (absolute)')
    ax4.set_ylim (0, 6000)

    # Legenda
    lines_labels = [*zip (dc_bars1, ['DC_max (s8→s9)']), *zip (dc_bars2, ['DC_max (s9→s8)'])]
    lines, labels_lines = zip (*lines_labels)
    ax4.legend (lines + (acc_bars1, acc_bars2), labels_lines + ('Accuracy (s8→s9)', 'Accuracy (s9→s8)'),
                loc='upper right')

    plt.suptitle ("WiSARD Model Behavior vs Distribution Divergence", fontsize=16)
    plt.tight_layout ()
    plt.show ()

def venn_diagram(connection_type):
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2, venn3, venn3_circles, venn2_circles



    #connection_type = 'LOS'  # 'LOS', 'NLOS', 'ALL'

    classes_LOS_s009, classes_NLOS_s009, classes_ALL_s009 = read_classes (dataset='s009')
    classes_LOS_s008, classes_NLOS_s008, classes_ALL_s008 = read_classes (dataset='s008')

    if connection_type == 'ALL':
        set1 = set(classes_ALL_s008.index.tolist())
        set2 = set(classes_ALL_s009.index.tolist())
    elif connection_type == 'LOS':
        set1 = set(classes_LOS_s008.index.tolist())
        set2 = set(classes_LOS_s009.index.tolist())
    elif connection_type == 'NLOS':
        set1 = set(classes_NLOS_s008.index.tolist())
        set2 = set(classes_NLOS_s009.index.tolist())


    font_size = 16
    v =venn2 ((set1, set2), set_labels=(' ', ' '),
             #set_colors=('mistyrose', 'lightblue'), alpha=0.9)
            set_colors = ('rebeccapurple', 'lightblue'), alpha = 0.7)
    for text in v.set_labels:
        text.set_fontsize (15)
    venn2_circles((set1, set2),  linewidth=1.5, color='grey', alpha=0.5)
    for x in range (len (v.subset_labels)):
        if v.subset_labels [x] is not None:
            v.subset_labels [x].set_fontsize (15)

    plt.annotate('s008 set', xy=[-0.68, 0.45], #xytext=(-70, -70),
                  ha='center', textcoords='offset points', fontsize=font_size,
                 bbox=dict(boxstyle='round,pad=0.5', fc='rebeccapurple', alpha=0.4))
                  #arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='gray'))
    plt.annotate (str (len (set1)) + ' classes', xy=[-0.68, 0.3], ha='center', fontsize=font_size)

    plt.annotate('s009 set', xy=[0.652, 0.43], #xytext=(-70, -70),
                  ha='center', textcoords='offset points', fontsize=font_size,
                 bbox=dict (boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7))
    plt.annotate(str(len(set2)) + ' classes', xy=[0.65, 0.3], fontsize=font_size, ha='center')#, textcoords='offset points')

    path_to_save=  '../analyses/divergence_analysis/'
    plt.savefig(path_to_save+'venn_diagram_' + connection_type + '.png', bbox_inches='tight', dpi=300)

    common_elements = list (set1.intersection (set2))

    diff_elements_s008_less_s009 = list (set1 - set2)
    diff_elements_s009_less_s008 = list (set2 - set1)

    print(" set |\t connection |\t n.class |\t commum_class |\t diff_class")
    print("s008 |\t    "+connection_type+"     |\t  ", len (set1), "  |\t\t", len (common_elements), " \t |\t", len (diff_elements_s008_less_s009))
    print("s009 |\t    "+connection_type+"      |\t  ", len (set2), "  |\t\t", len (common_elements), " \t |\t", len (diff_elements_s009_less_s008))
    print("------------------------------------------------")





#plot_radar()
venn_diagram('NLOS')
#venn_diagram('ALL')
#calculate_the_diff_between_distributions()
#diff_of_distributions()


#distributions_differences_empirical()
#intersection_classes_LOS_s008, sample_of_intersection_classes_LOS_s008, sample_of_intersection_classes_LOS_s009 = classes_intersection()
#a=0


#classes_intersection()

#plot_relation_between_accuracy_and_divergence()
#shannon_entropy()