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

def classes_intersection():
    classes_LOS_s009, classes_NLOS_s009, classes_ALL_s009 = read_classes (dataset='s009')
    classes_LOS_s008, classes_NLOS_s008, classes_ALL_s008 = read_classes (dataset='s008')

    """
    Calculate the intersection of classes between two datasets.
    """
    # Calculate the intersection of classes
    intersection_LOS = set (classes_LOS_s008.index).intersection (set (classes_LOS_s009.index))
    intersection_NLOS = set (classes_NLOS_s008.index).intersection (set (classes_NLOS_s009.index))
    intersection_ALL = set (classes_ALL_s008.index).intersection (set (classes_ALL_s009.index))
    print ("Intersection of classes:")
    print ("LOS:", len (intersection_LOS))
    print ("NLOS:", len (intersection_NLOS))
    print ("ALL:", len (intersection_ALL))

    # calculate the number of samples in the intersection for each dataset
    samples_intersection_LOS_in_s008 = classes_LOS_s008 [classes_LOS_s008.index.isin (intersection_LOS)]
    samples_intersection_LOS_in_s009 = classes_LOS_s009 [classes_LOS_s009.index.isin (intersection_LOS)]
    samples_intersection_NLOS = sum (classes_NLOS_s008 [classes_NLOS_s008.index.isin (intersection_NLOS)]) + sum (
        classes_NLOS_s009 [classes_NLOS_s009.index.isin (intersection_NLOS)])
    samples_intersection_ALL = sum (classes_ALL_s008 [classes_ALL_s008.index.isin (intersection_ALL)]) + sum (
        classes_ALL_s009 [classes_ALL_s009.index.isin (intersection_ALL)])
    # print("Samples in intersection:")

    intersection_classes_LOS_s008 = samples_intersection_LOS_in_s008.index.tolist ()
    sample_of_intersection_classes_LOS_s008 = samples_intersection_LOS_in_s008.values.tolist ()

    intersection_classes_LOS_s009 = samples_intersection_LOS_in_s009.index.tolist ()
    sample_of_intersection_classes_LOS_s009 = samples_intersection_LOS_in_s009.values.tolist ()

    plot_distribution = True
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

def calculate_the_diff_between_distributions():
    classes_LOS_s009, classes_NLOS_s009, classes_ALL_s009 = read_classes (dataset='s009')
    classes_LOS_s008, classes_NLOS_s008, classes_ALL_s008 = read_classes(dataset='s008')

    distributions_differences_empirical(classes_s009=classes_LOS_s009,
                                        classes_s008=classes_LOS_s008,
                                        type_connection='LOS')
    print("---------------------------------------------------------")
    distributions_differences_empirical(classes_s009=classes_NLOS_s009,
                                        classes_s008=classes_NLOS_s008,
                                        type_connection='NLOS')
    print("---------------------------------------------------------")
    distributions_differences_empirical(classes_s009=classes_ALL_s009,
                                        classes_s008=classes_ALL_s008,
                                        type_connection='ALL')

def distributions_differences_empirical(classes_s009, classes_s008, type_connection):
    # This function calculates the empirical differences between distributions
    # of classes in the datasets s008 and s009.
    # It uses the read_classes function to get the class distributions.

    #classes_LOS_s009, classes_NLOS_s009, classes_ALL_s009 = read_classes(dataset='s009')
    #classes_LOS_s008, classes_NLOS_s008, classes_ALL_s008 = read_classes(dataset='s008')

    #diff_NLOS = classes_NLOS_s009 - classes_NLOS_s008
    #diff_ALL = classes_ALL_s009 - classes_ALL_s008

    # Calculate the empirical differences between the distributions of classes
    #diff_LOS_s009_less_s008 = classes_LOS_s009 - classes_LOS_s008
    #diff_LOS_s008_less_s009 = classes_LOS_s008 - classes_LOS_s009


    # Calculate the empirical differences between the distributions of classes
    diff_classes_s009_less_s008 = classes_s009 - classes_s008
    diff_classes_s008_less_s009 = classes_s008 - classes_s009



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

    assuming_diff_s008_less_s009 = diff_classes_s008_less_s009.copy()
    assuming_diff_s008_less_s009[np.isnan(assuming_diff_s008_less_s009.values)] = 0



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


calculate_the_diff_between_distributions()
#distributions_differences_empirical()
#classes_intersection = classes_intersection()




#shannon_entropy()