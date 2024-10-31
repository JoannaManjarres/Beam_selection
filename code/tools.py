import numpy as np

def calculate_mean_score(data):
    #all_score = data ['Score'].tolist ()
    average_score = []
    for i in range(len(data)):
        i = i + 1
        average_score.append(np.mean(data[0:i]))
    return average_score