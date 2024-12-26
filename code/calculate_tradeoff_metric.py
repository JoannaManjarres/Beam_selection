import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def read_results_sliding(input_type, ref, window_type='sliding_window'):
    # Read the data
    path ='../results/score/'+ref+'/servidor_land/online/'+input_type+'/'+window_type+'/'

    print('0_all_results_sliding_window_100_top_k')
    window_size=[100, 500, 1000, 1500, 2000]
    all_mean_accuracy_top_1_of_windows = []
    all_mean_time_train_of_windows = []
    for i in range (len (window_size)):
        filename = 'all_results_'+window_type+'_'+str(window_size[i])+'_top_k.csv'
        if ref =='Wisard':
            filename = '0_all_results_'+window_type+'_'+str(window_size[i])+'_top_k.csv'
        data = pd.read_csv(path+filename)
        score_top_1 = data [data ['top-k'] == 1]
        mean = score_top_1 ['score'].mean ()
        time_train = score_top_1 ['trainning_process_time']* 1e-9
        time_train = time_train.mean ()
        all_mean_accuracy_top_1_of_windows.append (mean)
        all_mean_time_train_of_windows.append (time_train)

    df = pd.DataFrame({'window_size': window_size, 'accuracy': all_mean_accuracy_top_1_of_windows, 'time_train': all_mean_time_train_of_windows})


    return df


def calculate_adjusted_efficienc_Metric( data_results):

    # Métrica de Eficiência Ajustada (MEA)
    # Uma métrica que pondera a acurácia pela inversa do tempo de treinamento,
    # para favorecer modelos que alcançam alta acurácia em menos tempo.
    # MEA = Acurácia média / log(1 + Tempo médio de treinamento)

    window_size = data_results['window_size']
    MEA_of_windows = []
    for i in range(len(window_size)):
        accuracy = data_results['accuracy'][i]
        time_train = data_results['time_train'][i]
        MEA = accuracy /np.log(1+time_train)
        MEA_of_windows.append(MEA)

    print(window_size)
    print(MEA_of_windows)

def calculate_time_penalty_function(data_results):
    # métrica que penaliza a acurácia pelo tempo de treinamento.
    # metrica = accuracy - alpha*tempo de treinamento
    # alpha é um hiperparâmetro que controla o peso do tempo de treinamento em relação à acurácia.
    # Quanto maior o valor de alpha, mais a acurácia é penalizada.
    # Isso pode ser ajustado conforme sua aplicação (ex.: quanto o tempo importa em relação à acurácia).

    window_size = data_results['window_size']
    weight_acc = 0.6
    weight_time = 0.4
    metric_of_windows = []
    min_accuracy = min(data_results['accuracy'])
    max_time_train = max(data_results['time_train'])
    accuracy_norm = data_results['accuracy'].apply(lambda x: (x - min_accuracy)/(max(data_results['accuracy']) - min_accuracy))
    time_norm = data_results['time_train'].apply(lambda x: (x - min(data_results['time_train']))/(max(data_results['time_train']) - min(data_results['time_train'])))
    for i in range(len(window_size)):
        metric = accuracy_norm[i]*weight_acc - time_norm[i]*weight_time
        metric_of_windows.append(metric)

    print('Acuracias: ', data_results['accuracy'])
    print('tempo: ', data_results['time_train'] )
    print(metric_of_windows)

    #plt.plot( data_results['accuracy'], data_results['time_train'], 'o-')
    fig, ax = plt.subplots ()
    scat = ax.scatter (x=data_results['accuracy'], y=data_results['time_train'], c=window_size, s=200, marker='o')
    fig.colorbar (scat)
    for i in range (len (window_size)):
        ax.text (data_results['accuracy'][i], data_results['time_train'][i]+0.01, str (np.round(metric_of_windows[i],3)))
    plt.xlabel ('accuracy')
    plt.yticks(data_results['time_train'])
    plt.title('Metrica de tradeoff para diferentes janelas')
    plt.ylabel ('time_train')
    plt.grid()


def calculate_tradeoff_norm(data_results):
    window_size = data_results ['window_size']

    metric_of_windows = []
    for i in range (len (window_size)):
        accuracy = data_results ['accuracy'] [i]
        time_train = data_results ['time_train'] [i]
        metric = time_train/accuracy
        metric_of_windows.append (metric)

    print (metric_of_windows)

def calculate_Pareto_curve(data_results):
    # plotar a acurácia no eixo y
    # y e o tempo de treinamento no eixo x
    # A Curva de Pareto (ou frente de Pareto) identifica os modelos que representam o melhor tradeoff
    # possível. Modelos na frente de Pareto são aqueles em que não é possível melhorar a acurácia sem
    # aumentar o tempo de treinamento, ou reduzir o tempo sem perder acurácia.

    window_size = data_results['window_size']
    accuracy = data_results['accuracy']






def calculate_metric():
    data_results = read_results_sliding(input_type='coord', ref='Wisard', window_type='sliding_window')
    calculate_time_penalty_function(data_results)
    calculate_tradeoff_norm(data_results)
    calculate_adjusted_efficienc_Metric(data_results)

calculate_metric()