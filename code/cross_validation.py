import numpy as np
import matplotlib.pyplot as plt
import read_data as readData
import pandas as pd
from sklearn.model_selection import KFold
import beam_selection_wisard as bs
from sklearn.model_selection import train_test_split
import ast
import pre_process_lidar
import seaborn as sns
import problexity as px
import collections


def cross_validation_k_fold(connetion_type):
    s008_LOS, s008_NLOS, s008_ALL = readData.read_data_s008()
    s009_LOS, s009_NLOS, s009_ALL = readData.read_data_s009()

    if connetion_type == 'ALL':
        complete_dataset = pd.concat([s008_ALL, s009_ALL], ignore_index=True)
    elif connetion_type == 'LOS':
        complete_dataset = pd.concat([s008_LOS, s009_LOS], ignore_index=True)
    elif connetion_type == 'NLOS':
        complete_dataset = pd.concat([s008_NLOS, s009_NLOS], ignore_index=True)

    X = np.array(complete_dataset['lidar'])
    y = np.array(complete_dataset['index_beams'])
    score_kfold = []
    kfold = []
    score_mean = []

    k_fold = [2,3,4,5,6,7,8,9,10]
    for k in range(len(k_fold)):
        kf = KFold (n_splits=k_fold[k], shuffle=True, random_state=42)

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train = [str(x) for x in y_train]
            y_test = [str (x) for x in y_test]

            top_k, score = bs.top_k_wisard_beam_selection(x_train=X_train.tolist(),
                                                              x_test=X_test.tolist(),
                                                              y_train=y_train,
                                                              y_test=y_test,
                                                              address_of_size=44,
                                                              name_of_conf_input='')
            #print (f"Fold {i + 1}")
            #print("Top K: ", top_k)
            #print("Score: ", score)
            #print ("-" * 30)
            score_kfold.append(score)
            #kfold.append(i)

        mean_score_kfold = [np.mean(score_kfold, axis=0)]
        score_mean.append(mean_score_kfold)
        kfold.append(k_fold[k])
        #print("Mean Score K-Fold: ", mean_score_kfold)
    all_results = pd.DataFrame({"kfold": kfold, "score_kfold": score_mean})
    all_results.to_csv("../results/score/Wisard/k_fold/"+connetion_type+"/results_"+connetion_type+"_lidar_k_fold.csv", index=False)

    #
    plt.figure ()
    for i in range (len(all_results['score_kfold'].tolist())):
        plt.plot(top_k, all_results['score_kfold'].tolist()[i][0],
                 label=f'K-Fold {all_results["kfold"].tolist()[i]}')
    plt.xlabel('Top-k')
    plt.ylabel('Acurácia')
    plt.title('Seleção de Feixes Wisard com '+connetion_type+' LiDAR \n K-Fold Cross Validation')
    plt.legend()
    plt.savefig('../results/score/Wisard/k_fold/'+connetion_type+'/plot_'+connetion_type+'_lidar_k_fold.png', dpi=300, bbox_inches='tight')

    b=0
def train_test_tradicional(connection_type, input_type):
    s008_LOS, s008_NLOS, s008_ALL = readData.read_data_s008 ()
    s009_LOS, s009_NLOS, s009_ALL = readData.read_data_s009 ()

    print('Beam selection for '+connection_type+' connection type')

    inverter_dataset = False
    if connection_type == 'LOS':
        if inverter_dataset:
            print('with inverted dataset')
            data_train = s009_LOS
            data_test = s008_LOS
        else:
            print('with traditional dataset')
            data_train = s008_LOS
            data_test = s009_LOS
    elif connection_type == 'NLOS':
        if inverter_dataset:
            data_train = s009_NLOS
            data_test = s008_NLOS
        else:
            data_train = s008_NLOS
            data_test = s009_NLOS
    elif connection_type == 'ALL':
        if inverter_dataset:
            data_train = s009_ALL
            data_test = s008_ALL
        else:
            data_train = s008_ALL
            data_test = s009_ALL

    print("Data train: ", data_train.shape)
    print("Data test: ", data_test.shape)

    if input_type == 'lidar':
        X_train = np.array (data_train ['lidar'])
        X_test = np.array (data_test ['lidar'])
    if input_type == 'coord':
        X_train = np.array (data_train ['enconding_coord'])
        X_test = np.array (data_test ['enconding_coord'])
    if input_type == 'lidar_coord':
        X_train = np.array (data_train ['lidar_coord'])
        X_test = np.array (data_test ['lidar_coord'])

    y_train = np.array (data_train ['index_beams'])
    y_test = np.array (data_test ['index_beams'])

    y_train = [str (x) for x in y_train]
    y_test = [str (x) for x in y_test]

    top_k, accuracy = bs.top_k_wisard_beam_selection (x_train=X_train.tolist(),
                                                      x_test=X_test.tolist(),
                                                      y_train=y_train,
                                                      y_test=y_test,
                                                      address_of_size=44,
                                                      name_of_conf_input='')

    # Guarda os resultados
    results={
        "top_k": top_k,
        "Acurácia": accuracy,
        "Tamanho Treino": len (X_train),
        "Tamanho Teste": len (X_test)
    }


    df_results = pd.DataFrame (results)
    if inverter_dataset:
        filename = "_lidar_dataset_inverter.csv"
    else:
        filename = "_lidar_dataset.csv"
        filename =input_type+"_results_top_k_wisard_"+connection_type+".csv"


    #df_results.to_csv (
    #"../results/score/Wisard/split_dataset/" + connection_type + "/results_" + connection_type + filename,
    #index=False)
    df_results.to_csv (
        "../results/score/Wisard/split_dataset/" + connection_type + "/" +input_type + "/"  + filename,
        index=False)


def split_dataset(connection_type='ALL'):
    s008_LOS, s008_NLOS, s008_ALL = readData.read_data_s008()
    s009_LOS, s009_NLOS, s009_ALL = readData.read_data_s009()

    print('Beam selection for '+connection_type+' connection type')

    inverter_dataset = True
    if connection_type == 'LOS':
        if inverter_dataset:
            print('with inverted dataset')
            complete_dataset = pd.concat([s009_LOS, s008_LOS], ignore_index=True)
        else:
            print('with traditional dataset')
            complete_dataset = pd.concat([s008_LOS, s009_LOS], ignore_index=True)
    elif connection_type == 'NLOS':
        if inverter_dataset:
            complete_dataset = pd.concat([s009_NLOS, s008_NLOS], ignore_index=True)
        else:
            complete_dataset = pd.concat([s008_NLOS, s009_NLOS], ignore_index=True)
    elif connection_type == 'ALL':
        if inverter_dataset:
            complete_dataset = pd.concat([s009_ALL, s008_ALL], ignore_index=True)
        else:
            complete_dataset = pd.concat([s008_ALL, s009_ALL], ignore_index=True)


    X = np.array(complete_dataset['lidar'])
    y = np.array(complete_dataset['index_beams'])

    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    # Loop pelas diferentes divisões
    for test_size in test_sizes:
        # Divide os dados com o tamanho de teste atual
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=42,
                                                            shuffle=False)
                                                            #stratify=y)
        print("Data train: ", X_train.shape)
        print("Data test: ", X_test.shape)
        y_train = [str (x) for x in y_train]
        y_test = [str (x) for x in y_test]

        top_k, accuracy = bs.top_k_wisard_beam_selection(x_train=X_train.tolist(),
                                                      x_test=X_test.tolist(),
                                                      y_train=y_train,
                                                      y_test=y_test,
                                                      address_of_size=44,
                                                      name_of_conf_input='')

        # Guarda os resultados
        results.append ({
            "Tamanho Treino": len (X_train),
            "Tamanho Teste": len (X_test),
            "Proporção Teste": test_size,
            "Acurácia": accuracy
        })
    df_results = pd.DataFrame (results)
    if inverter_dataset:
        filename = "results_" + connection_type + "_lidar_split_dataset_inverter.csv"
    else:
        filename = "results_" + connection_type + "_lidar_split_dataset.csv"

    path = "../results/score/Wisard/split_dataset/" + connection_type + "/"
    df_results.to_csv(path + filename, index=False)

    b=0

def plot_results_split_dataset():
    #read a csv file
    dataset_invert = True
    connection_type = 'LOS'

    if dataset_invert:
        split_dataset_name = "results_"+connection_type+"_lidar_split_dataset_inverter.csv"
        traditional_dataset_name = "results_"+connection_type+"_lidar_dataset_inverter.csv"
        name_for_plot = 'plot_lidar_compare_split_dataset_invert.png'
        label_test = 's008'

    else:
        split_dataset_name = "results_"+connection_type+"_lidar_split_dataset.csv"
        traditional_dataset_name = "results_"+connection_type+"_lidar_dataset.csv"
        name_for_plot = 'plot_lidar_compare_split_dataset.png'
        label_test = 's009'

    split_dataset = pd.read_csv ("../results/score/Wisard/split_dataset/"+connection_type+"/"+split_dataset_name)
    traditional_dataset = pd.read_csv ("../results/score/Wisard/split_dataset/"+connection_type+"/"+traditional_dataset_name)

    top_k = np.arange(1, 51)
    size_dataset = ['0.1', '0.2', '0.3', '0.4', '0.5']


    #plot
    #plt.figure(figsize=(6,8))
    # Criando a figura
    fig, ax = plt.subplots (figsize=(6, 8))

    ax.plot (top_k [:10], traditional_dataset ['Acurácia'] [:10],
              label='Teste ' + label_test,
              marker='o',
              color='r')

    score_top_1 = []
    samples_for_test =[]
    colors = ['b', 'g', 'gold', 'c', 'm', 'y', 'k']
    for i in range(len(size_dataset)):
        ax.plot(top_k[:10], split_dataset ['Acurácia'].apply(ast.literal_eval)[i][:10],
                 label=f'Teste {size_dataset[i]}',
                 marker='o',
                 color=colors[i])
        valor = round(split_dataset['Acurácia'].apply(ast.literal_eval)[i][0],3)
        samples_test = split_dataset['Tamanho Teste'][i]
        score_top_1.append(valor)
        samples_for_test.append(samples_test)

    score_top_1.append(round(traditional_dataset['Acurácia'][0], 3))
    samples_for_test.append(traditional_dataset['Tamanho Teste'][0])
    size_dataset.append(label_test)

    top_1 = {'data_teste': size_dataset,
             'score_top_1': score_top_1,
             'samples_test': samples_for_test}

    top_1_1 = {
        'data_teste': [100, 500, 1000],  # Tamanho do dataset
        'score_top_1': [0.85, 0.88, 0.90],  # Score de acurácia
        'samples_test': [10, 50, 100]  # Número de amostras no teste
    }
    table_data = list (zip (top_1 ['data_teste'],
                            top_1 ['score_top_1'],
                            top_1 ['samples_test']))
    tabela = ax.table (cellText=table_data,  # Dados da tabela
                       colLabels=['Tamanho \n Dataset',
                                  'Score \n Top-1',
                                  'Amostras \n Teste'],  # Cabeçalhos
                       colWidths=[0.12] * 3,
                       cellLoc='center',  # Alinhamento central
                       loc='center right')#,  # Posiciona abaixo do gráfico
                       #bbox=[0, -0.3, 1, 0.2])  # Ajusta posição e tamanho [x, y, largura, altura]
    tabela.auto_set_font_size (False)
    tabela.set_fontsize (9)
    tabela.scale (1.5, 1.5)

    # Configurando cores e bordas
    for (i, j), cell in tabela.get_celld ().items ():
        cell.set_edgecolor ('black')  # Borda preta
        if i == 0:
            cell.set_text_props (weight='bold', color='white')  # Cabeçalho em negrito
            cell.set_facecolor ('#4B8BBE')  # Fundo azul para o cabeçalho
        else:
            cell.set_facecolor ('#EAEAF2')  # Fundo cinza claro para o corpo da tabela



    plt.xlabel('Top-k')
    plt.ylabel('Acurácia')
    plt.legend(loc='lower right')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xticks(top_k[:10])
    plt.title('Seleção de Feixes Wisard com '+ connection_type +' LiDAR \n variacao do tamanho do dataset de teste')

    plt.savefig('../results/score/Wisard/split_dataset/'+connection_type+'/'+name_for_plot, dpi=300, bbox_inches='tight')


def test_train_s008_LOS_test_s009_NLOS():
        input_type = 'lidar'
        s008_LOS, s008_NLOS, s008_ALL = readData.read_data_s008 ()
        s009_LOS, s009_NLOS, s009_ALL = readData.read_data_s009 ()

        label_s009_test = s009_NLOS ['index_beams'].tolist ()
        label_s008_train = s008_LOS ['index_beams'].tolist ()

        input_s009_test = s009_NLOS ['lidar'].tolist ()
        input_s008_train = s008_LOS ['lidar'].tolist ()

        X_train = input_s008_train
        y_train = label_s008_train
        X_test = input_s009_test
        y_test = label_s009_test
        y_train = [str (x) for x in y_train]
        y_test = [str (x) for x in y_test]

        top_k, accuracy = bs.top_k_wisard_beam_selection (x_train=X_train,
                                                          x_test=X_test,
                                                          y_train=y_train,
                                                          y_test=y_test,
                                                          address_of_size=44,
                                                          name_of_conf_input='')

        results = {
            "top_k": top_k,
            "samples_train_s008_LOS": len (X_train),
            "samples_teste_s009_NLOS": len (X_test),
            "accuracy": accuracy
        }
        df_results = pd.DataFrame (results)
        path = "../results/score/Wisard/test_train_s008_LOS_test_s009_NLOS/" + input_type + "/"
        filename = ("results_" + input_type + "_s008_LOS_train_s009_NLOS_test.csv")
        df_results.to_csv (path + filename, index=False)

def split_dataset_manual_intervalo_confianca():
    s008_LOS, s008_NLOS, s008_ALL = readData.read_data_s008 ()
    s009_LOS, s009_NLOS, s009_ALL = readData.read_data_s009 ()
    input_type = 'lidar_coord'
    connection_type = 'ALL'

    if connection_type == 'LOS':
        s009_data = s009_LOS
        s008_data = s008_LOS
        percentual_s008 = [0, 0.0568, 0.1137, 0.1703, 0.2272]
        percentual_s009 = [1, 0.75, 0.5, 0.25, 0]
        percentual_s009_filename = [1, 0.75, 0.5, 0.25, 0]

    elif connection_type == 'NLOS':
        s009_data = s009_NLOS
        s008_data = s008_NLOS
        # percentual_s008 = [1, 0.75, 0.5, 0.25, 0]
        percentual_s009 = [0.5771, 0.4328, 0.2885, 0.1442, 0]
        percentual_s009_filename = [1, 0.75, 0.5, 0.25, 0]
        percentual_s008 = [0, 0.25, 0.5, 0.75, 1]

    elif connection_type == 'ALL':
        s009_data = s009_ALL
        s008_data = s008_ALL
        percentual_s008 = [0, 0.2153, 0.4305, 0.6457, 0.861]
        percentual_s009 = [1, 0.75, 0.5, 0.25, 0]
        percentual_s009_filename = [1, 0.75, 0.5, 0.25, 0]

    for i in range (len (percentual_s009)):
        s009_train = s009_data.sample (frac=percentual_s009 [i])
        s008_train = s008_data.sample (frac=percentual_s008 [i])
        s009_test = s009_data.drop (s009_train.index)
        s008_test = s008_data.drop (s008_train.index)

        label_s009_train = s009_train ['index_beams'].tolist ()
        label_s008_train = s008_train ['index_beams'].tolist ()
        label_s009_test = s009_test ['index_beams'].tolist ()
        label_s008_test = s008_test ['index_beams'].tolist ()

        if input_type == 'lidar':
            input_s009_train = s009_train ['lidar'].tolist ()
            input_s008_train = s008_train ['lidar'].tolist ()
            input_s009_test = s009_test ['lidar'].tolist ()
            input_s008_test = s008_test ['lidar'].tolist ()
        elif input_type == 'coord':
            input_s009_train = s009_train ['enconding_coord'].tolist ()
            input_s008_train = s008_train ['enconding_coord'].tolist ()
            input_s009_test = s009_test ['enconding_coord'].tolist ()
            input_s008_test = s008_test ['enconding_coord'].tolist ()
        elif input_type == 'lidar_coord':
            input_s009_train = s009_train ['lidar_coord'].tolist ()
            input_s008_train = s008_train ['lidar_coord'].tolist ()
            input_s009_test = s009_test ['lidar_coord'].tolist ()
            input_s008_test = s008_test ['lidar_coord'].tolist ()

        X_train = input_s009_train + input_s008_train
        y_train = label_s009_train + label_s008_train

        X_test = input_s009_test + input_s008_test
        y_test = label_s009_test + label_s008_test

        y_train = [str (x) for x in y_train]
        y_test = [str (x) for x in y_test]

        acc = []
        for j in range(10):
            top_k, accuracy = bs.top_k_wisard_beam_selection (x_train=X_train,
                                                          x_test=X_test,
                                                          y_train=y_train,
                                                          y_test=y_test,
                                                          address_of_size=44,
                                                          name_of_conf_input='')
            acc.append(accuracy)
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        # Guarda os resultados
        results = {
            "top_k": top_k,
            "samples_train": len (X_train),
            "train_s009_%": (percentual_s009_filename [i] * 100),
            "train_s009_samples": len (input_s009_train),
            "train_s008_%": (percentual_s008 [i] * 100),
            "train_s008_samples": len (input_s008_train),
            "samples_teste": len (X_test),
            "test_s009_samples": len (input_s009_test),
            "test_s008_samples": len (input_s008_test),
            "acc_mean": acc_mean,
            "acc_std": acc_std
            }
        df_results = pd.DataFrame (results)
        path = "../results/score/Wisard/split_datasets_manual_suffle/" + connection_type + "/" + input_type + "/std/"
        if connection_type == 'NLOS':
            filename = ("results_std_" + input_type + "_" +
                        str (percentual_s009_filename [i] * 100) + "%_s009_train_" + connection_type + ".csv")
        else:
            filename = ("results_std_" + input_type + "_" +
                        str (percentual_s009 [i] * 100) + "%_s009_train_" + connection_type + ".csv")
        df_results.to_csv (path + filename, index=False)

def split_dataset_manual(connection_type='NLOS'):

    s008_LOS, s008_NLOS, s008_ALL = readData.read_data_s008()
    s009_LOS, s009_NLOS, s009_ALL = readData.read_data_s009()
    input_type = 'coord'#'lidar' 'lidar_coord'
    if_shuffle = True


    if if_shuffle:
        if connection_type == 'LOS':
            s009_data = s009_LOS
            s008_data = s008_LOS
            percentual_s008 = [0, 0.0568, 0.1137, 0.1703, 0.2272]
            percentual_s009 = [1, 0.75, 0.5, 0.25, 0]
            percentual_s009_filename = [1, 0.75, 0.5, 0.25, 0]

        elif connection_type == 'NLOS':
            s009_data = s009_NLOS
            s008_data = s008_NLOS
            #percentual_s008 = [1, 0.75, 0.5, 0.25, 0]
            percentual_s009 = [0.5771, 0.4328, 0.2885, 0.1442, 0]
            percentual_s009_filename = [1, 0.75, 0.5, 0.25, 0]
            percentual_s008 = [0, 0.25, 0.5, 0.75, 1]

        elif connection_type == 'ALL':
            s009_data = s009_ALL
            s008_data = s008_ALL
            percentual_s008 = [0, 0.2153, 0.4305, 0.6457, 0.861]
            percentual_s009 = [1, 0.75, 0.5, 0.25, 0]
            percentual_s009_filename = [1, 0.75, 0.5, 0.25, 0]

        # Dataset          --->   %  --> s009  --> s008
        # ALL
        # 9635 --> 1    --> 9638 --> 0 (0/11194) 0%
        # 9638 --> 0.75 --> 7228 --> 2410 (2410/11194) 21,5%
        # 9638 --> 0.5  --> 4819 --> 4819 (4819/11194) 48,5%
        # 9638 --> 0.25 --> 2410 --> 7228 (7228/11194) 72,5%
        # 9638 --> 0    --> 0    --> 9638 (9638/11194) 86%
        for i in range(len(percentual_s009)):
            s009_train = s009_data.sample(frac=percentual_s009[i])
            s008_train = s008_data.sample(frac=percentual_s008[i])
            s009_test = s009_data.drop(s009_train.index)
            s008_test = s008_data.drop(s008_train.index)

            label_s009_train = s009_train ['index_beams'].tolist ()
            label_s008_train = s008_train ['index_beams'].tolist ()
            label_s009_test = s009_test ['index_beams'].tolist ()
            label_s008_test = s008_test ['index_beams'].tolist ()

            if input_type == 'lidar':
                input_s009_train = s009_train['lidar'].tolist()
                input_s008_train = s008_train['lidar'].tolist()
                input_s009_test = s009_test['lidar'].tolist()
                input_s008_test = s008_test['lidar'].tolist()
            elif input_type == 'coord':
                input_s009_train = s009_train['enconding_coord'].tolist ()
                input_s008_train = s008_train['enconding_coord'].tolist ()
                input_s009_test = s009_test['enconding_coord'].tolist ()
                input_s008_test = s008_test['enconding_coord'].tolist ()
            elif input_type == 'lidar_coord':
                input_s009_train = s009_train['lidar_coord'].tolist()
                input_s008_train = s008_train['lidar_coord'].tolist()
                input_s009_test = s009_test['lidar_coord'].tolist()
                input_s008_test = s008_test['lidar_coord'].tolist()



            X_train = input_s009_train + input_s008_train
            y_train = label_s009_train + label_s008_train

            X_test = input_s009_test + input_s008_test
            y_test = label_s009_test + label_s008_test

            y_train = [str (x) for x in y_train]
            y_test = [str (x) for x in y_test]

            top_k, accuracy = bs.top_k_wisard_beam_selection (x_train=X_train,
                                                              x_test=X_test,
                                                              y_train=y_train,
                                                              y_test=y_test,
                                                              address_of_size=44,
                                                              name_of_conf_input='')
            # Guarda os resultados
            results = {
                "top_k": top_k,
                "samples_train": len (X_train),
                "train_s009_%": (percentual_s009_filename[i] * 100),
                "train_s009_samples": len (input_s009_train),
                "train_s008_%": (percentual_s008[i] * 100),
                "train_s008_samples": len (input_s008_train),
                "samples_teste": len (X_test),
                "test_s009_samples": len (input_s009_test),
                "test_s008_samples": len (input_s008_test),
                "accuracy": accuracy
            }
            save_results = False
            if save_results:
                df_results = pd.DataFrame (results)
                path = "../results/score/Wisard/split_datasets_manual_suffle/" + connection_type + "/" + input_type + "/"
                if connection_type == 'NLOS':
                    filename = ("results_" + input_type + "_" +
                            str(percentual_s009_filename[i] * 100) + "%_s009_train_" + connection_type + ".csv")

                else:
                    filename = ("results_" + input_type + "_" +
                            str(percentual_s009[i] * 100) + "%_s009_train_" + connection_type + ".csv")
                df_results.to_csv(path + filename, index=False)
            else:
                return results

    else:
        #Tamanho do dataset para treinamento:
        # ALL= 9638

        # LOS=  1473
        # NLOS = 4712

        if input_type == 'lidar':
            input_s009 = s009_ALL['lidar'].tolist()
            input_s008 = s008_ALL['lidar'].tolist()
            label_s009 = s009_ALL['index_beams'].tolist()
            label_s008 = s008_ALL['index_beams'].tolist()

        size_dataset_train = len(s009_ALL)
        percentual = [1, 0.75, 0.5, 0.25, 0]
        print("| percentual | Star index s009 | End index s009 | Star index s008 | End index s008")
        for i in range(len(percentual)):

            star_index_s009 = 0
            end_index_s009 = int(size_dataset_train * percentual[i])
            star_index_s008 = 0
            end_index_s008 = size_dataset_train-end_index_s009
            print(" | ", percentual[i], " | ", star_index_s009, " | ", end_index_s009, " | ", star_index_s008, " | ", end_index_s008)

            X_train = input_s009[star_index_s009:end_index_s009] + input_s008 [star_index_s008:end_index_s008]
            X_test = input_s009[end_index_s009:] + input_s008[end_index_s008:]
            y_train = label_s009[star_index_s009:end_index_s009] + label_s008 [star_index_s008:end_index_s008]
            y_test = label_s009[end_index_s009:] + label_s008[end_index_s008:]
            print("Data train: ", len(X_train))
            print("Data test: ", len(X_test))

        y_train = [str(x) for x in y_train]
        y_test = [str(x) for x in y_test]

        top_k, accuracy = bs.top_k_wisard_beam_selection (x_train=X_train,
                                                          x_test=X_test,
                                                          y_train=y_train,
                                                          y_test=y_test,
                                                          address_of_size=44,
                                                          name_of_conf_input='')
        # Guarda os resultados
        results = {
            "proporção Treino %": (percentual [i] * 100),
            "top_k": top_k,
            "Tamanho Train": len (X_train),
            "Train s009": len (input_s009 [star_index_s009:end_index_s009]),
            "Train s008": len (input_s008 [star_index_s008:end_index_s008]),
            "Tamanho Teste": len (X_test),
            "Test s009": len (input_s009 [end_index_s009:]),
            "Test s008": len (input_s008 [end_index_s008:]),
            "Acurácia": accuracy
        }
        df_results = pd.DataFrame (results)
        path = "../results/score/Wisard/split_datasets_manual_sequential/" + connection_type + "/" + input_type + "/"
        filename = "results_" + input_type + "_" + str (percentual [i] * 100) + "%_s009_train_" + connection_type + ".csv"
        df_results.to_csv (path + filename, index=False)

def plot_results_split_dataset_manual():
    #read a csv file
    dataset_invert = True
    connection_type = 'ALL'
    input_type = 'lidar_coord'
    suffle = 3
    special_case = False #treino s008 LOS e teste s009 NLOS
    results_std = True

    if special_case:
        path = "../results/score/Wisard/test_train_s008_LOS_test_s009_NLOS/lidar/"
        filename = "results_lidar_s008_LOS_train_s009_NLOS_test.csv"
        data = pd.read_csv (path + filename)
        fig, ax = plt.subplots (figsize=(6, 8))

        ax.plot (data ['top_k'] [:10], data ['accuracy'] [:10],
                 #label=data ['train_s009_%'].astype (str).tolist () [0] + '%',
                 marker='o')
                 #color=colors [i])
        for i in range(len(data['top_k'][:10])):
            plt.text(data['top_k'][:10][i]+0.5, data['accuracy'][:10][i], str(round(data['accuracy'][:10][i], 2)),
                     fontsize=8, ha='center', va='bottom', color='black')
        #plt.text()
        plt.legend (loc='lower right', title='Treino s008 LOS \n Teste s009 NLOS')
        plt.xlabel ('Top-k')
        plt.xticks (data ['top_k'] [:10])
        plt.ylabel ('Acurácia')
        plt.grid (linestyle='--', linewidth=0.5)
        plt.title('Seleção de Feixes Wisard \n Train s008 LOS Test s009 NLOS  ')
        plt.savefig (path + 'train_s008_LOS_Test_s009_NLOS.png', dpi=300, bbox_inches='tight')
        plt.show ()
        a=0

    percentual = [1, 0.75, 0.5, 0.25, 0]
    fig, ax = plt.subplots (figsize=(6, 8))
    colors = ['b', 'g', 'gold', 'c', 'm']

    if results_std:
        for i in range (len (percentual)):
            path = "../results/score/Wisard/split_datasets_manual_suffle/" + connection_type + "/" + input_type + "/std/"
            filename = "results_std_" + input_type + "_" + str (
                percentual [i] * 100) + "%_s009_train_" + connection_type + ".csv"
            data = pd.read_csv (path + filename)

            #fmt='-o')
            ax.plot (data ['top_k'][:10], data['acc_mean'][:10],
                     label=data['train_s009_%'].astype(str).tolist()[0] + '%',
                     marker='.', linestyle='--', linewidth=0.7,
                     color=colors[i])
            ax.fill_between (data ['top_k'] [:10], data ['acc_mean'] [:10] - data ['acc_std'] [:10],
                             data ['acc_mean'] [:10] + data ['acc_std'] [:10],
                             alpha=0.15, color=colors[i])
            #ax.errorbar (data ['top_k'] [:10], data ['acc_mean'] [:10],
            #             yerr=data ['acc_std'] [:10], marker='.', color='red')
            a=0
        plt.legend (loc='lower right', title='percentual Treino s009:')
        plt.xlabel ('Top-k')
        plt.xticks (data ['top_k'] [:10])
        plt.ylabel ('Acurácia')
        plt.grid (linestyle='--', linewidth=0.3)
        plt.title (
            'Seleção de Feixes Wisard com ' + connection_type + ' '+
            input_type +' \n variacao na conformacao do dataset de treino - shuffle com std')
        plt.savefig(path + '_compare_split_manual_dataset_suffle_std.png', dpi=300, bbox_inches='tight' )
        #plt.show()

    if suffle==1:
        for i in range (len (percentual)):
            path = "../results/score/Wisard/split_datasets_manual_suffle/" + connection_type + "/" + input_type + "/"
            filename = "results_" + input_type + "_" + str (
                percentual [i] * 100) + "%_s009_train_" + connection_type + ".csv"
            data = pd.read_csv (path + filename)

            ax.plot (data ['top_k'] [:10], data['accuracy'][:10],
                     label=data['train_s009_%'].astype(str).tolist()[0] + '%',
                     marker='o',
                     color=colors[i])
        plt.legend (loc='lower right', title='percentual Treino s009:')
        plt.xlabel ('Top-k')
        plt.xticks (data ['top_k'] [:10])
        plt.ylabel ('Acurácia')
        plt.grid (linestyle='--', linewidth=0.5)
        plt.title (
            'Seleção de Feixes Wisard com ' + connection_type + ' '+
            input_type +' \n variacao na conformacao do dataset de treino - shuffle')
        plt.savefig(path + '_compare_split_manual_dataset_suffle.png', dpi=300, bbox_inches='tight' )
        plt.show()
    elif suffle ==2:
        for i in range(len(percentual)):
            path = "../results/score/Wisard/split_datasets_manual/" + connection_type + "/" + input_type + "/"
            filename = "results_" +input_type +"_"+ str(percentual[i]*100)+"%_s009_train_" +connection_type +".csv"
            data = pd.read_csv(path + filename)

            ax.plot (data['top_k'][:10], data['Acurácia'][:10],
                     label=data['proporção Treino %'].astype(str).tolist()[0] + '%',
                     marker='o',
                     color=colors[i])
        plt.legend(loc='lower right', title='percentual Treino s009:')
        plt.xlabel('Top-k')
        plt.xticks(data['top_k'][:10] )
        plt.ylabel('Acurácia')
        plt.grid(linestyle='--', linewidth=0.5)
        plt.title('Seleção de Feixes Wisard com '+ connection_type +' LiDAR \n variacao da proporcao do dataset de treino')
        plt.show()

def read_results_conventional_evaluation(label_input_type):
    path = '../results/score/Wisard/split_dataset/'
    connection_type = 'LOS'
    path_result = path + connection_type +'/'+ label_input_type + '/'
    file_name = label_input_type + '_results_top_k_wisard_' + connection_type + '.csv'
    data_LOS = pd.read_csv (path_result + file_name, delimiter=',')
    LOS = data_LOS[data_LOS['top_k'] <= 10]

    connection_type = 'NLOS'
    path_result = path + connection_type +'/'+ label_input_type + '/'
    file_name = label_input_type + '_results_top_k_wisard_' + connection_type + '.csv'
    data_NLOS = pd.read_csv (path_result + file_name, delimiter=',')
    NLOS = data_NLOS[data_NLOS['top_k'] <= 10]

    connection_type = 'ALL'
    path_result = path + connection_type +'/'+ label_input_type + '/'
    file_name = label_input_type + '_results_top_k_wisard_' + connection_type + '.csv'
    data_ALL = pd.read_csv (path_result + file_name, delimiter=',')
    ALL = data_ALL[data_ALL['top_k'] <= 10]

    return LOS, NLOS, ALL
def plot_test_LOS_NLOS():
    import matplotlib.pyplot as plt

    label_input_type = 'coord'
    data_LOS_coord, data_NLOS_coord, data_ALL_coord = read_results_conventional_evaluation(label_input_type)
    label_input_type = 'lidar'
    data_LOS_lidar, data_NLOS_lidar, data_ALL_lidar = read_results_conventional_evaluation(label_input_type)
    label_input_type = 'lidar_coord'
    data_LOS_lidar_coord, data_NLOS_lidar_coord, data_ALL_lidar_coord = read_results_conventional_evaluation(label_input_type)


    fig, ax = plt.subplots (1, 3, figsize=(14, 6), sharey=True)
    plt.subplots_adjust (left=0.08, right=0.98, bottom=0.1, top=0.9, hspace=0.12, wspace=0.05)
    size_of_font = 18
    ax [0].plot (data_LOS_coord['top_k'], data_LOS_coord ['Acurácia'], label='Coord LOS', marker='o')
    ax [0].text (data_LOS_coord ['top_k'].min (), data_LOS_coord ['Acurácia'] [0],
                 str (round (data_LOS_coord ['Acurácia'] [0], 3)))
    ax [0].plot (data_NLOS_coord['top_k'], data_NLOS_coord ['Acurácia'], label='Coord NLOS', marker='o')
    ax [0].text (data_NLOS_coord ['top_k'].min (), data_NLOS_coord ['Acurácia'] [0],
                 str (round (data_NLOS_coord ['Acurácia'] [0], 3)))
    ax [0].plot (data_ALL_coord['top_k'], data_ALL_coord ['Acurácia'], label='Coord ALL', marker='o')
    ax [0].text (data_ALL_coord['top_k'].min(), data_ALL_coord ['Acurácia'][0],
                 str(round(data_ALL_coord ['Acurácia'][0], 3)))
    ax [0].grid ()
    ax [0].set_xticks (data_LOS_coord ['top_k'])
    ax [0].set_xlabel ('Coordenadas \n Top-k  ', font='Times New Roman', fontsize=size_of_font)

    ax [1].plot (data_LOS_lidar['top_k'], data_LOS_lidar['Acurácia'], label='LOS', marker='o')
    ax [1].plot (data_NLOS_lidar['top_k'], data_NLOS_lidar['Acurácia'], label='NLOS', marker='o')
    ax [1].plot (data_ALL_lidar['top_k'], data_ALL_lidar['Acurácia'], label='ALL', marker='o')
    ax [1].text (data_LOS_lidar['top_k'].min(), data_LOS_lidar['Acurácia'][0],
                 str(round(data_LOS_lidar ['Acurácia'][0], 3)))
    ax [1].text (data_NLOS_lidar ['top_k'].min (), data_NLOS_lidar ['Acurácia'] [0],
                 str (round (data_NLOS_lidar['Acurácia'][0], 3)))
    ax [1].text (data_ALL_lidar ['top_k'].min (), data_ALL_lidar ['Acurácia'] [0],
                 str (round (data_ALL_lidar ['Acurácia'] [0], 3)))

    ax [1].grid ()
    ax [1].set_xticks (data_LOS_coord ['top_k'])
    ax [1].set_xlabel ('Lidar \n Top-k  ', font='Times New Roman', fontsize=size_of_font)

    ax [2].plot (data_LOS_lidar_coord['top_k'], data_LOS_lidar_coord['Acurácia'],
                 label='Lidar Coord LOS', marker='o')
    ax [2].plot (data_NLOS_lidar_coord['top_k'], data_NLOS_lidar_coord['Acurácia'],
                 label='Lidar Coord NLOS', marker='o')
    ax [2].plot (data_ALL_lidar_coord['top_k'], data_ALL_lidar_coord['Acurácia'],
                 label='Lidar Coord ALL', marker='o')
    ax [2].text (data_LOS_lidar_coord ['top_k'].min (), data_LOS_lidar_coord ['Acurácia'] [0],
                 str (round (data_LOS_lidar_coord ['Acurácia'] [0], 3)))
    ax [2].text (data_NLOS_lidar_coord ['top_k'].min (), data_NLOS_lidar_coord ['Acurácia'] [0],
                 str (round (data_NLOS_lidar_coord ['Acurácia'] [0], 3)))
    ax [2].text (data_ALL_lidar_coord ['top_k'].min (), data_ALL_lidar_coord ['Acurácia'] [0],
                 str (round (data_ALL_lidar_coord ['Acurácia'] [0], 3)))

    ax [2].grid ()
    ax [2].set_xticks (data_LOS_coord['top_k'])
    ax [2].set_xlabel ('Lidar e Coordenadas \n Top-k  ', font='Times New Roman', fontsize=size_of_font)

    ax [0].set_ylabel ('Acurácia', font='Times New Roman', fontsize=size_of_font)
    ax [1].legend ()
    plt.suptitle('Selecao de Feixe usando o modelo WiSARD', fontsize=size_of_font, font='Times New Roman')


    path_to_save = '../results/score/wisard/split_dataset/'
    file_name = 'performance_accuracy_all_LOS_NLOS_wisard.png'
    plt.savefig (path_to_save + file_name, dpi=300, bbox_inches='tight')

def plot_beams_with_position():
    type_connetion = 'LOS'
    s008_LOS, s008_NLOS, s008_ALL = readData.read_data_s008()
    s009_LOS, s009_NLOS, s009_ALL = readData.read_data_s009()

    if type_connetion == 'LOS':
        s009_data = s009_LOS
        s008_data = s008_LOS
    if type_connetion == 'NLOS':
        s009_data = s009_NLOS
        s008_data = s008_NLOS
    if type_connetion == 'ALL':
        s009_data = s009_ALL
        s008_data = s008_ALL

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 8))#, sharey=True)

    sns.scatterplot(data=s009_data, x='x', y='y', hue="index_beams", palette="deep", ax=ax[0], legend=False, label='s009')
    sns.scatterplot(data=s008_data, x='x', y='y', hue="index_beams", palette="deep", ax=ax[1], legend=False, label='s008')

    text = 'Amostras: \n'
    ax[0].text(x=750, y=515, s=text + str(len(s009_data)), fontsize=8, color='black')
    ax[1].text(x=750, y=450, s=text + str(len(s008_data)), fontsize=8, color='black')

    ax[0].set_title('s009 '+ type_connetion)
    ax[1].set_title('s008 '+ type_connetion)
    ax[0].set_xlabel('X')
    ax[1].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[1].set_ylabel('Y')
    ax[0].set_facecolor('#EEEEF5')
    ax[1].set_facecolor('#EEEEF5')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(True)
    ax[0].spines['bottom'].set_visible(True)
    ax[1].spines['left'].set_visible(True)
    ax[1].spines['bottom'].set_visible(True)
    plt.tight_layout()


    a=0

def analise_interclas():
    s008_LOS, s008_NLOS, s008_ALL = readData.read_data_s008()
    s009_LOS, s009_NLOS, s009_ALL = readData.read_data_s009()

    #Classe com frequencia > 1%
    connection_type = 'LOS'
    s008 = [s008_ALL['index_beams'].tolist(),
            s008_LOS['index_beams'].tolist(),
            s008_NLOS['index_beams'].tolist()]
    labels, counts = np.unique(s008[1], return_counts=True)
    percent = [i / sum(counts) * 100 for i in counts]
    stats_by_classes = pd.DataFrame ({'index': labels,
                                      'counts': counts,
                                      'percent': percent})
    stats_by_classes = stats_by_classes[stats_by_classes['percent'] > 1]


    #class_ = 158
    classes = stats_by_classes['index'].tolist()
    for i in range(len(stats_by_classes['index'])):

        s008_LOS_class = s008_LOS[s008_LOS['index_beams'] == classes[i]]
        x_mean =s008_LOS_class['x'].mean()
        y_mean = s008_LOS_class['y'].mean()
        x_max = s008_LOS_class['x'].max()
        y_max = s008_LOS_class['y'].max()
        text = 'x mean:'+str(np.round(x_mean,2)) + '\n' + 'y mean:'+str(np.round(y_mean,2))
        plt.plot(s008_LOS_class['x'], s008_LOS_class['y'], 'o', label='index: '+ str(classes[i]), alpha=0.5)
        plt.title('s008 '+connection_type+': Dist do indice de feixe: ' + str(classes[i]))
        plt.text(x_max-1, y_max-0.5, text, fontsize=8)
        path = '../results/score/Wisard/analise_interclasse/beams_com_freq_>_1%/s008/'+connection_type+'/'
        file_name = "index_beam_"+str(classes[i]) + ".png"
        plt.savefig(path + file_name, dpi=300, bbox_inches='tight')

        plt.clf()
    a=0

    #s009_LOS['index_beams'].value_counts().plot(kind='bar')
    #s008_LOS['index_beams'].value_counts().plot(kind='bar')
    #s009_NLOS['index_beams'].value_counts().plot(kind='bar')
    #s008_NLOS['index_beams'].value_counts().plot(kind='bar')

    #plt.show()

def complexity_calculator():

    lidar =False
    coord = True
    s008_LOS, s008_NLOS, s008_ALL = readData.read_data_s008()
    y = np.array(s008_LOS['index_beams'])

    if lidar:
        X = s008_ALL['lidar'].tolist()

    if coord:
        X = pd.concat([s008_LOS['x'], s008_LOS['y']], axis=1)
        X = np.array (X).reshape (len (s008_LOS), 2)

        #X = np.array(s008_ALL['x']).reshape(len(s008_ALL), 1)

    print( X.shape, y.shape)

    counts = collections.Counter (y)
    mask = np.array ([counts [label] >= 10 for label in y])

    X_filtered = X [mask]
    y_filtered = y [mask]

    cc = px.ComplexityCalculator()
    cc.fit (X_filtered, y_filtered)

    #----------------------------------
    plot=False
    if plot:
        # Conta as amostras por classe
        label_counts = collections.Counter(y)

        # Organiza os dados
        classes = list(label_counts.keys ())
        counts = list(label_counts.values ())

        # Ordena para visualização mais limpa
        classes, counts = zip (*sorted (zip (classes, counts), key=lambda x: x [0]))

        # Gráfico
        plt.figure (figsize=(12, 6))
        plt.bar (classes, counts, color='skyblue', edgecolor='black')
        plt.xlabel ('Rótulos (classes / feixes)')
        plt.ylabel ('Número de amostras')
        plt.title ('Distribuição das amostras por classe')
        plt.xticks (rotation=90)
        plt.grid (axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout ()
        plt.show ()
    # ----------------------------------
    print (px.__version__)
    counts =collections.Counter(y)
    mask = np.array ([counts[label] >= 2 for label in y])

    X_filtered = X [mask]
    y_filtered = y [mask]

    print ("X shape:", X_filtered.shape)
    print ("y shape:", y_filtered.shape)

    knn_metrics = ['N1', 'N2', 'LSC', 'NEC']
    custom_args = {m: {'k': 1} for m in knn_metrics}


    cc = px.ComplexityCalculator (metrics=knn_metrics, custom_metric_args=custom_args)
    cc.fit (X_filtered, y_filtered)

    # Exibir resultados
    for m in knn_metrics:
        print (f"{m}: {cc.results_ [m]:.4f}")

    # Initialize CoplexityCalculator with default parametrization
    custom_args = {
        'N1': {'k': 1},
        'N2': {'k': 1},
        'LSC': {'k': 1},
        'NEC': {'k': 1},
    }
    safe_metrics = ['F1', 'F2', 'F3', 'T1', 'T2']
    cc = px.ComplexityCalculator(metrics=safe_metrics)
    cc.fit (X, y)

    # Fit model with data


    cc.n_classes = len(np.unique(y))
    cc.n_features = X.shape[1]
    cc.n_samples = X.shape[0]
    cc.fit (X, y)
    # Calculate complexity metrics
    cc.calculate_complexity_metrics ()
    # Print complexity metrics
    print ("-" * 30)
    print ("Complexity metrics")
    print ("-" * 30)

    cc.report()
    cc.metrics()

    a=0

def analise_k_means():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    s008_LOS, s008_NLOS, s008_ALL = readData.read_data_s008 ()

    y = np.array (s008_LOS ['index_beams'])
    unique_classes, counts = np.unique(y, return_counts=True)
    percent = [i / sum (counts) * 100 for i in counts]
    stats_by_classes = pd.DataFrame ({'index': unique_classes,
                                      'counts': counts,
                                      'percent': percent})
    stats_by_classes = stats_by_classes [stats_by_classes ['percent'] > 1]

    class_ = []
    n_clusters = []
    param_silhouette = []
    for classe in stats_by_classes['index'].tolist():
        # Filtrar dados da classe
        #X_classe = X[y == classe]
        X_classe_x = s008_LOS[s008_LOS['index_beams'] == classe]['x']
        X_classe_y = s008_LOS[s008_LOS['index_beams'] == classe]['y']
        X_classe = pd.concat((X_classe_x, X_classe_y), axis=1)

        # Ajustando K-means para essa classe
        clusters_num = [2,3,4,5]
        for i in range(len(clusters_num)):
            kmeans = KMeans(n_clusters=clusters_num[i])  # Ajustar o número de clusters conforme necessário
            kmeans.fit(X_classe)

            silhouette = silhouette_score (X_classe, kmeans.labels_)
            if silhouette > 0.85:
                class_.append(classe)
                n_clusters.append(clusters_num[i])
                param_silhouette.append(silhouette)

                #print (f"Classe {classe} - Número de clusters: {clusters_num[i]} - Silhouette: {silhouette:.4f}")
                break

    print("Classes \t n_clusters \t param_silhouette: \n")
    for i in range(len(class_)):
        print("\t", class_[i],"\t", n_clusters[i], "\t",param_silhouette[i])
    print("Classes com Silhouette > 0.85: ", class_)
    print("Número de clusters: ", n_clusters)
    a=0
'''
def plot_clusters(class_, s008_LOS):
    for j in range(len(class_)):
        # Plotar os resultados de cada classe
        X_classe_x = s008_LOS [s008_LOS ['index_beams'] == class_[j]] ['x']
        X_classe_y = s008_LOS [s008_LOS ['index_beams'] == class_[j]] ['y']
        kmeans = KMeans (n_clusters=n_clusters[j])  # Ajustar o número de clusters conforme necessário
        kmeans.fit(X_classe)

        plt.scatter(X_classe['x'], X_classe['y'], c=kmeans.labels_, cmap='viridis')
        plt.title(f"Clusters para Classe {classe}")
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.show()

    a=0
'''

def hierarchical_model(input_type='lidar'):
    s008_LOS, s008_NLOS, s008_ALL = readData.read_data_s008(scale_to_coord=8)
    s009_LOS, s009_NLOS, s009_ALL = readData.read_data_s009(scale_to_coord=8)

    y_train_ = s008_ALL['LOS']
    y_test_ = s009_ALL['LOS']

    y_train = np.where(y_train_ == 'LOS=1', 1, 0)
    y_test = np.where(y_test_ == 'LOS=1', 1, 0)
    y_train = [str(x) for x in y_train]
    y_test = [str(x) for x in y_test]

    input_type=['coord', 'lidar', 'lidar_coord']
    all_results = []
    for i in range(len(input_type)):
        if input_type[i] == 'lidar':
            X_train = s008_ALL['lidar'].tolist()
            X_test = s009_ALL['lidar'].tolist()
        elif input_type[i] == 'coord':
            X_train = s008_ALL['enconding_coord'].tolist()
            X_test = s009_ALL['enconding_coord'].tolist()
        elif input_type[i] == 'lidar_coord':
            X_train = s008_ALL['lidar_coord'].tolist()
            X_test = s009_ALL['lidar_coord'].tolist()

        acc = []
        add = [6, 12, 24, 36, 44, 56, 64]
        for j in range (len(add)):
            accuracy = bs.LOS_NLOS_classification(x_train=X_train,
                                                         x_test=X_test,
                                                         y_train=y_train,
                                                         y_test=y_test,
                                                         address_of_size=add[j])
            acc.append(accuracy)
        results = {"input_type": input_type[i],
                   "add": add,
                   "accuracy": acc}
        all_results.append(results)
    results_ = pd.DataFrame(all_results)

    results_coord = results_[results_['input_type']=='coord']
    results_lidar = results_[results_['input_type']=='lidar']
    results_lidar_coord = results_[results_['input_type']=='lidar_coord']

    plt.plot(results_coord['add'].tolist()[0], results_coord['accuracy'].tolist()[0], marker='o', label='Coord')
    plt.plot(results_lidar['add'].tolist()[0], results_lidar['accuracy'].tolist()[0], marker='o', label='Lidar')
    plt.plot(results_lidar_coord['add'].tolist()[0], results_lidar_coord['accuracy'].tolist()[0], marker='o', label='Lidar e Coord')
    plt.xlabel('Add')
    plt.legend()
    plt.xticks(results_coord['add'].tolist()[0])
    plt.ylabel('Accuracy')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.title('Classificacao LOS-NLOS com a rede WiSARD')


    a=0



#split_dataset_manual_intervalo_confianca()
hierarchical_model()

