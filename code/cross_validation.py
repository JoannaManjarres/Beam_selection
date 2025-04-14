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
def train_test_tradicional(connection_type):
    s008_LOS, s008_NLOS, s008_ALL = readData.read_data_s008 ()
    s009_LOS, s009_NLOS, s009_ALL = readData.read_data_s009 ()

    print('Beam selection for '+connection_type+' connection type')

    inverter_dataset = True
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

    X_train = np.array (data_train ['lidar'])
    y_train = np.array (data_train ['index_beams'])
    X_test = np.array (data_test ['lidar'])
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
        "Tamanho Treino": len (X_train),
        "Tamanho Teste": len (X_test),
        "Acurácia": accuracy
    }


    df_results = pd.DataFrame (results)
    if inverter_dataset:
        filename = "_lidar_dataset_inverter.csv"
    else:
        filename = "_lidar_dataset.csv"

    df_results.to_csv (
    "../results/score/Wisard/split_dataset/" + connection_type + "/results_" + connection_type + filename,
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
    connection_type = 'NLOS'
    input_type = 'coord'
    suffle = True
    special_case = True

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

    if suffle:
        for i in range (len (percentual)):
            path = "../results/score/Wisard/split_datasets_manual_suffle/" + connection_type + "/" + input_type + "/"
            filename = "results_" + input_type + "_" + str (
                percentual [i] * 100) + "%_s009_train_" + connection_type + ".csv"
            data = pd.read_csv (path + filename)

            ax.plot (data ['top_k'] [:10], data ['accuracy'] [:10],
                     label=data ['train_s009_%'].astype (str).tolist () [0] + '%',
                     marker='o',
                     color=colors [i])
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
    else:
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




#cross_validation_k_fold('NLOS')
#train_test_tradicional('LOS')
#split_dataset(connection_type='LOS')
#split_dataset_manual()
#plot_results_split_dataset_manual()
#plot_beams_with_position()
#test_train_s008_LOS_test_s009_NLOS()
analise_interclas()
