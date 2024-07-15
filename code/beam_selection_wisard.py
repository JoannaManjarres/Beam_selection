import numpy as np
import csv
import wisardpkg as wp
import timeit
import math
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score


def tic():
    global tic_s
    tic_s = timeit.default_timer()
def toc():
    global tic_s
    toc_s = timeit.default_timer()

    return (toc_s - tic_s)
def config_red_wisard(addressSize):
    # addressSize # number of addressing bits in the ram
    ignoreZero = False  # optional; causes the rams to ignore the address 0

    # False by default for performance reasons,
    # when True, WiSARD prints the progress of train() and classify()
    verbose = False

    wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)
    return wsd
def redWizard(data_train,
              label_train,
              data_validation,
              addressSize):
    # addressSize # number of addressing bits in the ram
    ignoreZero = False  # optional; causes the rams to ignore the address 0

    # False by default for performance reasons,
    # when True, WiSARD prints the progress of train() and classify()
    verbose = False

    wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose, bleachingActivated=True)

    #print('\n Training WISARD net ...')
    tic()
    # train using the input data
    wsd.train(data_train, label_train)

    tiempo_entrenamiento_ms = toc()

    #print('\n Selecting Beams using WISARD network ...')
    tic()
    # classify some data
    salida_de_la_red = wsd.classify(data_validation)
    tiempo_test_ms = toc()


    return salida_de_la_red, tiempo_entrenamiento_ms, tiempo_test_ms


def calculoDesvioPadrao(input_vector):
    sumatoria = 0
    numero_de_elementos = len(input_vector)
    for i in range(numero_de_elementos):
        sumatoria = sumatoria + input_vector[i]

    media = sumatoria / numero_de_elementos
    sumatoria = 0
    for i in range(numero_de_elementos):
        sumatoria = + (input_vector[i] - media) ** 2
    desvio_padrao = math.sqrt(sumatoria / numero_de_elementos)

    return [media, desvio_padrao]

def plotarResultados(x_vector,
                     y_vector,
                     desvio_padrao_vector,
                     titulo,
                     nombre_curva,
                     x_label,
                     y_label,
                     ruta):
    #print(ruta)
    plt.figure()
    plt.errorbar(x_vector, y_vector, yerr=desvio_padrao_vector, fmt='o', label=nombre_curva, capsize=5, ecolor='red')
    plt.grid()
    plt.legend(loc="best")
    plt.title(titulo)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_vector)
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    #plt.show()

def select_best_beam(input_train,
                     input_validation,
                     label_train,
                     label_validation,
                     figure_name,
                     antenna_config,
                     type_of_input,
                     titulo_figura,
                     user,
                     enableDebug=False,
                     plot_confusion_matrix_enable=False):

    # config parameters
    if (enableDebug):
        address_size = [28]
        numero_experimentos = 2
    else:
        #address_size = [28]

        #address_size = [4, 8, 10]
        #address_size = [4, 8, 12, 16, 20,24, 28,]
        #address_size = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48,]
        #address_size = [ 8, 12, 16, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
        #address_size = [6,  12, 18, 24, 28, 30, 36, 42, 48, 54, 60]
        address_size = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]

        numero_experimentos = 10

    vector_time_train_media = []
    vector_time_test_media = []
    vector_acuracia_media = []

    vector_acuracia_desvio_padrao = []
    vector_time_train_desvio_padrao = []
    vector_time_test_desvio_padrao = []

    path_result = "../results"
    print("Tamanho da \nMemoria\t\t|\tRodada\t|\tAcuracia")

    for j in range(len(address_size)):  # For encargado de variar el tamano de la memoria

        vector_acuracia = []
        vector_time_test = []
        vector_time_train = []
        vector_matriz_confusion = []
        #matriz_confusion_sumatoria = np.zeros((numero_de_grupos, numero_de_grupos), dtype=float)

        print('\t'+str(address_size[j]))

        for i in range(numero_experimentos):  # For encargado de ejecutar el numero de rodadas (experimentos)


            # -----------------USA LA RED WIZARD -------------------
            out_red, time_train, time_test = redWizard(input_train,
                                                       label_train,
                                                       input_validation,
                                                       address_size[j])

            vector_time_train.append(time_train)
            vector_time_test.append(time_test)

            # #----------------- CALCULA MATRIZ DE CONFUSION -----------------------
            titulo = "** MATRIZ DE CONFUSÃO " + str(i) + " **" + " \n Address Size " + str(address_size[j])

            #matrizdeconfusion = calcularMatrixDeConfusion(label_validation,
            #                                              out_red,
            #                                              titulo)
            #matriz_confusion_sumatoria = matriz_confusion_sumatoria + matrizdeconfusion

            #print('\n Measuring output performance ...')
            acuracia = accuracy_score(label_validation, out_red)
            npz_index_predict = path_result+'/'+type_of_input+'/index_beams_predict/' + f'index_beams_predict_{i}' + '.npz'
            np.savez (npz_index_predict, output_classification=npz_index_predict)
            vector_acuracia.append(acuracia)
            print('\t\t\t\t\t' + str(i) + '\t|\t' + str(acuracia))

        # ----------------- CALCULA ESTADISTICAS -----------------------
        [acuracia_media, acuracia_desvio_padrao] = calculoDesvioPadrao(vector_acuracia)
        [time_train_media, time_train_desvio_padrao] = calculoDesvioPadrao(vector_time_train)
        [time_test_media, time_test_desvio_padrao] = calculoDesvioPadrao(vector_time_test)
        #matriz_confusion_media = matriz_confusion_sumatoria / numero_experimentos

        # ----------------- GUARDA VECTORES DE ESTADISTICAS -----------------------
        vector_acuracia_media.append(acuracia_media)
        vector_acuracia_desvio_padrao.append(acuracia_desvio_padrao)

        vector_time_train_media.append(time_train_media)
        vector_time_train_desvio_padrao.append(time_train_desvio_padrao)

        vector_time_test_media.append(time_test_media)
        vector_time_test_desvio_padrao.append(time_test_desvio_padrao)

        # np.savez( path_result+"metricas.npz",
        #          matriz_confusao = vector_matriz_confusion)

        # ----------------- IMPRIME MATRIZ DE CONFUSION MEDIA -----------------------
        #titulo_mc = "** MATRIZ DE CONFUSÃO MÉDIA ** \n Address Size " + str(address_size[j])
        #df_cm = pd.DataFrame(matriz_confusion_media, index=range(0, numero_de_grupos),
        #                     columns=range(0, numero_de_grupos))
        #path_confusion_matriz = path_result + 'confusionMatrix/' + titulo_mc + ".png"
        #if plot_confusion_matrix_enable:
        #    pretty.pretty_plot_confusion_matrix(df_cm, cmap='Blues', title=titulo_mc,
        #                                        nombreFigura=path_confusion_matriz)

    # ----------------- GUARDA EM CSV VECTORES DE ESTADISTICAS  -----------------------
    print ("-------------------------------------------")
    print('\n Saving results files ...')
    print ("-------------------------------------------")

    with open(path_result + '/accuracy/'+antenna_config+'/'+type_of_input+'/'+user+'/acuracia_' + figure_name + '.csv', 'w') as f:
        writer_acuracy = csv.writer(f, delimiter='\t')
        writer_acuracy.writerows(zip(address_size, vector_acuracia_media, vector_acuracia_desvio_padrao))

    with open(path_result + '/processingTime/'+antenna_config+'/'+type_of_input + '/' + user + '/time_train_' + figure_name + '.csv', 'w') as f:
        writer_time_train = csv.writer(f, delimiter='\t')
        writer_time_train.writerows(zip(address_size, vector_time_train_media, vector_time_train_desvio_padrao))

    with open(path_result + '/processingTime/'+antenna_config+'/'+type_of_input + '/' + user +'/time_test_' + figure_name + '.csv', 'w') as f:
        writer_time_test = csv.writer(f, delimiter='\t')
        writer_time_test.writerows(zip(address_size, vector_time_test_media, vector_time_test_desvio_padrao))

    # ----------------- PLOT DE RESULTADOS  ------------------------------
    titulo = titulo_figura
    nombre_curva = "Dado com desvio padrão"

    plotarResultados(address_size,
                     vector_acuracia_media,
                     vector_acuracia_desvio_padrao,
                     titulo,
                     nombre_curva,
                     "Tamanho da memória",
                     "Acuracia Média",
                     ruta=path_result + '/accuracy/'+antenna_config+'/'+type_of_input + '/' + user +'/acuracia_'+figure_name+'.png')


    plotarResultados(address_size,
                     vector_time_train_media,
                     vector_time_train_desvio_padrao,
                     titulo,
                     nombre_curva,
                     "Tamanho da memória",
                     "Tempo de treinamento Médio (s)",
                     ruta=path_result + '/processingTime/'+antenna_config+'/'+type_of_input + '/' + user +'/time_train_'+figure_name+'.png''')

    plotarResultados(address_size,
                     vector_time_test_media,
                     vector_time_test_desvio_padrao,
                     titulo,
                     nombre_curva,
                     "Tamanho da memória",
                     "Tempo de Teste Médio (s)",
                     ruta=path_result + '/processingTime/'+antenna_config+'/'+type_of_input + '/' + user +'/time_test_'+figure_name+'.png')


    return out_red#, df_cm





def beam_selection_top_k_wisard(x_train, x_test,
                                y_train, y_test,
                                data_input, data_set,
                                address_of_size,
                                name_of_conf_input):

    #print("Calculate top-k with Wisard")
    print ("... Calculando os top-k com Wisard")
    addressSize = address_of_size
    ignoreZero = False
    verbose = True
    var = True
    wsd = wp.Wisard(addressSize,
                    ignoreZero=ignoreZero,
                    verbose=verbose,
                    returnConfidence=var,
                    returnActivationDegree=var,
                    returnClassesDegrees=var)
    wsd.train(x_train, y_train)

    # the output is a list of string, this represent the classes attributed to each input
    out = wsd.classify(x_test)

    #wsd_1 = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)
    #wsd_1.train(x_train, y_train)
    #out_1 = wsd_1.classify(x_test)


    content_index = 0
    ram_index = 0
    #print(wsd.getsizeof(ram_index,content_index))
    #print(wsd.json())
    #print(out)

    #top_k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
    #top_k = [ 10, 20, 30, 40, 50]
    top_k = np.arange(1, 51, 1)


    acuracia = []
    score = []

    all_classes_order = []

    for sample in range(len(out)):

        classes_degree = out[sample]['classesDegrees']
        dict_classes_degree_order = sorted(classes_degree, key=itemgetter('degree'), reverse=True)

        classes_by_sample_in_order = []
        for x in range(len(dict_classes_degree_order)):
            classes_by_sample_in_order.append(dict_classes_degree_order[x]['class'])
        all_classes_order.append(classes_by_sample_in_order)

    path_index_predict = '../results/index_beams_predict/WiSARD/top_k/' + name_of_conf_input + '/'
    for i in range (len(top_k)):
        acerto = 0
        nao_acerto = 0
        best_classes = []
        best_classes_int = []
        for sample in range (len(all_classes_order)):
            best_classes.append(all_classes_order[sample][:top_k[i]])
            best_classes_int.append([int(x) for x in best_classes[sample]])
            if (y_test[sample] in best_classes[sample]):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        score.append(acerto / len(out))

        file_name = 'index_beams_predict_top_' + str(top_k[i]) + '.npz'
        npz_index_predict = path_index_predict + file_name
        np.savez(npz_index_predict, output_classification=best_classes_int)

    df_score_wisard_top_k = pd.DataFrame ({"Top-K": top_k, "Acuracia": score})
    path_csv = '../results/accuracy/8X32/' + data_input + '/top_k/'
    df_score_wisard_top_k.to_csv (path_csv + 'score_wisard_' + name_of_conf_input + '_top_k.csv', index=False)



    print ('Enderecamento de memoria: ', addressSize)
    '''
    for i in range(len(top_k)):
        acerto = 0
        nao_acerto = 0

        if top_k[i] == 1:
            a=0
            #acuracia_tpo_1 = accuracy_score(y_test, out_1)
            #print('Acuracia top k =1: ', acuracia_tpo_1)

        for amostra_a_avaliar in range(len(out)):

            lista_das_classes = out[amostra_a_avaliar]['classesDegrees']
            dict_com_classes_na_ordem = sorted(lista_das_classes, key=itemgetter('degree'), reverse=True)
            #f.write(str(dict_com_classes_na_ordem))

            classes_na_ordem_descendente = []
            for x in range(len(dict_com_classes_na_ordem)):
                classes_na_ordem_descendente.append(dict_com_classes_na_ordem[x]['class'])

            top_5 = classes_na_ordem_descendente[0:top_k[i]]

            if top_k[i] == 1:
                index_predict_top_1.append(top_5)
            if top_k[i] == 2:
                index_predict_top_2.append(top_5)
            if top_k[i] == 3:
                index_predict_top_3.append(top_5)
            if top_k [i] == 4:
                index_predict_top_4.append (top_5)
            elif top_k[i] == 5:
                index_predict_top_5.append(top_5)
            elif top_k[i] == 6:
                index_predict_top_6.append(top_5)
            elif top_k[i] == 7:
                index_predict_top_7.append(top_5)
            elif top_k[i] == 8:
                index_predict_top_8.append(top_5)
            elif top_k[i] == 9:
                index_predict_top_9.append(top_5)
            elif top_k[i] == 10:
                index_predict_top_10.append(top_5)
            elif top_k[i] == 20:
                index_predict_top_20.append(top_5)
            elif top_k[i] == 30:
                index_predict_top_30.append(top_5)
            elif top_k[i] == 40:
                index_predict_top_40.append(top_5)
            elif top_k[i] == 50:
                index_predict_top_50.append(top_5)


            if( y_test[amostra_a_avaliar] in top_5):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        acuracia.append(acerto/len(out))

    #print("len(out):", len(out))
    #print("TOP-K: ", top_k)
    #print("Acuracia: ",acuracia)
    #f.close()
    path_index_predict = '../results/index_beams_predict/WiSARD/top_k/'+name_of_conf_input+'/'
    for i in range(len(top_k)):
        file_name = 'index_beams_predict_top_'+str(top_k[i])+'.npz'
        npz_index_predict = path_index_predict + file_name
        if top_k[i] == 1:
            np.savez(npz_index_predict, output_classification=index_predict_top_1)
        if top_k[i] == 2:
            np.savez(npz_index_predict, output_classification=index_predict_top_2)
        if top_k[i] == 3:
            np.savez(npz_index_predict, output_classification=index_predict_top_3)
        if top_k[i] == 4:
            np.savez(npz_index_predict, output_classification=index_predict_top_4)
        elif top_k[i] == 5:
            np.savez(npz_index_predict, output_classification=index_predict_top_5)
        elif top_k[i] == 6:
            np.savez(npz_index_predict, output_classification=index_predict_top_6)
        elif top_k[i] == 7:
            np.savez(npz_index_predict, output_classification=index_predict_top_7)
        elif top_k[i] == 8:
            np.savez(npz_index_predict, output_classification=index_predict_top_8)
        elif top_k[i] == 9:
            np.savez(npz_index_predict, output_classification=index_predict_top_9)
        elif top_k[i] == 10:
            np.savez(npz_index_predict, output_classification=index_predict_top_10)
        elif top_k[i] == 20:
            np.savez(npz_index_predict, output_classification=index_predict_top_20)
        elif top_k[i] == 30:
            np.savez(npz_index_predict, output_classification=index_predict_top_30)
        elif top_k[i] == 40:
            np.savez(npz_index_predict, output_classification=index_predict_top_40)
        elif top_k[i] == 50:
            np.savez(npz_index_predict, output_classification=index_predict_top_50)

    #npz_index_predict = '../results/index_beams_predict/top_k/' + f'index_beams_predict_top_{top_k[i]}' + '.npz'
    #np.savez (npz_index_predict, output_classification=estimated_beams)
    '''
    print ("-----------------------------")
    print ("TOP-K \t\t|\t Acuracia")
    print("-----------------------------")
    for i in range(len(top_k)):
        if top_k[i] == 1:
            print('K = ', top_k[i], '\t\t|\t ', np.round(score[i],3)), '\t\t|'
        elif top_k[i] == 5:
            print ('K = ', top_k [i], '\t\t|\t ', np.round (score [i], 3)), '\t\t|'
        else:
            print('K = ', top_k[i], '\t|\t ', np.round(score[i],3)), '\t\t|'
    print ("-----------------------------")


    #df_acuracia_wisard_top_k = pd.DataFrame({"Top-K": top_k, "Acuracia": acuracia})
    #path_csv='../results/accuracy/8X32/'+data_input+'/top_k/'
    #print(path_csv+'acuracia_wisard_' + data_input + '_top_k.csv')
    #df_acuracia_wisard_top_k.to_csv(path_csv + 'acuracia_wisard_' + data_input + '_' + data_set + '_top_k.csv', index=False)
    #df_acuracia_wisard_top_k.to_csv (path_csv + 'acuracia_wisard_' + name_of_conf_input + '_top_k.csv',
    #                                 index=False)


    plot_top_k(top_k, score, data_input, name_of_conf_input=name_of_conf_input)
    return top_k, acuracia

def plot_top_k(top, acuracia, data_input, name_of_conf_input):
    plt.figure()
    plt.plot(top, acuracia, 'o-')
    #plt.plot(top, acuracia, width=3, label='Wisard')
    #for i in range(len(top)):
    #    plt.text(top[i], acuracia[i], str("{:.2f}".format(acuracia[i])), ha='center', va='bottom')

    #plt.text(1, acuracia[0], str(acuracia[0]), ha='center', va='bottom')
    #plt.grid(False)
    plt.title('Classificacao Top-k Wisard com '+name_of_conf_input)
    plt.xlabel('Top-K')
    plt.xticks(top)
    plt.ylabel('Acuracia')
    plt.legend()
    #plt.savefig('../results/accuracy/8X32/'+data_input+'/acuracia_top_k_wisard_'+data_input+'_'+dataset+'.png', dpi=300, bbox_inches='tight')
    #plt.savefig (
    #    '../results/accuracy/8X32/' + data_input + '/top_k/acuracia_top_k_wisard_' + name_of_conf_input + '.png',
    #    dpi=300, bbox_inches='tight')
    plt.show()
