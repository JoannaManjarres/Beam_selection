import numpy as np

def teste_nao_atualiza_dados():
    top_k = [1,2,3,4,5]
    all_classes_order =[[159, 30, 40, 25, 120, 30],
                        [160, 22, 167, 32, 49, 7],
                        [37, 67, 32,1 , 120, 30]]

    y_test = [30, 22, 1]

    #best_classes =[]
    score = []
    for i in range (len(top_k)):
        acerto = 0
        nao_acerto = 0
        best_classes = []

        for sample in range (len (all_classes_order)):
            best_classes.append (all_classes_order[sample][:top_k[i]])
            if (y_test[sample] in best_classes[sample]):
                acerto = acerto + 1
            else:
                nao_acerto = nao_acerto + 1

        score.append(acerto / len(all_classes_order))
    a=2+2


def dictionary_test():
    #a = [3,3,3,3,3,4]
    #my_dict = {0: a, 1: [26,36,13]}
    #print(my_dict[0])

    vector = [31,52,37]
    my_dict = {}
    for i in range(len(vector)):
        my_dict[i] = vector[i]
    print(my_dict)

#teste_nao_atualiza_dados()
dictionary_test()