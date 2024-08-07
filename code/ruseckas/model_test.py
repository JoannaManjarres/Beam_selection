
import numpy as np
import pandas as pd
import timeit
from tensorflow.keras.models import model_from_json
from tensorflow.keras import layers, metrics, optimizers, losses, callbacks, regularizers

from resnet import AddRelu

def tic():
    global tic_s
    tic_s = timeit.default_timer()
def toc():
    global tic_s
    toc_s = timeit.default_timer()
    return (toc_s - tic_s)


def get_beams_output(filename, threshold_below_max=6):
    #y_matrix = load_data (filename, 'output_classification')
    y_matrix = load_data (filename)
    y_matrix = np.abs(y_matrix)
    y_matrix /= np.max(y_matrix)  # normalize

    num_classes = y_matrix.shape [1] * y_matrix.shape [2]
    y = y_matrix.reshape (y_matrix.shape [0], num_classes)

    y = beams_remove_small (y, threshold_below_max)
    return y, num_classes


def beams_remove_small(y, threshold_below_max):
    num = y.shape [0]
    for i in range (num):
        beams = y [i, :]
        logs = 20 * np.log10 (beams + 1e-30)
        beams [logs < (np.amax (logs) - threshold_below_max)] = 0
        beams = beams / np.sum (beams)
        y[i, :] = beams

    return y
def load_data(filename):
    cache = np.load (filename, allow_pickle=True)
    keys = list (cache.keys())
    X = cache[keys[0]]

    return X

def process_coordinates(X, means=None, stds=None):
    X_xyz, means, stds = normalize_data(X[:, :3], means, stds)
    X = np.concatenate((X_xyz, X[:, 3:]), axis=1)
    return X, means, stds
def normalize_data(X, means=None, stds=None):
    if means is None: means = np.mean(X, axis=0)
    if stds is None: stds = np.std(X, axis=0)
    X_norm = (X - means) / stds
    return X_norm, means, stds



DATA_DIR = '../../data/'
LIDAR_DATA_DIR = DATA_DIR +'lidar/s009/'
COORDINATES_DATA_DIR = DATA_DIR + 'coord/ruseckas/'
#IMAGES_DATA_DIR = DATA_DIR + 'image_input/'
#BEAMS_DATA_DIR = DATA_DIR + 'beams_output/beams_generate_by_me/'
BEAMS_DATA_DIR = DATA_DIR + 'beams_output/beam_output_baseline_raymobtime_s009/'


LIDAR_TEST_FILE = LIDAR_DATA_DIR + 'lidar_test_raymobtime.npz'
COORDINATES_TEST_FILE = COORDINATES_DATA_DIR + 'my_coord_test.npz'
#BEAMS_TEST_FILE = BEAMS_DATA_DIR + 'beams_output_8x32_test.npz'
BEAMS_TEST_FILE = BEAMS_DATA_DIR + 'beams_output_test.npz'
print(BEAMS_TEST_FILE)


BATCH_SIZE = 32
BEST_WEIGTHS = 'my_model_weights_both.h5'
KERNEL_REG = 1.e-4
num_classes = 256

#   Read the data
X_lidar_test = load_data(LIDAR_TEST_FILE)

X_coord_test = load_data(COORDINATES_TEST_FILE)

cache = np.load(COORDINATES_DATA_DIR+'coord_train_stats.npz')
coord_means = cache['coord_means']
coord_stds = cache['coord_stds']
X_coord_test, _, _ = process_coordinates(X_coord_test, coord_means, coord_stds)

index_true, _ = get_beams_output(BEAMS_TEST_FILE)

only_lidar = False
only_coord = False
both = True

if (only_lidar):
    #test_generator = TestDataGeneratorLidar(X_lidar_test, index_true, BATCH_SIZE)
    X_data = X_lidar_test
    flag_file = 'lidar'
    folder_model = 'model_'+flag_file+'/'
    BEST_WEIGTHS =folder_model+'my_model_weights_lidar.h5'
    with open(folder_model+'my_model_lidar.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json, custom_objects={'AddRelu': AddRelu})

if(only_coord):
    #test_generator = TestDataGeneratorLidar(X_coord_test, index_true, BATCH_SIZE)
    X_data = X_coord_test
    flag_file = 'coord'
    folder_model = 'model_' + flag_file + '/'
    BEST_WEIGTHS = folder_model+'my_model_weights_coord.h5'
    with open (folder_model+'my_model_coord.json', 'r') as json_file:
        loaded_model_json = json_file.read ()
        model = model_from_json (loaded_model_json, custom_objects={'AddRelu': AddRelu})

if(both):
    #test_generator = TestDataGeneratorBoth(X_lidar_test, X_coord_test, index_true, BATCH_SIZE)
    X_data = (X_lidar_test, X_coord_test)
    flag_file = 'lidar_coord'
    folder_model = 'model_' + flag_file + '/'
    BEST_WEIGTHS = folder_model+'my_model_weights_both.h5'
    with open (folder_model+'my_model.json', 'r') as json_file:
        loaded_model_json = json_file.read ()
        model = model_from_json (loaded_model_json, custom_objects={'AddRelu': AddRelu})

optim = optimizers.legacy.Adam()


top_k = np.arange (1, 51, 1)
#top_k = [1,5,10]


all_score = []
process_time=[]
for i in range(len(top_k)):
    model_metrics = [metrics.CategoricalAccuracy(), metrics.TopKCategoricalAccuracy(k=top_k[i])]
    model_loss = losses.CategoricalCrossentropy()
    model.compile(loss=model_loss, optimizer = optim, metrics=model_metrics)
    model.load_weights(BEST_WEIGTHS)

    tic ()
    out = model.evaluate(x=X_data, y=index_true, verbose=1)
    delta_time = toc ()
    all_score.append(out[2])
    process_time.append (delta_time)

all_index_predict = model.predict(X_data, verbose=1)
all_index_predict_order = np.zeros ((all_index_predict.shape[0], all_index_predict.shape [1]))
for i in range (len (all_index_predict)):
    all_index_predict_order[i] = np.flip(np.argsort(all_index_predict[i]))

path_index_predict = '../../results/index_beams_predict/ruseckas/top_k/'+flag_file+'/'
file_name = 'index_beams_predict_top_k.npz'
npz_index_predict = path_index_predict + file_name
np.savez (npz_index_predict, index_predict=all_index_predict_order)

## Testanto  a acuracia calculada pelo metodo de avaliacao do keras (evaluate)
top_1_predict = all_index_predict_order[:,0].astype(int)
true_label = []
for i in range(len(index_true)):
    true_label.append(index_true[i,:].argmax())


acerto = 0
nao_acerto = 0

for sample in range(len(index_true)):
    if (true_label[sample] == top_1_predict[sample]):
        acerto = acerto + 1
    else:
        nao_acerto = nao_acerto + 1

score= acerto / len(all_index_predict)

#path_score = '../../results/accuracy/8x32/accuracy_new_labels/'+flag_file+'/'
path_score = '../../results/score/ruseckas/top_k/'+flag_file+'/'
file_name = 'score_'+flag_file+'_top_k.csv'
df_acuracia_top_k = pd.DataFrame({"Top-K": top_k, "Acuracia": all_score})
df_acuracia_top_k.to_csv(path_score+file_name, index=False)

path_to_save_process_time = '../../results/processingTime/ruseckas/'+flag_file+'/'
df_test_time = pd.DataFrame({"test_time": process_time})
df_test_time.to_csv(path_to_save_process_time + 'test_time_' + flag_file +'.csv', index=False)

print(all_score)