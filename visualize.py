from operator import le, length_hint
from matplotlib import pyplot as plt
import os
import glob

def export_mse_mae_from_txt(pth):
    # pth = "/home/andrew/Time Series/TSDatasets/automl_searching/cnu_result_lstm/seaching_process_log_1.txt"
    with open(pth, 'r') as f:
        last_line = f.readlines()[-1]
    last_line = last_line.split(" ")
    find_mse_index = lambda x: "mse" in last_line[x] 
    find_mae_index = lambda x: "mae" in last_line[x] 

    mse_index = list(filter(find_mse_index, range(len(last_line))))[0] + 1
    mae_index = list(filter(find_mae_index, range(len(last_line))))[0] + 1
    mae = float(last_line[mae_index])
    mse = float(last_line[mse_index])
    return mse, mae

def list_error(listdir):
    list_mse = []
    list_mae = []
    for filename in  listdir:
        mse, mae = export_mse_mae_from_txt(filename)
        list_mae.append(mae)
        list_mse.append(mse)
    return list_mse, list_mae

def plot_data_error(type_display, lstm_err, our_err):
    fig, ax = plt.subplots()
    length_max = max(len(lstm_err), len(our_err))
    ax.plot(list(range(1,25)), lstm_err[:length_max],
    marker='.', linestyle='-', linewidth=0.5, label='lstm')
    ax.plot(list(range(1,25)), our_err[:length_max],
    marker='o', markersize=8, linestyle='-', label='our')
    ax.set_ylabel(type_display + ' on Test dataset')
    ax.legend()
    plt.savefig(type_display + " test.png", dpi=120)
    plt.clf()

import time
t1 = time.time()
path_folder1 = "automl_searching/household_result_lstm/*.txt"
listdir = glob.glob(path_folder1)
listdir.sort(key=lambda x: os.path.getmtime(x))
lstm_errs = list_error(listdir)
print(time.time() - t1)

path_folder2 = "automl_searching/household_result_auto/*.txt"
listdir = glob.glob(path_folder2)
listdir.sort(key=lambda x: os.path.getmtime(x))
our_errs = list_error(listdir)

t1 = time.time()
plot_data_error("MSE", lstm_errs[0], our_errs[0])
print(time.time() - t1)
plot_data_error("MAE", lstm_errs[1], our_errs[1])


