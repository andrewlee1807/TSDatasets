"""
Measure the number of parameters after training models : autoML, LSTM, GRU
on 3 datasets about energy consumptions
- CNU
- household
- Spain
"""

from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import os
import glob

keywords = "Total params"
list_dataset = ['household', 'spain', 'cnu']
markers = ['.', 'o', '*', '+', 'x', '^', "v", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", ",", "h", "H", "X", "D",
           "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
           ]


def export_keywords_value_from_txt(pth):
    with open(pth, 'r') as f:
        lines = f.readlines()

    find_params_line = lambda x: keywords in lines[x]
    params_line_index = list(filter(find_params_line, range(len(lines) - 1, -1, -1)))[0]
    params_value = float(lines[params_line_index].split(' ')[-1].replace('\n', "").replace(",", ""))
    return params_value


def list_params_num(listdir):
    list_params_value = []
    for filename in listdir:
        params_value = export_keywords_value_from_txt(filename)
        list_params_value.append(params_value)
    return list_params_value


def plot_num_param_on_a_dataset(dataset_order, dict_method_params):
    """
    :param dataset_order: {0,1,2}
    :param dict_method_params: {"gru":123, "lstm":211}
    :return:
    """
    fig, ax = plt.subplots()
    # Avoid others experiments are not finished yet
    length_min = min(len(dict_method_params[i]) for i in dict_method_params)

    for name_method, m in zip(dict_method_params, markers):
        data_plot = dict_method_params[name_method][:length_min]
        ax.plot(list(range(1, length_min + 1)), data_plot,
                marker=m, linestyle='-', linewidth=0.5, label=name_method)

    ax.set_ylabel("Number of parameters")
    ax.set_xlabel("Hours")
    ax.set_title(f"Dataset {list_dataset[dataset_order]}")
    ax.legend()
    plt.savefig(f"Number of Params compare on {list_dataset[dataset_order]}.png", dpi=120)
    plt.show()
    plt.clf()


def compare_delayNet_result():
    # list_dataset = ['household', 'spain', 'cnu']
    num_dataset_observation = 0
    dict_method_params = dict()
    # Fedot
    # import numpy as np
    # fedot_errs = np.loadtxt("automl_searching/fedot_err.txt", dtype=float)
    # dict_method_error["fedot"] = fedot_errs.transpose()

    # TCN auto-generated search
    path_folder2 = f"automl_searching/{list_dataset[num_dataset_observation]}_result_auto/*.txt"
    listdir = glob.glob(path_folder2)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    list_params_value_our = list_params_num(listdir)
    dict_method_params["auto-tcn"] = list_params_value_our

    # AUTO-CORRELATION
    correlation_auto_pth = f"auto_correlation/{list_dataset[num_dataset_observation]}_auto/*.txt"
    listdir = glob.glob(correlation_auto_pth)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    list_params_value_correlation = list_params_num(listdir)
    dict_method_params["auto-stride_dilated_net"] = list_params_value_correlation

    plot_num_param_on_a_dataset(num_dataset_observation, dict_method_params)


def compare_all_datasets():
    for dataset_order in range(0, len(list_dataset)):
        dict_method_params = dict()
        #  GRU
        path_folder1 = f"automl_searching/{list_dataset[dataset_order]}_result_gru/*.txt"
        listdir = glob.glob(path_folder1)
        listdir.sort(key=lambda x: os.path.getmtime(x))
        list_params_value_gru = list_params_num(listdir)
        dict_method_params["gru"] = list_params_value_gru

        # LSTM
        path_folder3 = f"automl_searching/{list_dataset[dataset_order]}_result_lstm/*.txt"
        listdir = glob.glob(path_folder3)
        listdir.sort(key=lambda x: os.path.getmtime(x))
        list_params_value_lstm = list_params_num(listdir)
        dict_method_params["lstm"] = list_params_value_lstm

        # TCN auto-generated search
        path_folder2 = f"automl_searching/{list_dataset[dataset_order]}_result_auto/*.txt"
        listdir = glob.glob(path_folder2)
        listdir.sort(key=lambda x: os.path.getmtime(x))
        list_params_value_our = list_params_num(listdir)
        dict_method_params["ours"] = list_params_value_our

        plot_num_param_on_a_dataset(dataset_order, dict_method_params)


if __name__ == '__main__':
    # compare_all_datasets()
    compare_delayNet_result()
