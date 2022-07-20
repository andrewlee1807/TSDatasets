"""
Plot MSE and MAE after training models : autoML, LSTM, GRU
on 3 datasets about energy consumptions
- CNU
- household
- Spain
"""
from matplotlib import pyplot as plt
import os
import glob

list_dataset = ['household', 'spain', 'cnu']
markers = ['.', 'o', '*', '+', 'x', '^', "v", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", ",", "h", "H", "X", "D",
           "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
           ]
type_displays = ["MSE", "MAE"]


def export_mse_mae_from_txt(pth):
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
    for filename in listdir:
        mse, mae = export_mse_mae_from_txt(filename)
        list_mae.append(mae)
        list_mse.append(mse)
    return list_mse, list_mae


def plot_data_error(type_display, dataset_order, lstm_err, our_err, gru_err):
    fig, ax = plt.subplots()
    length_max = max(len(lstm_err), len(our_err), len(gru_err))
    ax.plot(list(range(1, 25)), lstm_err[:length_max],
            marker='.', linestyle='-', linewidth=0.5, label='lstm')

    ax.plot(list(range(1, 25)), our_err[:length_max],
            marker='o', markersize=8, linestyle='-', label='our')

    ax.plot(list(range(1, 25)), gru_err[:length_max],
            marker='*', markersize=8, linestyle='-', label='gru')

    ax.set_ylabel(type_display + f" on Dataset {dataset_order + 1} test set")
    ax.legend()
    plt.savefig(type_display + f" {list_dataset[dataset_order]}.png", dpi=120)
    plt.clf()


def compare_tcn_auto():
    for dataset_order in range(0, len(list_dataset)):
        # LSTM
        path_folder1 = f"automl_searching/{list_dataset[dataset_order]}_result_lstm/*.txt"
        listdir = glob.glob(path_folder1)
        listdir.sort(key=lambda x: os.path.getmtime(x))
        lstm_errs = list_error(listdir)

        # TCN auto-generated search
        path_folder2 = f"automl_searching/{list_dataset[dataset_order]}_result_auto/*.txt"
        listdir = glob.glob(path_folder2)
        listdir.sort(key=lambda x: os.path.getmtime(x))
        our_errs = list_error(listdir)

        #  GRU
        path_folder3 = f"automl_searching/{list_dataset[dataset_order]}_result_gru/*.txt"
        listdir = glob.glob(path_folder3)
        listdir.sort(key=lambda x: os.path.getmtime(x))
        gru_errs = list_error(listdir)

        plot_data_error("MSE", dataset_order, lstm_errs[0], our_errs[0], gru_errs[0])
        plot_data_error("MAE", dataset_order, lstm_errs[1], our_errs[1], gru_errs[1])


def plot_data_errors_on_a_dataset(num_type_display, dataset_order, dict_method_error):
    """
    :param type_display: "MSE" or "MAE"
    :param dataset_order: {0,1,2}
    :param dict_method_error: {"gru":123, "lstm":211}
    :return:
    """
    fig, ax = plt.subplots()
    # Avoid others experiments are not finished yet
    length_min = min(len(dict_method_error[i][num_type_display]) for i in dict_method_error)
    for name_method, m in zip(dict_method_error, markers):
        data_plot = dict_method_error[name_method][num_type_display][:length_min]
        ax.plot(list(range(1, length_min + 1)), data_plot,
                marker=m, linestyle='-', linewidth=0.5, label=name_method)

    ax.set_ylabel(type_displays[num_type_display])
    ax.set_xlabel("Hours")
    ax.set_title(f"Dataset {dataset_order + 1}")
    ax.legend()
    plt.savefig(type_displays[num_type_display] + f" {list_dataset[dataset_order]}.png", dpi=120)
    plt.clf()


def compare_delayNet_result():
    dict_method_error = dict()
    # Fedot
    # import numpy as np
    # fedot_errs = np.loadtxt("automl_searching/fedot_err.txt", dtype=float)
    # dict_method_error["fedot"] = fedot_errs.transpose()

    # DelayedNet
    # path_folder = f'auto_correlation/cnu_result/T100_kernal128/*.txt'
    path_folder = f'auto_correlation/cnu_result/T100_kernal32/*.txt'
    listdir = glob.glob(path_folder)
    # print(listdir)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    delay_errs = list_error(listdir)
    dict_method_error["stride_dilated_net"] = delay_errs

    path_folder1 = f"automl_searching/{list_dataset[2]}_result_lstm/*.txt"
    listdir = glob.glob(path_folder1)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    lstm_errs = list_error(listdir)
    dict_method_error["lstm"] = lstm_errs

    # TCN auto-generated search
    path_folder2 = f"automl_searching/{list_dataset[2]}_result_auto/*.txt"
    listdir = glob.glob(path_folder2)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    our_errs = list_error(listdir)
    dict_method_error["auto-tcn"] = our_errs

    #  GRU
    path_folder3 = f"automl_searching/{list_dataset[2]}_result_gru/*.txt"
    listdir = glob.glob(path_folder3)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    gru_errs = list_error(listdir)
    dict_method_error["gru"] = gru_errs

    # AUTO-CORRELATION
    cnu_auto_pth = "auto_correlation/cnu_auto/*.txt"
    listdir = glob.glob(cnu_auto_pth)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    correlation_errs = list_error(listdir)
    dict_method_error["auto-stride_dilated_net"] = correlation_errs

    plot_data_errors_on_a_dataset(0, 2, dict_method_error)


def compare_auto_correlation():
    dict_method_error = dict()


    # TCN auto-generated search
    path_folder2 = f"automl_searching/{list_dataset[2]}_result_auto/*.txt"
    listdir = glob.glob(path_folder2)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    our_errs = list_error(listdir)
    dict_method_error["auto-tcn"] = our_errs

    # AUTO-CORRELATION
    cnu_auto_pth = "auto_correlation/cnu_auto/*.txt"
    listdir = glob.glob(cnu_auto_pth)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    correlation_errs = list_error(listdir)
    dict_method_error["auto-correlation"] = correlation_errs

    plot_data_errors_on_a_dataset(0, 2, dict_method_error)


if __name__ == '__main__':
    # compare_tcn_auto()
    # compare_delayNet_result()
    compare_auto_correlation()
