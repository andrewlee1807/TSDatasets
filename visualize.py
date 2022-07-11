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
    for filename in  listdir:
        mse, mae = export_mse_mae_from_txt(filename)
        list_mae.append(mae)
        list_mse.append(mse)
    return list_mse, list_mae

def plot_data_error(type_display, dataset_order, lstm_err, our_err, gru_err):
    fig, ax = plt.subplots()
    length_max = max(len(lstm_err), len(our_err), len(gru_err))
    ax.plot(list(range(1,25)), lstm_err[:length_max],
    marker='.', linestyle='-', linewidth=0.5, label='lstm')

    ax.plot(list(range(1,25)), our_err[:length_max],
    marker='o', markersize=8, linestyle='-', label='our')

    ax.plot(list(range(1,25)), gru_err[:length_max],
    marker='*', markersize=8, linestyle='-', label='gru')
    
    ax.set_ylabel(type_display + f" on Dataset {dataset_order + 1} test set")
    ax.legend()
    plt.savefig(type_display + f" {list_dataset[dataset_order]}.png", dpi=120)
    plt.clf()


def main():
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


if __name__ == '__main__':
    main()

