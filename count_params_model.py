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


def export_keywords_value_from_txt(pth):
    with open(pth, 'r') as f:
        lines = f.readlines()
    
    find_params_line = lambda x: keywords in lines[x] 
    params_line_index = list(filter(find_params_line, range(len(lines)-1, -1, -1)))[0]
    params_value = float(lines[params_line_index].split(' ')[-1].replace('\n',"").replace(",",""))
    return params_value

def list_params_num(listdir):
    list_params_value = []
    for filename in  listdir:
        params_value = export_keywords_value_from_txt(filename)
        list_params_value.append(params_value)
    return list_params_value

def plot_params_value(dataset_order, our_params, gru_params, lstm_params):
    fig, ax = plt.subplots()

    length_max = max(len(lstm_params), len(our_params), len(gru_params))
    # length_max = max(len(our_params), len(gru_params))

    ax.plot(list(range(1,25)), lstm_params[:length_max],
    marker='.', linestyle='-', linewidth=0.5, label='lstm')
    
    ax.plot(list(range(1,25)), our_params[:length_max],
    marker='o', markersize=8, linestyle='-', label='our')

    ax.plot(list(range(1,25)), gru_params[:length_max],
    marker='*', markersize=8, linestyle='-', label='gru')

    ax.set_ylabel(f'Number of parameters on Dataset {dataset_order + 1} test set')

    ax.ticklabel_format(axis='y', scilimits=[-3, 3])

    ax.legend()
    plt.savefig(f"Number of Params compare on {list_dataset[dataset_order]}.png", dpi=120)
    plt.clf()

def main():
    for dataset_order in range(0, len(list_dataset)):
        #  GRU
        path_folder1 = f"automl_searching/{list_dataset[dataset_order]}_result_gru/*.txt"
        listdir = glob.glob(path_folder1)
        listdir.sort(key=lambda x: os.path.getmtime(x))
        list_params_value_gru = list_params_num(listdir)

        # LSTM
        path_folder3 = f"automl_searching/{list_dataset[dataset_order]}_result_lstm/*.txt"
        listdir = glob.glob(path_folder3)
        listdir.sort(key=lambda x: os.path.getmtime(x))
        list_params_value_lstm = list_params_num(listdir)

        # TCN auto-generated search
        path_folder2 = f"automl_searching/{list_dataset[dataset_order]}_result_auto/*.txt"
        listdir = glob.glob(path_folder2)
        listdir.sort(key=lambda x: os.path.getmtime(x))
        list_params_value_our = list_params_num(listdir)

        plot_params_value(dataset_order, list_params_value_our, list_params_value_gru, list_params_value_lstm)

if __name__ == '__main__':
    main()
