from matplotlib import pyplot as plt

col_name = ['LSTM', 'GRU', 'TCN', 'Heuristic-stride-TCN', 'Stride-TCN-2layers', 'Stride-TCN-3layers',
            'Stride-TCN-4layers']


def export_mse_mae_from_txt(pth):
    error_singles = []
    error_multis = []
    with open(pth, 'r') as f:
        data = f.readlines()
    for line in data:
        break_line = line.split(" ")
        error1 = float(break_line[2])
        error2 = float(break_line[5])
        error_singles.append(error1)
        error_multis.append(error2)
    return error_singles, error_multis


def plot_data_error():
    fig, ax = plt.subplots()

    ax.plot(list(range(1, len(error_singles) + 1)), error_singles,
            marker='.', linestyle='-', linewidth=0.5, label='error_singles')

    ax.plot(list(range(1, len(error_singles) + 1)), error_multis,
            marker='o', markersize=8, linestyle='-', label='error_multis')
    ax.set_ylabel("MSE")
    ax.set_xlabel("Days")

    ax.legend()
    plt.show()


def plot_error(error_df, y_label="MSE"):
    fig, ax = plt.subplots()

    for name in col_name:
        ax.plot(list(range(1, len(error_df.loc[name].to_numpy()) + 1)), error_df.loc[name].to_numpy(),
                marker='.', linestyle='-', linewidth=0.5, label=name)

    ax.set_ylabel(y_label)
    ax.set_xlabel("Forecasting Horizon")

    ax.legend()
    plt.show()


def read_csv_data():
    import pandas as pd
    df = pd.read_csv("show_compare_errors_datasetSpain.txt", sep=",", header=None)
    time = [1, 12, 24, 36, 48, 60, 72, 84]

    mse = pd.DataFrame()
    mae = pd.DataFrame()
    for col_i, tt in zip(range(1, 17, 2), time):
        mse[tt] = df[col_i]
        mae[tt] = df[col_i]

    mse['col_name'] = col_name
    mae['col_name'] = col_name

    mse = mse.set_index('col_name')
    mae = mae.set_index('col_name')

    return mse, mae


## plot_data_error()
# error_singles, error_multis = export_mse_mae_from_txt(
#     r"\\168.131.153.57\andrew\Time Series\TSDatasets\tcn_analysis\tcn_output_SinglevsMulti.txt")
#
# plot_data_error()

mse, mae = read_csv_data()
plot_error(mse, "MSE")
plot_error(mae, "MAE")
