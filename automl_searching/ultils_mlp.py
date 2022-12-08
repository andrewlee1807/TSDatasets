import pandas as pd
import numpy as np
# List dataset name
cnu_str = "CNU"
comed_str = "COMED"
spain_str = "SPAIN"
household_str = "HOUSEHOLD"

# Dataset path
CONFIG_PATH = {
    cnu_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/%EA%B3%B5%EB%8C%807%ED%98%B8%EA%B4%80_HV_02.csv",
    comed_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/COMED_hourly.csv",
    spain_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/spain/spain_ec_499.csv",
    household_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/household_daily_power_consumption.csv"
}

class DataLoader:
    """
    Class to be inheritance from others dataset
    """

    def __init__(self, path_file, data_name):
        self.raw_data = None
        if path_file is None:
            self.path_file = CONFIG_PATH[data_name]
        else:
            self.path_file = path_file

    def read_data_frame(self):
        return pd.read_csv(self.path_file)

    def read_a_single_sequence(self):
        return np.loadtxt(self.path_file)


# CNU dataset
class CNU(DataLoader):
    def __init__(self, path_file=None):
        super(CNU, self).__init__(path_file, cnu_str)
        self.raw_data = self.read_a_single_sequence()

    def export_sequences(self):
        return self.raw_data  # a single sequence

# Spain dataset
class SPAIN(DataLoader):
    def __init__(self, path_file=None):
        super(SPAIN, self).__init__(path_file, spain_str)
        self.dataframe = self.read_data_frame()

    def export_sequences(self):
        # Pick the customer no 20
        return self.dataframe.iloc[:, 20]  # a single sequence

def get_all_data_supported():
    return list(CONFIG_PATH.keys())



def fill_missing(data):
    one_day = 23 * 60
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if np.isnan(data[row, col]):
                data[row, col] = data[row - one_day, col]


class HouseholdDataLoader(DataLoader):
    def __init__(self, data_path=None):
        super(HouseholdDataLoader, self).__init__(data_path, household_str)
        self.df = None
        self.data_by_days = None
        self.data_by_hour = None
        self.load_data()

    def load_data(self):

        df = pd.read_csv(self.path_file, sep=';',
                         parse_dates={'dt': ['Date', 'Time']},
                         infer_datetime_format=True,
                         low_memory=False, na_values=['nan', '?'],
                         index_col='dt')

        droping_list_all = []
        for j in range(0, 7):
            if not df.iloc[:, j].notnull().all():
                droping_list_all.append(j)
        for j in range(0, 7):
            df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].mean())

        fill_missing(df.values)

        self.df = df
        self.data_by_days = df.resample('D').sum()  # all the units of particular day
        self.data_by_hour = df.resample('H').sum()  # all the units of particular day


class Dataset:
    """
    Dataset class hold all the dataset via dataset name
    :function:
    - Load dataset
    """

    def __init__(self, dataset_name):
        dataset_name = dataset_name.upper()
        if dataset_name not in get_all_data_supported():
            raise f"Dataset name {dataset_name} isn't supported"
        self.dataset_name = dataset_name
        # DataLoader
        self.dataloader = self.__load_data()

    def __load_data(self):
        if self.dataset_name == cnu_str:
            return CNU()
        elif self.dataset_name == spain_str:
            return SPAIN()
        elif self.dataset_name == household_str:
            return HouseholdDataLoader(r'/home/andrew/household_power_consumption.txt')