from fedot.api.main import Fedot
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
import numpy as np
import sys

result_path = ""
history_len = 300  # previous data
list_mse = []
# len_forecast = 24 # prediction

orig_stdout = sys.stdout
f = open(f'fedot_process_log.txt', 'w')
sys.stdout = f

for len_forecast in range(1, 25):
    list_mse_temp = []
    list_mae_temp = []
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(train_data_np)),
                            features=train_data_np,
                            target=train_data_np,
                            task=task,
                            data_type=DataTypesEnum.ts)

    #  Training
    model = Fedot(problem='ts_forecasting', task_params=task.task_params)
    chain = model.fit(train_input)

    for start_test in range(idx_max, len(raw_seq) - len_forecast):
        # Prepare data to train the model
        test_input = InputData(idx=np.arange(start_test, start_test + len_forecast),
                               features=raw_seq[start_test - idx_max: start_test],  # using long sequence
                               target=test_data_np[start_test - idx_max: start_test - idx_max + len_forecast],
                               task=task,
                               data_type=DataTypesEnum.ts)
        # print(test_input)
        # use model to obtain forecast
        forecast = model.predict(test_input)
        target = np.ravel(test_input.target)
        t = model.get_metrics(metric_names=['mse', 'mae'], target=target)
        list_mse_temp.append(t['mse'])
        list_mae_temp.append(t['mae'])

    # plot forecasting result
    # model.plot_prediction()
    del model, train_input, test_input

    list_mse_temp = np.asarray(list_mse_temp)
    print(len_forecast, ":", list_mse_temp.mean(), list_mae_temp.mean())
    list_mse.append(list_mse_temp)
    list_mae.append(list_mae_temp)

print("MSE:")
print(list_mse)
print("MAE:")
print(list_mae)

sys.stdout = orig_stdout
f.close()
