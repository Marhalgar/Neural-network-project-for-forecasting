import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#данные за период  с 2023 по 2025

years = pd.date_range(start= '2023-12-31', end= '2025-12-31', freq='YE')

data = {'Активы' : np.array([12349619, 19172166, 24449482]),
        'Обязательства' : np.array([7591063,  11007640, 12491234]),
        'Нераспределенная_прибыль' : np.array([4220950, 7466783, 11175519]),
        'Собственный_капитал' : np.array([500, 432, 436])}

Чистая_прибыль = np.array([3638324, 6040730, 6049220])

Оборотные_активы = np.array([7461218, 9890598, 12239024])

Краткосрочные_обязательства = np.array([4220950, 7088326, 9629969])

Налог_на_прибыль = np.array([2536, 132578, 393148])

Процентные_расходы  = np.array([42033, 111960, 612993])

Процентные_доходы = np.array([163445, 520392, 307953])


data_main = pd.DataFrame(data).set_index(years)

#блок с расчетами основных показателей

data_main['Чистый оборотный капитал'] = Оборотные_активы - Краткосрочные_обязательства
data_main['EBIT'] = Чистая_прибыль + Налог_на_прибыль + Процентные_расходы - Процентные_доходы

data_main['X_1'] = data_main['Чистый оборотный капитал'] / data_main['Активы']
data_main['X_2'] = data_main['Нераспределенная_прибыль'] / data_main['Активы']
data_main['X_3'] = data_main['EBIT'] / data_main['Активы']
data_main['X_4'] = data_main['Собственный_капитал'] / data_main['Обязательства']

data_update = (6.56 * data_main['X_1'] + 3.26 * data_main['X_2'] + 6.72* data_main['X_3'] + 1.05 * data_main['X_4']).values.reshape(-1, 1)

#перевод данных в иной формат для обучения нейросети

data_1 = MinMaxScaler()
data_3 = data_1.fit_transform(data_update)

def create_sequences(data, time_steps):
    Data_update_1, Data_update_3 = [], []
    for i in range(len(data) - time_steps):
        Data_update_1.append(data[i:i+time_steps])
        Data_update_3.append(data[i+time_steps, 0])
    return np.array(Data_update_1), np.array(Data_update_3)

time_steps = 1
Data_update_1, Data_update_3 = create_sequences(data_3, time_steps)

#создание модели нейросети и цикла обучения

AI_company_model = Sequential([
LSTM(35, activation= 'relu', return_sequences=True, input_shape=(time_steps, 1)),
Dropout(0.2),
LSTM(35, activation='relu'),
Dropout(0.2),
Dense(1)])

AI_company_model.compile(optimizer='adam', loss='mse')

AI_company_model.fit(Data_update_1, Data_update_3, epochs = 200, verbose=0)

A = AI_company_model.predict(Data_update_1)

A_3 = data_1.inverse_transform(A)

print(f'Прогноз итогового значения по моделе Альтмана: {A_3}')

#формирование прогнозов на период с 2026 по 2028 гг.

forecast_steps = 3
forecast = []

current_batch = data_3[-time_steps:].reshape(1, time_steps, 1)

for i in range(forecast_steps):
    pr = AI_company_model.predict(current_batch, verbose=0)[0][0]
    forecast.append(pr)
    new_val = np.array([[[pr]]])
    current_batch = np.append(current_batch[:, 1:, :], new_val, axis=1)


forecast = data_1.inverse_transform(np.array(forecast).reshape(-1, 1))

#создание графика с исходными данными и прогнозами

plt.figure(figsize=(10, 6))
plt.plot(data_main.index[-len(Data_update_1):], data_1.inverse_transform(A), label='Основные данные', color='blue', linestyle='solid')
plt.plot(pd.date_range(start=data_main.index[-1], periods=forecast_steps, freq='Y'), forecast, label='Прогноз', color='orange', linestyle='dashdot')
plt.title('Прогноз по изменению итогового значения модели Альтмана для «Группы Астра»')
plt.legend()
plt.show()
