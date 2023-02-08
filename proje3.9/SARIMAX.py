import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler

plt.style.use('fivethirtyeight')

df = pd.read_csv('NVDA.csv')

closePrices = df['Close'].to_numpy()

# show closing prices with graphic
df.set_index("Date").Close.plot(linewidth=1, figsize=(14, 7), title="NVIDIA Close Price", color='green')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.show()

df['Mean'] = (df['Low'] + df['High']) / 2

# Creating a copy for making small changes
dataset_for_prediction = df.copy()
dataset_for_prediction['Actual'] = dataset_for_prediction['Mean'].shift()
# print(dataset_for_prediction['Actual'].tail(10))
dataset_for_prediction = dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']
# print(dataset_for_prediction.index)

sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['Low', 'High', 'Open', 'Close', 'Volume', 'Mean']])
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X = scaled_input
X.rename(columns={0: 'Low', 1: 'High', 2: 'Open', 3: 'Close', 4: 'Volume', 5: 'Mean'}, inplace=True)
print("Normalized X")
print(X.head())

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output = pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y = scaler_output
y.rename(columns={0: 'BTC Price next day'}, inplace=True)
y.index = dataset_for_prediction.index
print("Normalized y")
print(y.head())

train_size = int(len(df) * 0.9)  # 3851
test_size = int(len(df)) - train_size  # 428
train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()  # 0,3851  ,
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()

model = SARIMAX(
    train_y,
    exog=train_X,
    order=(0, 1, 1),
    seasonal_order=(0, 0, 1, 12),
    enforce_invertibility=False,
    enforce_stationarity=False
)

results = model.fit()

print(test_X.shape)
print(train_y.shape)

predictions = results.predict(start=train_size, end=train_size + test_size - 2 , exog=test_X)

act = pd.DataFrame(scaler_output.iloc[train_size:, 0])
predictions = pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index = test_X.index
predictions['Actual'] = act['BTC Price next day']
predictions.rename(columns={0: 'Pred', 'predicted_mean': 'Pred'}, inplace=True)

testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

plt.figure(figsize=(20, 10))
plt.plot(predictions.index, testActual, label='Pred', color='blue' , marker='o')
plt.plot(predictions.index, testPredict, label='Actual', color='red', marker='o')
plt.legend()
plt.show()

print("RMSE:", rmse(testActual, testPredict))

"""


Görünüşe göre end=train_size + test_size - 2 içindeki -2, test kümesinin son
 iki satırını tahminlerden çıkarmak için kullanılıyor. Bunun nedeni, SARIMAX modelinin tahmin 
 yönteminin zaman serisindeki sonraki n adım için tahminler döndürmesi ve end parametresinin tahminlerin yapılacağı 
 son zaman adımını belirtmesidir. train_size + test_size değeri, test setinin 
son iki satırını içereceğinden, bu değerden 2 çıkarmak, bu son iki satırı etkili bir şekilde tahminlerden hariç tutar.

"""


"""
he -2 in the predictions line of code you provided comes from the line end=train_size + test_size - 2. This line sets the end of the prediction period to be the sum of the sizes of the training and test sets, minus 2.

In this particular case, it seems that the purpose of subtracting 2 is to exclude the first two predictions from the results. This is because the start parameter is set to train_size, which will start the prediction period at the end of the training set. By subtracting 2 from the end of the prediction period, the first two predictions are not included in the results.

It's worth noting that this code is using the SARIMAX model from the statsmodels library to fit the data and make predictions. The SARIMAX model is a seasonal autoregressive integrated moving average with exogenous regressors model, which can be used for time series forecasting with additional external variables (i.e., the exog parameter). The parameters passed to the SARIMAX model specify the order of the autoregressive (AR) and moving average (MA) components, as well as the order of the seasonal components of the model. The enforce_invertibility and enforce_stationarity parameters are used to specify whether the model should enforce invertibility and stationarity, respectively."""