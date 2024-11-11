
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Cargar el dataset
data = pd.read_csv('C:/Users/valen/OneDrive/Documentos/Universidad/IA/PROYECTO IA/3entrega/df_imputado.csv')
print(data.head())

# Preprocesar el dataset: convertir 'Fecha' a datetime y establecer como índice
data['Fecha'] = pd.to_datetime(data['Fecha'], format='%d/%m/%Y %H:%M', errors='coerce')
# specify format and handle errors
data.set_index('Fecha', inplace=True)

# Normalizar los datos de generación entre -1 y 1
scaler = MinMaxScaler(feature_range=(-1, 1))
DatosM2n = scaler.fit_transform(data[['Generacion_(kWh)']])

# Función para convertir la serie en secuencias
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'var{j+1}(t-{i})') for j in range(n_vars)]

    # Forecast sequence (t, t+1, ... t+n)
    for i in range(n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(f'var{j+1}(t)') for j in range(n_vars)]
        else:
            names += [(f'var{j+1}(t+{i})') for j in range(n_vars)]

    # Combinar todo
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # Eliminar filas con NaN
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Definir longitudes de secuencia
seq_length = 182
out_length = 13

data_train_LSTM = series_to_supervised(DatosM2n, seq_length, out_length)
X_train_LSTM = data_train_LSTM.values[:, 0:seq_length]
y_train_LSTM = data_train_LSTM.values[:, seq_length:]

X_train_LSTM = X_train_LSTM.reshape(X_train_LSTM.shape[0], X_train_LSTM.shape[1], 1)

"""# LSTM model creation"""

modelLSTM = Sequential()
modelLSTM.add(LSTM(64, activation='tanh', input_shape=(seq_length, 1), return_sequences=True))
modelLSTM.add(LSTM(45, activation='tanh', return_sequences=True))
modelLSTM.add(LSTM(32, activation='tanh'))
modelLSTM.add(Dense(out_length))  # La salida debe ser el número de pasos a predecir
modelLSTM.compile(optimizer='adam', loss='mse')

modelLSTM.summary()

callbacksLSTM = [
    EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('best_modelLSTM.keras', monitor='val_loss', save_best_only=True, verbose=1)]

"""# Model training"""

modelLSTM.fit(X_train_LSTM, y_train_LSTM, epochs=30,  verbose=1, callbacks=callbacksLSTM, validation_split=0.2)

loss_per_epoch = modelLSTM.history.history['loss']
val_loss_per_epoch = modelLSTM.history.history['val_loss']

plt.plot(range(len(loss_per_epoch)), loss_per_epoch, label='Training Loss')
plt.plot(range(len(val_loss_per_epoch)), val_loss_per_epoch, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()

"""# Model's Architecture"""

# Define the filename for the model image
model_image_filename = 'model_LSTM.png'

model_plot = tf.keras.utils.plot_model(modelLSTM, to_file=model_image_filename, show_shapes=True)
print(f"Model image saved as {model_image_filename}")

"""# Predict the model"""

# load the best model
modelLSTM.load_weights('best_modelLSTM.keras')

pred_LSTM = scaler.inverse_transform(modelLSTM.predict(X_train_LSTM))
real_data = scaler.inverse_transform(y_train_LSTM)

"""# Estimate Errors"""

mse = mean_squared_error(real_data, pred_LSTM)
rmse = np.sqrt(mse)
p_rmse = (rmse / 250) * 100
mae = mean_absolute_error(real_data, pred_LSTM)
p_mae = (mae / 250) * 100
r2 = r2_score(real_data, pred_LSTM)


print(f"{'Metric':<40} {'Value':>20}")
print("=" * 60)
print(f"{'Root Mean Square Error:':<40} {rmse:>20.2f}")
print(f"{'Percentage Root Mean Square Error:':<40} {p_rmse:>20.2f}%")
print(f"{'Mean Absolute Error:':<40} {mae:>20.2f}")
print(f"{'Percentage Mean Absolute Error:':<40} {p_mae:>20.2f}%")
print(f"{'MSE:':<40} {mse:>20.2f}")
print(f"{'R2 Score:':<40} {r2:>20.4f}")

"""# LSTM model predictions"""

datos2 = np.arange(0, real_data.shape[0])

plt.figure(figsize=(16, 6))
plt.plot(datos2, real_data[:, 0], label='Generación de energía real', color='blue', linewidth=2, linestyle='-')
plt.plot(datos2, pred_LSTM[:, 0], label='Generación de energía predicha', color='pink', linewidth=2, linestyle='-')

plt.title('Predicciones del modelo LSTM', fontsize=16)
plt.xlabel('Tiempo (hora)', fontsize=11)
plt.ylabel('Generación (kWh)', fontsize=11)
plt.legend(fontsize=11, loc='upper right')
plt.grid(color='lightgray', linestyle='--', linewidth=0.7)
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.xticks(fontsize=11, color='black')
plt.yticks(fontsize=11, color='black')
plt.tight_layout()
plt.show()

modelLSTM.save('/content/modelo_lstm.h5')