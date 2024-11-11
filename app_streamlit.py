import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Cargar el modelo LSTM
model = load_model('best_modelLSTM.keras')

# Configurar escalador para normalización
scaler = MinMaxScaler(feature_range=(-1, 1))

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """Convierte una serie en un DataFrame supervisado"""
    df = pd.DataFrame(data)
    cols, names = [], []
    
    # Secuencia de entrada (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'var1(t-{i})')]
    
    # Secuencia de salida (t, t+1, ... t+n)
    for i in range(n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += ['var1(t)']
        else:
            names += [(f'var1(t+{i})')]
    
    # Combinar y eliminar filas con NaN
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def prepare_data(data, window_size, forecast_horizon):
    # Convertir la columna de fecha y hora (si existe) y establecer como índice
    if 'Fecha' in data.columns:
        data['Fecha'] = pd.to_datetime(data['Fecha'], format='%d/%m/%Y %H:%M', errors='coerce')
        data.set_index('Fecha', inplace=True)
    
    # Seleccionar solo la columna de generación de energía y normalizarla
    energy_data = data[['Generacion_(kWh)']]
    energy_data_scaled = scaler.fit_transform(energy_data)
    
    # Estructurar en secuencias de acuerdo a `window_size` y `forecast_horizon`
    data_supervised = series_to_supervised(energy_data_scaled, window_size, forecast_horizon)
    
    # Seleccionar solo la última ventana como entrada para predecir los próximos pasos
    X = data_supervised.values[:, :window_size]
    return np.array(X[-1:]).reshape(1, window_size, 1)

# Configuración de la aplicación de Streamlit
st.title("Predicción de Generación de Energía Solar")

# Cargar los datos
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV con datos de entrada", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Mostrar datos cargados
    if st.checkbox("Ver datos originales"):
        st.write(data)

    # Configuración de parámetros de predicción
    window_size = st.sidebar.slider("Tamaño de la ventana de entrada (horas)", 24, 182, step=12)
    forecast_horizon = st.sidebar.slider("Horizonte de predicción (horas)", 1, 24)

    # Preprocesar los datos y realizar la predicción
    input_data = prepare_data(data, window_size, forecast_horizon)
    predicted_energy = model.predict(input_data)
    predicted_energy = scaler.inverse_transform(predicted_energy)  # Inversión de la normalización para obtener valores originales

    # Mostrar predicción
    st.subheader("Predicción de Energía Solar")
    st.write("Predicción para las próximas horas:")
    st.write(predicted_energy[0])

    # Gráfico de la predicción
    st.subheader("Gráfico de la Predicción de Generación de Energía Solar")
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_energy[0], label="Predicción")
    plt.xlabel("Paso")
    plt.ylabel("Generación de energía (kWh)")
    plt.legend()
    st.pyplot(plt)

else:
    st.write("Por favor sube un archivo CSV con los datos para realizar la predicción.")
