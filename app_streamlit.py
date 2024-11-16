import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit as st  
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go

import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
    <style>
        /* Establecer el fondo gris claro para toda la página */
        body {
            background-color: #dcd3d3;  /* Color de fondo */
            color: #000000;  /* Color de texto */
        }

        /* Estilo del título */
        h1 {
            font-size: 45px;  /* Tamaño de fuente */
            font-weight: bold;  /* Negrita */
            color: #2a5d84;  /* Color azul oscuro */
            text-align: center;  /* Centrado del texto */
            margin-top: 50px;  /* Espacio superior */
            margin-bottom: 20px;  /* Espacio inferior */
            font-family: 'Arial', sans-serif;  /* Fuente moderna */
        }

        /* Estilo para ubicar las pestañas a la derecha */
        .stTabs [role="tablist"] {
            display: flex;
            justify-content: flex-end;  /* Alinea las pestañas a la derecha */
        }

        /* Ajustes de contenedor de la página */
        .reportview-container {
            margin-left: 0;
            margin-right: 0;
        }

        /* Ajuste para el contenedor de los bloques */
        .block-container {
            padding: 0 2rem;
            max-width: 100%;
        }

        /* Asegura que las imágenes ocupen el ancho disponible y se centren */
        img {
            width: 90%;
            height: auto;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    <h1>Pronóstico de Generación de Energía Fotovoltaica De La Universidad Autónoma de Occidente Usando Modelos de IA</h1>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Datos Recolectados", "Modelo", "Predicción", "Referencias"])

with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
            <div style="background-color:#508991; padding: 20px; border-radius: 10px;">
                <h2 style="color:white;">Introducción</h2>
                <p style="color:white;">
                    Debido al cambio climático, se han implementado medidas globales para promover la generación limpia de electricidad. 
                    En Colombia, la Ley 99 de 1993 busca reemplazar recursos no renovables con tecnologías de energía no contaminante. 
                    La energía fotovoltaica juega un papel clave en esta mitigación, siendo más rentable y ambientalmente sostenible que 
                    los combustibles fósiles. Se estima que para 2030, el 10% del consumo energético en Colombia provendrá de proyectos fotovoltaicos. 
                    La Universidad Autónoma de Occidente ha implementado un programa de campus sostenible, destacándose en el ranking Green Metric 
                    por su uso de fuentes de energía alternativas. Además, cuenta con el sistema fotovoltaico más grande del país, compuesto 
                    por 1,632 paneles solares que aportan aproximadamente el 18% de la energía del campus, aunque la producción real depende 
                    de factores climáticos. Las estaciones meteorológicas de Celsia proporcionan los datos necesarios para predecir la energía generada, 
                    considerando variables como temperatura, irradiancia, humedad, y viento.
                </p>
            </div>
    """, unsafe_allow_html=True)
    
    with col2:
        st.image("images/paneles.jpg", caption="Paneles Solares")
        st.markdown("""
            <style>
                .stText {
                    color: white;
                }
                .css-ffhzg2 {
                    background-color: #4B8E8D;
                }
            </style>
        """, unsafe_allow_html=True)

    col4, col3 = st.columns([2, 3])
    with col4:
        st.image("images/solares.jpg", caption="Paneles Solares Fotovoltaicos")
        st.markdown("""
            <style>
                .stText {
                    color: white;
                }
                .css-ffhzg2 {
                    background-color: #4B8E8D;
                }
            </style>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px;">
                <h2 style="color:white;">Descripción del Problema</h2>
                <p style="color:white;">
                    La Universidad Autónoma de Occidente, líder en generación de energía solar en Colombia, enfrenta el reto 
                    de predecir con precisión la energía generada por su sistema fotovoltaico debido a factores climáticos variables, 
                    como la radiación solar, temperatura y humedad, así como la degradación de los paneles solares. Esta falta de pronóstico 
                    preciso limita la optimización de la energía renovable y complica la planificación a largo plazo de la universidad. 
                    El sistema fotovoltaico de la universidad está compuesto por 1.632 paneles solares que abastecen el 15% de su consumo, 
                    generando alrededor de 43.292 kWh mensuales y evitando la emisión de gases de efecto invernadero. Además, la universidad 
                    cuenta con certificación internacional por su uso de energía renovable.
                </p>
            </div>
    """, unsafe_allow_html=True)

    col5, col6 = st.columns([3, 2])
    with col5:
        st.markdown("""
            <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px;">
                <h2 style="color:white;">Objetivos</h2>
                <p style="color:white;">
                    El objetivo de este proyecto es implementar un modelo de inteligencia artificial 
                    para pronosticar la generación de energía fotovoltaica utilizando los datos recolectados 
                    en el campus de la Universidad Autónoma de Occidente. Para ello, se preparará el dataset 
                    mediante la limpieza y organización de los datos, seleccionando los modelos de IA más 
                    adecuados según el tipo de pronóstico que se desee realizar. A continuación, se implementarán 
                    y compararán diferentes modelos de IA para determinar cuál ofrece el mejor desempeño, y finalmente, 
                    se validará el modelo seleccionado utilizando un conjunto de datos de prueba para asegurar su 
                    precisión y robustez en las predicciones futuras.
                </p>
            </div>
    """, unsafe_allow_html=True)

    with col6:
        st.image("images/panel.jpg", caption="Predicción Paneles Solares UAO")

        st.markdown("""
            <style>
                .stText {
                    color: white;
                }
                .css-ffhzg2 {
                    background-color: #4B8E8D;
                }
            </style>
        """, unsafe_allow_html=True)

with tab2:
    col13, col14 = st.columns([3, 2])

    with col13:
        st.markdown("""
            <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px;">
                <h3 style="color:white;">Fuente de Datos</h3>
                <p style="color:white;">
                    Los datos provienen de los sistemas de paneles solares instalados en la <strong>Universidad Autónoma de Occidente</strong>, 
                    almacenados en la plataforma <strong>Celsia One</strong>. Estos datos corresponden al período entre el <strong>1 de mayo de 2018</strong> 
                    y el <strong>28 de febrero de 2023</strong>, con mediciones cada hora, de <strong>6:00 am a 6:00 pm</strong>.
                </p>
                <h3 style="color:white;">Características del Conjunto de Datos</h3>
                <p style="color:white;">
                    El conjunto de datos consta de dos columnas:
                </p>
                <ul style="color:white;">
                    <li><strong>Fecha</strong>: Registra la fecha y hora de la medición.</li>
                    <li><strong>Generación (kWh)</strong>: Mide la cantidad de energía generada por los paneles solares.</li>
                </ul>
                <p style="color:white;">
                    <strong>Número de registros</strong>: 25,727 filas, con <strong>2,314 valores faltantes</strong> que se imputaron utilizando técnicas de 
                    interpolación lineal y relleno hacia adelante y hacia atrás.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col14:
        st.image("images/Paneles-solares.jpg", caption="Estación Meteorológica UAO", use_container_width=True)

    col15 = st.columns(1)

    with col15[0]:
        st.markdown("""
            <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h3 style="color:white;">Análisis Exploratorio de Datos (EDA)</h3>
                <p style="color:white;">
                    La generación promedio registrada es de <strong>103.78 kWh</strong> por hora, con una 
                    <strong>desviación estándar de 83.09 kWh</strong>. El <strong>25%</strong> de las observaciones son menores a 
                    <strong>20.90 kWh</strong>, y el valor máximo alcanzó los <strong>301.96 kWh</strong>. La distribución de los 
                    datos muestra una <strong>alta variabilidad</strong> en la generación de energía a lo largo del tiempo.
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <style>
                .stText {
                    color: white;
                }
                .css-ffhzg2 {
                    background-color: #4B8E8D;
                }
            </style>
        """, unsafe_allow_html=True)

    col17, col18 = st.columns([3, 3])
    
    with col17:
        st.markdown("""
            <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px;">
                <h4 style="color:white;">Análisis Estadístico Descriptivo</h4>
                <p style="color:white;">
                    La generación promedio es de <strong>103.78 kWh</strong> por hora, con una desviación estándar de <strong>83.09 kWh</strong>, 
                    lo que indica una variabilidad significativa en la generación de energía.
                </p>
                <p style="color:white;">
                    La distribución de los datos muestra que el 25% de las observaciones están por debajo de 20.90 kWh, 
                    la mediana es 94.51 kWh, y el 75% de las observaciones están por debajo de 174.93 kWh. 
                    El valor máximo alcanzó 301.96 kWh.
                </p>
                <p style="color:white;">
                    La variabilidad y fluctuaciones muestran fluctuaciones notables, con algunos períodos de baja generación, 
                    especialmente en meses como octubre y noviembre de 2019, febrero y marzo de 2020, y enero de 2021.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col18:
        st.image("images/1.jpg", caption="Figura 1. Serie de tiempo, generación de energía fotovoltaica a lo largo del tiempo", use_container_width=True)

        st.markdown("""
            <style>
                .stText {
                    color: white;
                }
                .css-ffhzg2 {
                    background-color: #4B8E8D;
                }
            </style>
        """, unsafe_allow_html=True)

    col19, col20, col21 = st.columns([2, 2, 2])

    with col19: 
        st.markdown("""
            <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px;">
                <h4 style="color:white;">Imputación de Datos Faltantes</h4>
                <p style="color:white;">
                    Se utilizó un enfoque basado en la comparación con periodos similares (abril y junio) para imputar 
                    los datos faltantes, preservando los patrones temporales. Posteriormente, se aplicó <strong>interpolación 
                    lineal</strong> y técnicas de <strong>relleno hacia adelante y hacia atrás</strong> para asegurar la 
                    continuidad y coherencia de la serie temporal.
                </p>
                <p style="color:white;">
                    Los datos muestran una alta variabilidad, con fluctuaciones y caídas abruptas, especialmente desde 2023, 
                    probablemente debido a factores estacionales y cambios en las condiciones ambientales. Se observan patrones 
                    cíclicos típicos de la energía solar, con mayor generación en verano y menor en invierno.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <style>
                .stText {
                    color: white;
                }
                .css-ffhzg2 {
                    background-color: #4B8E8D;
                }
            </style>
        """, unsafe_allow_html=True)
        st.image("images/2.jpg", caption="Figura 2. Serie de tiempo con datos imputados.", use_container_width=True)


    with col20:
        st.markdown("""
            <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h3 style="color:white;">Generación de Energía del Día</h3>
                <p style="color:white;">
                    El diagrama de cajas refleja un patrón claro en la generación de energía a lo largo del día. La energía producida 
                    aumenta gradualmente durante la mañana, alcanzando su punto máximo entre las <strong>12:00 y las 14:00 horas</strong>, 
                    cuando la luz solar es más intensa. Posteriormente, la generación disminuye progresivamente durante la tarde, con 
                    valores mínimos al inicio y final del día.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <style>
                .stText {
                    color: white;
                }
                .css-ffhzg2 {
                    background-color: #4B8E8D;
                }
            </style>
        """, unsafe_allow_html=True)
        
        st.image("images/3.jpg", caption="Figura 3. Diagrama de caja de generación de energía por hora del día", use_container_width=True)

    with col21:
        st.markdown("""
            <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px;">
                <h4 style="color:white;">Estacionalidad y Patrones</h4>
                <p style="color:white;">
                    Los datos muestran un patrón de mayor <strong>variabilidad</strong> en <strong>2023</strong>, con picos en la generación 
                    durante ciertas épocas del año. La generación solar muestra un <strong>pico de producción entre 12:00 pm y 2:00 pm</strong> 
                    debido a la mayor radiación solar.
                </p>
            </div>
        """, unsafe_allow_html=True) 
        st.markdown("""
            <style>
                .stText {
                    color: white;
                }
                .css-ffhzg2 {
                    background-color: #4B8E8D;
                }
            </style>
        """, unsafe_allow_html=True)
        st.image("images/4.jpg", caption="Figura 4. Descomposición de la serie de tiempo.", use_container_width=True)

with tab3:
    col7, col8 = st.columns([10, 5])
    
    with col7:
        st.markdown("""
            <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px;">
                <h2 style="color:white;">Modelo LSTM (Long Short-Term Memory)</h2>
                <p style="color:white;">
                    Las <strong>LSTM</strong> son un tipo de red neuronal recurrente (RNN) diseñada para aprender y recordar dependencias 
                    a largo plazo en secuencias de datos. Su estructura incluye una célula de memoria que permite mantener el estado oculto 
                    a lo largo del tiempo, lo que le ayuda a "recordar" información relevante durante varios instantes, mejorando así su 
                    capacidad para procesar datos secuenciales. Se componen de <strong>células LSTM</strong> con tres "puertas" principales
                    que permiten manejar las dependencias temporales a lo largo del tiempo:
                </p>
                <ul style="color:white;">
                    <li><strong>Puerta de Entrada:</strong> Controla la cantidad de información que se incorpora a la célula LSTM.</li>
                    <li><strong>Puerta de Olvido:</strong> Determina qué información se olvida o descarta.</li>
                    <li><strong>Puerta de Salida:</strong> Regula qué información de la célula LSTM se utiliza para la predicción.</li>
                </ul>
                <p style="color:white;">
                    Las <strong>LSTM</strong> son especialmente útiles para tareas que requieren el análisis de <strong>dependencias temporales</strong> 
                    a largo plazo, como en el caso del <strong>pronóstico de la generación de energía fotovoltaica</strong>, ya que este tipo de 
                    modelo puede manejar secuencias de datos complejas con patrones de largo plazo.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col8:
        st.image("images/lstm.png", caption="LSTM Recurrent Neural Networks")

        st.markdown("""
            <style>
                .stText {
                    color: white;
                }
                .css-ffhzg2 {
                    background-color: #4B8E8D;
                }
            </style>
        """, unsafe_allow_html=True)

    col9, col10, col11 = st.columns([5, 5, 5])
    
    with col9:
        st.markdown("""
            <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h3 style="color:white;">¿Por qué las LSTM son útiles?</h3>
                <p style="color:white;">
                    Las <strong>LSTM</strong> permiten almacenar información importante durante secuencias largas y olvidar información 
                    irrelevante, lo que las hace ideales para el pronóstico de <strong>energía solar</strong> y otros datos secuenciales. 
                    Esto incluye datos como:
                </p>
                <ul style="color:white;">
                    <li>Radiación solar</li>
                    <li>Temperatura ambiental</li>
                    <li>Condiciones meteorológicas</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col10:
        st.markdown("""
            <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h3 style="color:white;">Aplicaciones de LSTM</h3>
                <p style="color:white;">
                    Las <strong>LSTM</strong> son aplicables en varios escenarios que involucran datos secuenciales complejos, como:
                </p>
                <ul style="color:white;">
                    <li><strong>Pronóstico de la demanda de energía</strong></li>
                    <li><strong>Predicción de ventas a largo plazo</strong></li>
                    <li><strong>Pronóstico del clima</strong></li>
                    <li><strong>Detección de anomalías en sistemas que evolucionan con el tiempo</strong></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col11:
        st.markdown("""
            <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h3 style="color:white;">Ventajas de las LSTM</h3>
                <p style="color:white;">
                    Las principales ventajas de las <strong>LSTM</strong> son:
                </p>
                <ul style="color:white;">
                    <li>Capacidad para capturar dependencias a largo plazo en los datos.</li>
                    <li>Memoria para retener información clave durante periodos largos.</li>
                    <li>Excelente para el análisis de secuencias complejas, como las series temporales.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

###########PREDICCIÓN###########

with tab4:
    st.header("Predicción")
    st.write("""
        En esta sección, describe el proceso de limpieza de datos y las transformaciones que realizamos para preparar 
        los datos antes de entrenar el modelo.
    """)
    
        # Configuración de la página de Streamlit

    # Cargar el modelo entrenado
    @st.cache_resource
    def load_model_cache():
        return load_model("best_modelLSTM.keras")

    model_LSTM = load_model_cache()

    # Cargar y preparar los datos
    @st.cache_data
    def load_data():
        df = pd.read_csv("df_imputado.csv", index_col=0, parse_dates=True)
        return df

    # Convertir la serie en datos supervisados
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # Preparar los datos para la predicción
    df = load_data()
    DatosM = pd.DataFrame({'Generacion_(kWh)': df['Generacion_(kWh)']})

    # Parámetro para seleccionar el tipo de predicción
    prediccion_tipo = st.sidebar.selectbox("Seleccione el tipo de predicción", ["Predicción Completa", "Predicción de Próximas 13 Horas"])

    train_size = int(len(DatosM) * 0.8)
    test_data = DatosM.iloc[train_size:]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    test_data_normalized = scaler.fit_transform(test_data[['Generacion_(kWh)']])
    seq_length = 13 * 14  # Longitud de secuencia de entrada

    # Definir la longitud de salida (13 horas)
    out_length = 13

    # Convertir los datos de prueba en formato supervisado
    data_test_LSTM = series_to_supervised(test_data_normalized, seq_length, out_length)
    X_test_LSTM = data_test_LSTM.values[:, 0:seq_length]
    X_test_LSTM = X_test_LSTM.reshape(X_test_LSTM.shape[0], X_test_LSTM.shape[1], 1)

    if prediccion_tipo == "Predicción Completa":
        # Realizar predicciones completas
        y_test_LSTM = data_test_LSTM.values[:, seq_length:]
        pred_LSTM = model_LSTM.predict(X_test_LSTM)
        pred_LSTM = scaler.inverse_transform(pred_LSTM)
        real_data = scaler.inverse_transform(y_test_LSTM)
        pred_LSTM = np.clip(pred_LSTM, a_min=0, a_max=None)
        
        # Calcular las métricas de predicción
        rmse = np.sqrt(mean_squared_error(real_data, pred_LSTM))
        mse = mean_squared_error(real_data, pred_LSTM)
        mae = mean_absolute_error(real_data, pred_LSTM)
        r2 = r2_score(real_data, pred_LSTM)

    else:
        # Predicción de Próximas 13 Horas
        input_seq = X_test_LSTM[0]  # Secuencia inicial
        pred = model_LSTM.predict(input_seq.reshape(1, seq_length, 1))
        pred_LSTM = scaler.inverse_transform(pred).reshape(-1, 1)
        
        # Calcular las métricas de predicción para las próximas 13 horas
        real_data = test_data['Generacion_(kWh)'].values[:out_length].reshape(-1, 1)
        rmse = np.sqrt(mean_squared_error(real_data, pred_LSTM))
        mse = mean_squared_error(real_data, pred_LSTM)
        mae = mean_absolute_error(real_data, pred_LSTM)
        r2 = r2_score(real_data, pred_LSTM)

    # Configuración de columnas para gráficas y cuadro de métricas
    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown(
            """
            <div style="background-color:#66cdaa; padding: 15px; border-radius: 10px; margin-top: 170px; margin-left: 30px; width: 90%;">
                <h3 style="color:white; text-align:center;">Métricas del Modelo</h3>
                <p style="color:white; font-size:16px;">Root Mean Square Error (RMSE): <b>{:.2f}</b></p>
                <p style="color:white; font-size:16px;">Mean Absolute Error (MAE): <b>{:.2f}</b></p>
                <p style="color:white; font-size:16px;">R2 Score: <b>{:.2f}</b></p>
                <p style="color:white; font-size:16px;">Mean Squared Error (MSE): <b>{:.2f}</b></p>
            </div>
            """.format(rmse, mae, r2, mse),
            unsafe_allow_html=True
        )

    with col1:
        st.title("Predicción de Generación de Energía con LSTM")

        if prediccion_tipo == "Predicción Completa":
            # Gráfico de predicción completa
            datos_full = np.array(range(len(real_data)))
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=datos_full, y=real_data[:, 0],
                                    mode='lines', line=dict(width=2, color='#145DA0'), 
                                    name="Generación real"))
            fig1.add_trace(go.Scatter(x=datos_full, y=pred_LSTM[:, 0],
                                    mode='lines', line=dict(width=2, color='#EC7C30'), 
                                    name="Generación predicha"))
            fig1.update_layout(title='Predicción Completa',
                            paper_bgcolor='white', plot_bgcolor='white',
                            font=dict(color='black'),
                            yaxis=dict(title='Generación (kWh)', color='black'),
                            xaxis=dict(title='Tiempo (horas)', color='black'),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='black')))
            st.plotly_chart(fig1, use_container_width=True)
            
            st.write("**Descripción del Gráfico de Predicción Completa:** El modelo logra capturar tendencias generales de generación, acercándose en gran medida a los valores reales. Sin embargo, se observan discrepancias en momentos de cambios abruptos, posiblemente debido a variaciones en la radiación solar que el modelo no detecta con precisión. Este análisis refleja que el modelo es eficaz en condiciones estables, aunque podría mejorarse para gestionar variaciones extremas con mayor precisión.")

            # Histograma de Errores
            errores = real_data - pred_LSTM
            fig3 = go.Figure(data=[go.Histogram(x=errores[:, 0], nbinsx=30, marker=dict(color='#2E86C1'))])
            fig3.update_layout(
                title='Distribución de Errores (Real - Predicho)',
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='black'),
                xaxis=dict(title='Error (kWh)', color='black'),
                yaxis=dict(title='Frecuencia', color='black')
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.write("**Distribución de Errores**: Este histograma muestra la distribución de los errores de predicción, calculados como la diferencia entre los valores reales y los valores predichos por el modelo (Real - Predicho). En el eje X se representa el error en kWh, y en el eje Y, la frecuencia de cada valor de error. La mayoría de los errores se agrupan alrededor del valor cero, lo que indica que el modelo en general tiene una buena precisión, ya que sus predicciones están cerca de los valores reales. Sin embargo, el hecho de que algunos errores se distribuyan hacia los lados positivos y negativos sugiere que el modelo a veces sobreestima y otras veces subestima los valores, aunque estos casos extremos son menos frecuentes. Esta visualización es útil para identificar si existe algún sesgo en el modelo; una distribución simétrica alrededor de cero, como la que vemos aquí, sugiere que el modelo no tiene un sesgo claro hacia sobreestimar o subestimar en promedio.")

        else:
            # Gráfico de predicción de próximas 13 horas sin etiquetas específicas
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(range(out_length)), y=real_data[:, 0],
                                    mode='lines', line=dict(width=2, color='#145DA0'), 
                                    name="Generación real"))
            fig2.add_trace(go.Scatter(x=list(range(out_length)), y=pred_LSTM[:, 0],
                                    mode='lines', line=dict(width=2, color='#EC7C30'), 
                                    name="Generación predicha"))
            fig2.update_layout(
                title='Predicción para las Próximas 13 Horas',
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='black'),
                yaxis=dict(title='Generación (kWh)', color='black', tickfont=dict(color='black')),
                xaxis=dict(title='Horas', color='black'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='black'))
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            st.write("**Predicción para las Próximas 13 Horas**: Este gráfico muestra la predicción del modelo para las próximas 13 horas. El eje X representa las horas de predicción desde el inicio de la secuencia, y el eje Y representa la generación de energía en kilovatios hora (kWh). La línea azul muestra los valores reales de generación, mientras que la línea naranja representa las predicciones del modelo.")

        # Gráfico de dispersión de valores reales vs. predichos
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=real_data[:, 0], y=pred_LSTM[:, 0],
                                mode='markers', marker=dict(color='#FF7F0E'), name="Predicho vs Real"))
        fig5.add_trace(go.Scatter(x=[0, 250], y=[0, 250],
                                mode='lines', line=dict(dash='dash', color='black'), name="Línea Perfecta"))
        fig5.update_layout(
            title='Valores Reales vs Predichos',
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='black'),
            xaxis=dict(title='Generación Real (kWh)', color='black'),
            yaxis=dict(title='Generación Predicha (kWh)', color='black'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='black'))
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.write("**Valores Reales vs Predichos**: La mayoría de los puntos se agrupan en torno a la línea perfecta, lo que indica un desempeño satisfactorio del modelo LSTM en la estimación de la generación de energía. Sin embargo, conforme aumentan los valores de generación de energía, se logra observar una mayor dispersión de los puntos, lo que podría sugerir que el modelo experimenta ligeras desviaciones.")

with tab5:
    st.markdown("""
        <div style="background-color:#4B8E8D; padding: 20px; border-radius: 10px;">
            <h2 style="color:white;">Referencias</h2>
            <ul style="color:white; font-size:16px; line-height:1.8;">
                <li>
                    Rodriguez-Leguizamon, C. K., López-Sotelo, J. A., Cantillo-Luna, S., & López-Castrillón, Y. U. (2023). 
                    PV power generation forecasting based on XGBoost and LSTM models. In <em>2023 IEEE Workshop on Power Electronics 
                    and Power Quality Applications (PEPQA)</em> (pp. 1–6). 
                    <a href="https://doi.org/10.1109/PEPQA59611.2023.10325757" style="color:#f1c40f;">https://doi.org/10.1109/PEPQA59611.2023.10325757</a>
                </li>
                <li>
                    RatedPower. (2022). Renewable Energy and Solar Research Report: What’s in Store for 2022. 
                    <a href="https://go.ratedpower.com/hubfs/Solar_Trends_Report_2022.pdf" style="color:#f1c40f;">https://go.ratedpower.com/hubfs/Solar_Trends_Report_2022.pdf</a>
                </li>
                <li>
                    Sharadga, H., Hajimirza, S., & Balog, R. S. (2020). Time series forecasting of solar power generation 
                    for large-scale photovoltaic plants. <em>Renewable Energy, 150</em>, 797–807. 
                    <a href="https://doi.org/10.1016/j.renene.2019.12.131" style="color:#f1c40f;">https://doi.org/10.1016/j.renene.2019.12.131</a>
                </li>
                <li>
                    Keddouda, A., Ihaddadene, R., Boukhari, A., Atia, A., Arıcı, M., Lebbihiat, N., & Ihaddadene, N. (2023). 
                    Solar photovoltaic power prediction using artificial neural network and multiple regression considering ambient and operating conditions. 
                    <em>Energy Conversion and Management, 288(117186)</em>, 117186. 
                    <a href="https://doi.org/10.1016/j.enconman.2023.117186" style="color:#f1c40f;">https://doi.org/10.1016/j.enconman.2023.117186</a>
                </li>
                <li>
                    Sostenible, C. (s/f). REPORTE DE SOSTENIBILIDAD. Campussostenible.org. Recuperado el 6 de septiembre de 2024, de 
                    <a href="https://campussostenible.org/wp-content/uploads/2023/11/Reporte-de-Sostenibilidad-2022.pdf" style="color:#f1c40f;">https://campussostenible.org/wp-content/uploads/2023/11/Reporte-de-Sostenibilidad-2022.pdf</a>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
