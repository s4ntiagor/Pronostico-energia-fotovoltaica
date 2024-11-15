import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import streamlit as st

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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Datos Recolectados", "Modelo", "Predicción", "Sobre Nosostros"])

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

with tab4:
    st.header("Predicción")
    st.write("""
        En esta sección, describe el proceso de limpieza de datos y las transformaciones que realizaste para preparar 
        los datos antes de entrenar el modelo. Puedes mencionar pasos como la eliminación de valores nulos, 
        normalización de datos, etc.
    """)