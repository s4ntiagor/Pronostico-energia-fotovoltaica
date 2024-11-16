# Pronóstico de Generación de Energía Fotovoltaica

Este repositorio contiene el código para el despliegue de un modelo LSTM, desarrollado para pronosticar la generación de energía fotovoltaica. La interfaz del despliegue está implementada con Streamlit, proporcionando una experiencia interactiva y visual.


## Pasos para ejecutar el proyecto

1. **Clonar el repositorio**  
   Abre tu terminal y ejecuta el siguiente comando:  
   ```bash
   git clone <https://github.com/s4ntiagor/tercera_entrega_IA.git>
   ```
   
2. **Crear configuraciones de Streamlit**
   Creamos una carpeta .streamlit en la raiz del proyecto, y dentro de ella creamos un archivo llamado 'config.toml' y copiamos lo siguiente:
   ```bash
   [theme]
   backgroundColor = "#dcd3d3"
   secondaryBackgroundColor = "#079898"
   textColor = "#000000"
   ```
    
4. **Instalar las dependencias**  
   Instala las librerías requeridas ejecutando el siguiente comando en la raíz del proyecto:  

   ```bash
   pip install -r requirements.txt
   ```

5. **Ejecutar la aplicación**
   Corre el archivo principal de la aplicación Streamlit: 

   ```bash
   streamlit run app_streamlit.py
   ```

   Después de ejecutar este comando, podrás ver la aplicación en tu navegador web. Streamlit proporcionará enlaces como los siguientes:

   ```bash
   Local URL: http://localhost:8501  
   Network URL: http://<tu-dirección-de-red>:8501  
   ```
