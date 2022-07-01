# Importamos librerías a utilizar en esta aplicación
from matplotlib.pyplot import text
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import linear_model
from PIL import Image

### Definimos las funciones que utilizaremos

## Función que nos sirve para llevar a cabo las operaciones
def operar_datos(data, parametro_x, parametro_y):
    # Obtenemos los parametros de x, y
    x = np.asarray(data[parametro_x]).reshape(-1,1)
    y = data[parametro_y]

    regr = linear_model.LinearRegression()
    regr.fit(x,y)

    # Agregamos la gráfica de puntos 
    with st.expander('Gráfica de Puntos:'):
        st.title('Gráfica de Puntos:')
        st.text('X: ' + parametro_x)
        st.text('Y: ' + parametro_y)
        plt.scatter(x, y, color='green')
        plt.savefig('puntos.png')
        plt.close()
        
        image = Image.open('puntos.png')
        st.image(image)

    # Agregamos la funcionalidad de definir función de tendencia
    with st.expander('Función de Tendencia:'):
        st.text('Función de tendencia: ')

    # Agregamos la funcionalidad de predicción de la tendencia
    with st.expander('Predicción de Tendencia:'):
        st.text('Predicción de Tendencia')


##Funcion que obtiene el csv e identifica la coleccion de datos a utilizar
def procesar_datos():
    if upload_file is not None:
        # Obtenemos la extensión del archivo
        split_tup = os.path.splitext(upload_file.name)
        nombre_archivo = split_tup[0]
        extension = split_tup[1]
        
        # Cramos la data
        bandera = False
        data:any

        # Identificamos la extensión del archivo
        if extension == '.csv':
            bandera = True
            data = pd.read_csv(upload_file)
        elif extension == '.json':
            bandera = True
            data = pd.read_json(upload_file)
        elif extension == '.xlsx' or extension == '.xls':
            bandera = True
            data = pd.read_excel(upload_file)
        else:
            st.text('Se insertó un archivo inválido')
    else:
        st.text('Para realizar un análisis, primero debe seleccionar un archivo')
    
    # Si bandera tiene un valor verdadero, es porque si se pudo llevar a cabo el análisis del archivo
    if bandera:
        # Colocamos las columnas que debemos seleccionar
        col_1, col_2 = st.columns(2)

        with col_1:
            param_x = st.selectbox('Parametro X:', data.columns)
        with col_2:
            param_y = st.selectbox('Parametro Y:', data.columns)
        
        btn_procesar = st.button('Procesar Datos')
        
        with st.expander('Tabla de Datos'):
            st.write(data)
        
        if btn_procesar:
            operar_datos(data, param_x, param_y)






## Iniciamos con el manejo de archivos
# Colocamos el título de la sección
st.title('Regresión Lineal')
upload_file = st.file_uploader('Seleccione un Archivo: ', type=['csv','json','xlsx','xls'])

if upload_file is not None:
    procesar_datos()

