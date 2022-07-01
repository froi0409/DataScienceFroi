# Importamos librerías a utilizar en esta aplicación
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import os

### Definimos las funciones que utilizaremos
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
            st.text('Se encontró un archivo csv')
            bandera = True
            data = pd.read_csv(upload_file)
        elif extension == '.json':
            st.text('Se encontró un archivo json')
            bandera = True
            data = pd.read_json(upload_file)
        elif extension == '.xlsx' or extension == '.xls':
            st.text('Se encontró un archivo xlsx o xls')
            bandera = True
            data = pd.read_excel(upload_file)
        else:
            st.text('Se insertó un archivo inválido')
    else:
        st.text('Para realizar un análisis, primero debe seleccionar un archivo')
    
    # Si bandera es verdadero, es porque si se pudo llevar a cabo el análisis del archivo
    if bandera:
        st.write(data)



## Iniciamos con el manejo de archivos
# Colocamos el título de la sección
st.title('Regresión Lineal')

col_1, col_2 = st.columns(2)

with col_1:
    upload_file = st.file_uploader("Seleccione  un archivo", type=['csv','json','xlsx','xls'])

with col_2:
    operacion = st.selectbox('Selecciona la Operación que deseas Realizar',('Graficar','Definir Función de Tendencia','Prediccion de Tendencia'))

button_process= st.button('procesar')


if button_process:
    procesar_datos()





