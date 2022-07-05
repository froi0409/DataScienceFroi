# Importamos librerías a utilizar en esta aplicación
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing
from sklearn import tree
import os
from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as PImage
from subprocess import check_call

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

### Definimos las funciones que utilizaremos

## Función que obtiene el archivo e identifica la colección de datos a utilizar
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
    

    col_resultado = any
    cols_evaluar = list()
    cols_encoded = list()
    dot_code = ''
    if bandera:
        name_last_column = str
        for x in data:
            name_last_column = str(x)

        # Si se presiona el botón de confirmación, se inicia el análisis
        #if confirmar_resultado:
        col_resultado = np.asarray(data[name_last_column]).reshape(-1,1)
        for x in data:
            if str(name_last_column) != str(x):
                col = np.asarray(data[str(x)]).reshape(-1,1)
                cols_evaluar.append(col)
        
        # Creamos un label enconder
        le = preprocessing.LabelEncoder()
        
        # Convertimos los strings de los arreglos en números
        for i in range(len(cols_evaluar)):
            cols_encoded.append(le.fit_transform(cols_evaluar[i]))

        resultado_encoded = le.fit_transform(col_resultado)

        # Combinamos los atributos en una lista de tuplas
        tuples = []
        for x in range(len(cols_encoded[0])):
            fila = []
            for i in range(len(cols_encoded)):
                fila.append(cols_encoded[i][x])
            tuples.append(tuple(fila))

        # Realizamos el modelo
        x_train, x_test, y_train, y_text = train_test_split(tuples, resultado_encoded, test_size=0.2)

        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=1000, solver='lbfgs').fit(x_train, y_train)



        st.text('Realizar una predicción. Ingrese los datos necesarios para predecir:')
        prediccion = st.text_input('Ingrese los datos códificados separados por coma (puede consultar la tabla)')
        btn_predecir = st.button('Realizar Predicción')
        

        with st.expander('Tabla de Datos'):
            col_tabla, col_encoded, col_res = st.columns(3)
            with col_tabla:
                st.text('Tabla')
                st.write(data)
            
            with col_encoded:
                st.text('Datos Encoded')
                st.dataframe(tuples)
            
            with col_res:
                st.text('Resultado Encoded')
                st.dataframe(resultado_encoded)
        

        if btn_predecir:
            with st.expander('Resultado de la Predicción: '):
                st.text('La predicción para la entrada: ' + prediccion + ' es:')
                prediction = [float(i) for i in prediccion.split(',')]
                predicted = str(le.inverse_transform(clf.predict([prediction]))).replace('[','').replace(']','').replace('\"','').replace('\'','')
                st.text(predicted)



## Inicio de la ejecución
# Colocamos el título de la sección
st.title('Clasificador: Redes Neuronales')
# Agregamos el manejador de archivos
upload_file = st.file_uploader('Seleccione un Archivo: ', type=['csv','json','xlsx','xls'])

if upload_file is not None:
    procesar_datos()
