# Importamos librerías a utilizar en esta aplicación
from statistics import LinearRegression, linear_regression
from matplotlib.pyplot import text
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import linear_model 
from sklearn.preprocessing import PolynomialFeatures 
from PIL import Image


### Definimos las funciones que utilizaremos

## Función que nos sirve para llevar a cabo las operaciones
def operar_datos(data, parametro_x, parametro_y, entrada_prediccion, grado_polinomio):
    x = np.asarray(data[parametro_x]).reshape(-1,1)
    y = data[parametro_y]
    
    polynomial_features = PolynomialFeatures(degree=grado_polinomio)
    x_transform = polynomial_features.fit_transform(x)

    regresion_lineal = linear_model.LinearRegression()

    #model = linear_model.LinearRegression().fit(x_transform, y)
    regresion_lineal.fit(x_transform, y)
    y_new = regresion_lineal.predict(x_transform)

    # Agregamos la gráfica de puntos 
    with st.expander('Gráfica de Puntos:'):
        st.title('Gráfica de Puntos:')
        st.text('X: ' + parametro_x)
        st.text('Y: ' + parametro_y)
        plt.scatter(x, y, color='green')
        plt.plot(x, y_new, color='red')
        plt.savefig('polinomial.png')
        plt.close()
        
        image = Image.open('polinomial.png')
        st.image(image)

    with st.expander('Función de Tendencia:'):
        ecuacion = str(regresion_lineal.intercept_).replace('[','').replace(']','')
        cont = 0
        for x in regresion_lineal.coef_:
            if x > 0 and cont != 0:
                ecuacion += '+'
            ecuacion += str(x).replace('[','').replace(']','') + "x^" + str(cont)
            cont = cont + 1


        st.latex(ecuacion)


    # Agregamos la funcionalidad de predicción de la tendencia
    if entrada_prediccion:
        with st.expander('Predicción de Tendencia:'):
            x_new_min = (entrada_prediccion) - 0.0
            x_new_max = entrada_prediccion

            x_new = np.linspace(x_new_min, x_new_max, int(x_new_max))
            x_new = x_new[:,np.newaxis]

            x_new_transform = polynomial_features.fit_transform(x_new)
            prediccion = regresion_lineal.predict(x_new_transform)

            #prediccion = regresion_lineal.predict([[entrada_prediccion]])
            st.text('La predicción a ' + str(entrada_prediccion) + ' (unidades de tiempo) es: ' + str(prediccion[0]))






## Función que obtiene el archivo y obtiene la coleccion de datos a utilizar
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

    # Si bandera tiene un valor verdadero es porque si se pudo llevar a cabo el análisis del archivo 
    if bandera:
        # Colocamos las columnas que debemos seleccionar
        col_1, col_2 = st.columns(2)

        with col_1:
            param_x = st.selectbox('Parametro X:', data.columns)
        with col_2:
            param_y = st.selectbox('Parametro Y:', data.columns)
        
        grado = st.slider('Seleccione el Grado de la Regresión Polinomial', max_value=10, min_value=2)

        check_predict = st.checkbox('¿Desea Realizar una Predicción?')
        entrada_prediccion = 0
        if check_predict:
            entrada_prediccion = st.number_input('Ingresa las unidades de tiempo que se usarán para la predicción:')    

        btn_procesar = st.button('Procesar Datos')
        
        with st.expander('Tabla de Datos'):
            st.write(data)
        
        if btn_procesar:
            operar_datos(data, param_x, param_y, entrada_prediccion,grado)


## Colocamos los elementos que utilizaremos en la página

st.title('Regresión Polinomial')
# Colocamos el manejador de archivos
upload_file = st.file_uploader('Seleccione un Archivo:',type=['csv','json','xlsx','xls'])

if upload_file is not None:
    procesar_datos()
