# Importamos librerías a utilizar en esta aplicación
from matplotlib.pyplot import text
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import linear_model
from PIL import Image
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
        resultado = st.selectbox('Seleccione la columna que será utilizada como el resultado', data.columns)
        confirmar_resultado = st.button('Confirmar')
        
        # Si se presiona el botón de confirmación, se inicia el análisis
        if confirmar_resultado:
            col_resultado = np.asarray(data[resultado]).reshape(-1,1)
            for x in data:
                if str(resultado) != str(x):
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
            clf = DecisionTreeClassifier().fit(tuples,resultado_encoded)
            plot_tree(clf, filled=True)
            plt.savefig('clasiArbol.png')

            # exportar el modelo a archivo .dot
            # with open(r"clasiArbol.dot", 'w') as f:
            #    f = tree.export_graphviz(clf,
            #                            out_file=f,
            #                            max_depth = 7,
            #                            impurity = True,
            #                            #feature_names = list(tuples.drop(['top'], axis=1)),
            #                            #class_names = ['No', 'N1 Billboard'],
            #                            rounded = True,
            #                            filled= True 
            #                            )
            # dot_code = f
            

            # Convertir el archivo .dot a png para poder visualizarlo
            # check_call(['dot','-Tpng',r'clasiArbol.dot','-o',r'clasiArbol.png'])
            # PImage("clasiArbol.png")

            #for i in range(len(cols_evaluar)):
            #    st.text(cols_evaluar[i])


        with st.expander('Tabla de Datos'):
            st.write(data)
            if confirmar_resultado:
                st.text(tuples)
            
        if confirmar_resultado:
            with st.expander('Clasificador de Arboles de Decisión:'):
                image = Image.open('clasiArbol.png')
                st.image(image)


## Inicio de la ejecución
# Colocamos el título de la sección
st.title('Clasificador: Arboles de Decisión')
# Agregamos el manejador de archivos
upload_file = st.file_uploader('Seleccione un Archivo: ', type=['csv','json','xlsx','xls'])

if upload_file is not None:
    procesar_datos()
