from nltk.corpus.reader import KEYWORD
import streamlit as st 
import utils
import pickle
import numpy as np 
from sklearn.preprocessing import LabelEncoder
import json

st.set_page_config(page_title="ChatBot Vocacional",
                    page_icon="🚀",
                    layout="wide")
        
#st.title("ChatBot Recomendación Vocacional")

#Función para cargar el modelo (con cache)
@st.cache_resource
def load_model():
    try: 
        model = pickle.load(open('/content/drive/MyDrive/CursoAI/Copia_de_la_copia_de_ProyectoAi.pkl', "rb"))
        return model
    except FileNotFoundError:
        st.error("No se encontró el archivo model.pkl. Asegúrate de que esté en el mismo directorio.")
        return None

#cargar el modelo ya entrenado y guardado  #Corregir
model = load_model()

#Titulo de la Aplicación 
st.title("🚀 ChatBot Recomendación Vocacional")
st.markdown("---")

if model is not None:
    #crear columnas para el layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Ingresa tus Notas")

        #Inputs para las caracteristcas (ajustar a nuestro modelo)
        nota1 = st.number_input(
            "Nota en Lengua Castellana",
        min_value = 0.0, 
        max_value= 5.0,
        value = 5.0,
        step = 0.1
    )
    
        nota2 = st.number_input(
            "Nota en Artistica",
        min_value = 0.0,
        max_value = 5.0,
        value = 5.0,
        step = 0.1
    )

        nota3 = st.number_input(
            "Nota en Educación Fisica",
        min_value = 0.0,
        max_value = 5.0,
        value = 5.0,
        step = 0.1
    )

    nota4 = st.number_input(
            "Nota en Matematicas",
        min_value = 0.0,
        max_value= 5.0,
        value = 5.0,
        step = 0.1
    )

    nota5 = st.number_input(
            "Nota en Dibujo Técnico",
        min_value = 0.0,
        max_value = 5.0,
        value = 5.0,
        step = 0.1
    )

    nota6 = st.number_input(
            "Nota en Civica y Urbanidad",
        min_value = 0.0,
        max_value = 5.0,
        value = 5.0,
        step = 0.1
    )

    nota7 = st.number_input(
            "Nota en Etica y Valores",
        min_value = 0.0,
        max_value = 5.0,
        value = 5.0,
        step = 0.1
    )

    nota8 = st.number_input(
            "Nota en Ciencias Naturales",
        min_value = 0.0,
        max_value = 5.0,
        value = 5.0,
        step = 0.1
    )

          #convertir nota de escala 1-5 a escala 1-20, para facilitar la interacción con el usuario
    def convertir_nota_5_a_20(nota_5):
        
    #Conversion lineal de la escala 1-5 a 1-20
        nota_20 = ((nota_5 -1) / 4) * 19 + 1
        return nota_20



    #Boton de Predicción
    if st.button("🔮 Realizar Predicción", type = "primary"): 
        #Preparar los datos para la predicción
        features1 = np.array([[nota1, nota2, nota3, nota4, nota5, nota6, nota7, nota8]])
        features = []
        for i in features1: 
            features.append(convertir_nota_5_a_20(i)) 


        try:
            #Realizar la Predicción
            prediction = model.predict(features)
            primeras_5 = prediction[:5]
            prediction_proba = model.prediction_proba(features) if hasattr(model, 'predic_proba') else None

            #Mostrar el resultado
            with col2:
                st.header("Resultado de la Predicción")
                st.success(f"**Resultado Predicho:**{primeras_5}")
                #st.success(f"**Resultado Predicho:**{prediction[1]}")

                #Mostrar probabilidades si estan disponibles
                if prediction_proba is not None:
                    st.subheader("Probabilidades:")
                    #Ajustar las clases según nuestro modelo
                    classes = model.classes_ #Esto para llamar las 71 clases del dataframe y no escribarlas.
                    for i, prob in enumerate(prediction_proba[0]):
                        st.write(f"**{classes[1]}:** {prob:.2%}") 

                        #Grafico de barras de probabilidades
                        prob_df = pd.dataframe({
                            'Recomendaciones': classes,
                            'Probabilidad': prediction_proba[0]
                        })

        except Exception as e:
            st.error(f"Error al realizar la predicción: {str(e)}")

    with col2:
        if 'prediction' not in locals():
            st.header("Resultado")
            st.info("Ingresa los valores y presiona 'Realizar Predicción'")

    #información adicional
    st.markdown("---")
    st.subheader("Información del Modelo")

    col3, col4 = st.columns(2)
    with col3:
        st.info("**Tipo de Modelo: ** Random Forest Classifier")
    with col4:
        st.info("**Características: **Predicción de Profesiones")

else:
    st.error("No se pudo cargar el modelo. Verifica que el archivo model.pkl esté disponible.")

#Sidebar con información adicional
with st.sidebar:
    st.header("🗿Información")
    st.markdown("""
    ### Cómo usar la aplicación:
    1. Ingresa las calificaciones de las materias
    2. Presiona 'Realizar Predicción'
    3. Observa el resultado
    4. Escribe si tienes alguna pregunta en el cuadro 'Escribe tu mensaje'
    
    ### Caracteristicas requeridas:
    - Notas recientes de la escuela o colegio
    - Calificaciones en escala de 1.00 a 5.00
    """)

    st.markdown("---")
    st.markdown("**Desarrollado en el Bootcamp - TalentoTech - 2025/1**🪐")

#historial / guardar en cache el historial de las entradas de los usuarios

if "history" not in st.session_state:
    st.session_state.history = []

#contexto

if "context" not in st.session_state:
    st.session_state.context = []

#Construccion del espacio, emisor - mensaaje - ciclo que se repite constantemente

for sender, msg in st.session_state.history:
    if sender == 'Tú':
        st.markdown(f'**🤓{sender}:**{msg}')
    else:
        st.markdown(f'**🤖{sender}:**{msg}')

#Si no hay entrada 
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

#Procesamiento de la entrada
def send_msg():
    user_input = st.session_state.user_input.strip()
    if user_input:
        tag = utils.predict_class(user_input)
        st.session_state.context.append(tag)
        response = utils.get_response(tag, st.session_state.context)
        st.session_state.history.append(('Tú', user_input))
        st.session_state.history.append(('Bot', response))
        st.session_state.user_input = ""


#Crear campo de texto
st.text_input("Escribe tu mensaje: ", key="user_input", on_change=send_msg)