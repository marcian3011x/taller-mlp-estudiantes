import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cargar los modelos previamente entrenados
modelo = joblib.load('mlp_model.pkl')
escalador = joblib.load('scaler.pkl')
imputador = joblib.load('imputer.pkl')

st.title("Predicción de Aprobación - Estudiantes")
st.write("Ingresa los datos del estudiante para predecir si aprobará o reprobará la asignatura.")

# Crear el formulario en la web
tareas = st.number_input("Calificación de Tareas (0-100)", min_value=0.0, max_value=100.0, value=50.0)
examenes = st.number_input("Calificación de Exámenes (0-100)", min_value=0.0, max_value=100.0, value=50.0)
asistencia = st.number_input("Porcentaje de Asistencia (0-100)", min_value=0.0, max_value=100.0, value=80.0)
estudio = st.number_input("Horas de Estudio por Semana", min_value=0.0, value=10.0)
sueno = st.number_input("Horas de Sueño Diarias", min_value=0.0, value=7.0)
extra = st.selectbox("¿Participa en Actividades Extracurriculares?", ["No", "Sí"])

# Convertir la respuesta de texto a binario
extra_num = 1 if extra == "Sí" else 0

if st.button("Realizar Predicción"):
    # Organizar los datos tal como los espera el modelo (en el mismo orden del dataset)
    input_data = pd.DataFrame([[tareas, examenes, asistencia, estudio, sueno, extra_num]], 
                              columns=['Tareas', 'Examenes', 'Asistencia', 'Estudio_Horas', 'Sueno_Horas', 'Extracurriculares'])
    
    # Aplicar el mismo preprocesamiento del Jupyter Notebook
    input_imputed = imputador.transform(input_data)
    input_scaled = escalador.transform(input_imputed)
    
    # Hacer la predicción
    prediccion = modelo.predict(input_scaled)
    
    # Mostrar el resultado
    st.markdown("---")
    if prediccion[0] == 1:
        st.success("**Resultado:** El estudiante probablemente **APROBARÁ**.")
    else:
        st.error("**Resultado:** El estudiante probablemente **REPROBARÁ** (Se sugiere intervención pedagógica).")