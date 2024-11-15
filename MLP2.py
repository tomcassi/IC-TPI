from pruebaMusic21 import procesar_primera_pista
from cargar_caracteristicas import cargarPista
from crear_secuencias import crear_secuencia
from crear_secuencias import aplanar_secuencia
from sklearn.neural_network import MLPRegressor  # Para regresión, no MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # Métrica adecuada para regresión
import numpy as np
import os


# Función para entrenar el modelo
def entrenar_modelo(X, y, modelo1):
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)

    # Entrenar el modelo
    modelo1.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = modelo1.predict(X_test)

    # Evaluar el modelo usando el Error Cuadrático Medio (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print("Error cuadrático medio (MSE):", mse)

    return modelo1,y_pred,y_test

# Parámetros principales
secuencia_len = 5  # Número de notas en cada secuencia de entrada

# Crear el modelo MLPRegressor para regresión
mlp = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=10000)



# Definir la carpeta donde estoy parado
carpeta_audios = "Audios/"


maximo_tamaño_acorde = 6

for nombre_archivo in os.listdir("Audios/"):
    ruta_completa = os.path.join(carpeta_audios, nombre_archivo)
    print(ruta_completa)
    # Agarramos de a una canción
    todos_caracteristicas = cargarPista(ruta_completa)
    
    # Crear las secuencias
    x, y = crear_secuencia(todos_caracteristicas, maximo_tamaño_acorde, secuencia_len)

    # Aplanar las secuencias
    x, y = aplanar_secuencia(x, y)


    #Entrenar el modelo y obtener el modelo entrenado
    mlp,y_pred,y_test = entrenar_modelo(x, y, mlp)
    








