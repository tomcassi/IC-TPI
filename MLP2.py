from pruebaMusic21 import procesar_primera_pista
from cargar_caracteristicas import cargarPista
from crear_secuencias import crear_secuencia
from crear_secuencias import aplanar_secuencia
from sklearn.neural_network import MLPRegressor  # Para regresión, no MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # Métrica adecuada para regresión
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



errores = []  # Lista para almacenar los errores en cada época


def entrenar_modelo(X, y, modelo1):

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True, random_state=42)
    
    # Normalizar las características (X)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.fit_transform(X_test)

    # Normalizar las etiquetas (y)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)  
    y_test_scaled = scaler_y.fit_transform(y_test)

    # Entrenar el modelo
    modelo1.fit(X_train_scaled, y_train_scaled)

    # Hacer predicciones en el conjunto de prueba
    y_pred_scaled = modelo1.predict(X_test_scaled)

    # Desnormalizar las predicciones y las etiquetas para evaluar en escala original
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_scaled)



    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])-1):
            y_pred=y_pred.astype(int)
            if y_pred[i][j]<10:
                y_pred[i][j]=0


    # Evaluar el modelo usando el Error Cuadrático Medio (MSE)
    mse = mean_squared_error(y_test_original, y_pred)
    print("Error cuadrático medio (MSE):", mse)
    
    # Almacenar el error en la lista global
    errores.append(mse)

    

    return modelo1, y_pred, y_test_original ,X_test

# Parámetros principales
secuencia_len = 5  # Número de notas en cada secuencia de entrada

# Crear el modelo MLPRegressor para regresión
mlp = MLPRegressor(
    hidden_layer_sizes=(1000, 1000),  # Dos capas ocultas de 128 neuronas
    activation='relu',             # Función de activación no lineal
    solver='adam',                 # Optimizador Adam
    learning_rate_init=0.001,      # Tasa de aprendizaje inicial
    max_iter=5000,                 # Iteraciones máximas
    random_state=42
)



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
    mlp,y_pred,y_test,x_test = entrenar_modelo(x, y, mlp)
    




plt.plot(errores)
plt.xlabel("Época")
plt.ylabel("MSE")
plt.title("Evolución del error durante el entrenamiento")
plt.show()

# num_predicciones=100
# X_nueva = [x[0]]  # Usar las últimas 10 notas para predecir la siguiente
# scaler_X = StandardScaler()
# X_nueva = scaler_X.transform(X_nueva)

# cancion_final=[]
# for i in range(num_predicciones):
    
#         cancion_final.append(X_nueva)
#         # Hacer predicción para la siguiente nota
       
#         prediccion = mlp.predict(X_nueva)
        
#         # Añadir la predicción a las notas
#         X_nueva= prediccion 
    




