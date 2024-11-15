from cargarnotasyacordes import *
import os
from cargar_caracteristicas import *



errores = []  # Lista para almacenar los errores en cada época


def entrenar_modelo(X, y, modelo1):
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True, random_state=42)

    # Entrenar el modelo
    modelo1.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = modelo1.predict(X_test)

    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])-1):
            y_pred=y_pred.astype(int)
            if y_pred[i][j]<10:
                y_pred[i][j]=0


    # Evaluar el modelo usando el Error Cuadrático Medio (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print("Error cuadrático medio (MSE):", mse)
    
    # Almacenar el error en la lista global
    errores.append(mse)

    

    return modelo1, y_pred, y_test ,X_test



def crear_secuencias(caracteristicas, longitud_secuencia):
    X, y = [], []
    for nota in range(len(caracteristicas) - longitud_secuencia):
        # Extrae una secuencia de notas de longitud especificada
        
        listanotasX = caracteristicas[nota:nota + longitud_secuencia]
        X.append(listanotasX)
        
        y.append(caracteristicas[nota + longitud_secuencia])
    return X, y


if __name__ == "__main__":
    carpeta_audios = "Audios/"
    print("\n Cargando acordes")
    for nombre_archivo in os.listdir(carpeta_audios):
        archivo_midi = os.path.join(carpeta_audios, nombre_archivo)
        
        try:
            pitches = cargar_acordes(archivo_midi)
            print(f'Archivo cargado: {archivo_midi}')
        except ValueError as e:
            print(e)
            continue
        
        for pitch in pitches:
            if pitch not in notasyacordes:
                notasyacordes.append(pitch)
    print("\n Cargando caracteristicas")
                

    for nombre_archivo in os.listdir(carpeta_audios):
        archivo_midi = os.path.join(carpeta_audios, nombre_archivo)
        
        todos_caracteristicas = cargarPista(archivo_midi)
        for i, nota_acorde in enumerate(todos_caracteristicas[0]):
            indice = notasyacordes.index(sorted(nota_acorde))
            todos_caracteristicas[0][i] = indice
        
        
        
        

        
        
        
    



# # Parámetros principales
# secuencia_len = 10  # Número de notas en cada secuencia de entrada

# # Crear el modelo MLPRegressor para regresión
# mlp = MLPRegressor(hidden_layer_sizes=(100, 100),  # Dos capas ocultas de 128 neuronas
#     activation='relu',             # Función de activación no lineal
#     solver='adam',                 # Optimizador Adam
#     learning_rate_init=0.001,      # Tasa de aprendizaje inicial
#     max_iter=5000,                 # Iteraciones máximas
#     random_state=42
# )



# # Definir la carpeta donde estoy parado
# carpeta_audios = "Audios/"


# maximo_tamaño_acorde = 6

# for nombre_archivo in os.listdir("Audios/"):
#     ruta_completa = os.path.join(carpeta_audios, nombre_archivo)
#     print(ruta_completa)
#     # Agarramos de a una canción
#     todos_caracteristicas = cargarPista(ruta_completa)
    
#     # Crear las secuencias
#     x, y = crear_secuencia(todos_caracteristicas, maximo_tamaño_acorde, secuencia_len)

#     # Aplanar las secuencias
#     xa, ya = aplanar_secuencia(x, y)


#     #Entrenar el modelo y obtener el modelo entrenado
#     mlp,y_pred,y_test,x_test = entrenar_modelo(xa, ya, mlp)
    




# plt.plot(errores)
# plt.xlabel("Época")
# plt.ylabel("MSE")
# plt.title("Evolución del error durante el entrenamiento")
# plt.show()

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
    




