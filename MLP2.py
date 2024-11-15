from pruebaMusic21 import procesar_primera_pista
from cargar_caracteristicas import cargarPista
from crear_secuencias import crear_secuencia
from crear_secuencias import aplanar_secuencia
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




def entrenar_modelo(X ,y, modelo1):
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)

    # Entrenar el modelo
    modelo1.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = modelo1.predict(X_test)

    # Evaluar la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión del modelo:", accuracy)

  
        
    return mlp





# Parámetros principales
secuencia_len = 5  # Número de notas en cada secuencia de entrada

# Definir la carpeta donde estoy parado
carpeta_audios = r'C:\Users\Rama\Desktop\b\IC-TPI\Audios\beethoven1.mid'


#Agarramos de a una cancion
todos_caracteristicas = cargarPista(carpeta_audios)
maximo_tamaño_acorde=5

x,y = crear_secuencia(todos_caracteristicas,maximo_tamaño_acorde,secuencia_len)

    
x,y = aplanar_secuencia (x,y)


mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=1000)

#mlp_entrenado = entrenar_modelo(x,y,mlp)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, shuffle=True)

    # Entrenar el modelo
mlp.fit(X_train, y_train)

    # Hacer predicciones
y_pred = mlp.predict(X_test)

    # Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)





