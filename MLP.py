# Implementacion con perceptron multicapa
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def crear_secuencias(midi_data, longitud_secuencia):
    X, y = [], []
    
    for nota in range(len(midi_data.instruments[0].notes) - longitud_secuencia):
        # Extrae una secuencia de notas de longitud especificada
        listanotasX = midi_data.instruments[0].notes[nota:nota + longitud_secuencia]
        listapitchX = [n.pitch for n in listanotasX]  # Extrae los pitches de la secuencia
        
        # Agrega la secuencia a X y la nota siguiente como objetivo en y
        X.append(listapitchX)
        y.append(midi_data.instruments[0].notes[nota + longitud_secuencia].pitch)
    return X, y

def predecir_sig_notas():
    #Llamaria a tiempos, velocities y pitches y armaria la estructura de notas, devuelve notas
    return

def predecir_sig_tiempos():
    #hacer
    return

def predecir_sig_velocities():
    #hacer
    return

def predecir_sig_pitches(notas_originales, modelo, cant_predicciones):
    n_predicciones = 0
    pitches = []
    
    #cargo notas originales
    for nota in notas_originales:
        pitches.append(nota.pitch)

    print(pitches)
    
    while n_predicciones < cant_predicciones:
        pitches_input = np.array(pitches[-len(notas_originales):]).reshape(1, -1)
        print("Input para predicción:", pitches_input)
        
        prediccion = modelo.predict(pitches_input)[0]
        print("Predicción:", prediccion)
        
        pitches.append(prediccion)
        n_predicciones += 1
    
    return pitches


def entrenar_modelo(X ,y):
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)

    mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=1000)

    # Entrenar el modelo
    mlp.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = mlp.predict(X_test)

    # Evaluar la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión del modelo:", accuracy)

    # # Imprimir algunas predicciones
    # for i in range(len(y_test)):
    #     print(f"Predicción: {y_pred[i]}, Real: {y_test[i]}")
        
    return mlp


if __name__ == "__main__":
    print("MLP")