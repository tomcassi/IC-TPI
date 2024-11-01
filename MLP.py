# Implementacion con perceptron multicapa
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def entrenamiento(X ,y):
    # Cargar un conjunto de datos de ejemplo
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Crear el modelo MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(5,5), max_iter=1000)

    # Entrenar el modelo
    mlp.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = mlp.predict(X_test)

    # Evaluar la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión del modelo:", accuracy)

    # Imprimir algunas predicciones
    for i in range(len(y_test)):
        print(f"Predicción: {y_pred[i]}, Real: {y_test[i]}")
    
    return y_pred



def crear_secuencias(midi_data, longitud_secuencia = 5):
    X, y = [],[]
    
    for nota in range(0,len(midi_data.instruments[0].notes),longitud_secuencia):
        listanotasX = []
        listapitchX = []
        
        if nota+longitud_secuencia < len(midi_data.instruments[0].notes):
            listanotasX = midi_data.instruments[0].notes[nota:nota+longitud_secuencia]
            print("X",nota,nota+longitud_secuencia)
            print("y",nota+longitud_secuencia)
            for n in listanotasX:
                listapitchX.append(n.pitch)
            y.append(midi_data.instruments[0].notes[nota+longitud_secuencia].pitch) 
            X.append(listapitchX)        
        



    return X,y


if __name__ == "__main__":
    print("MLP")