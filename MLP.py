# Implementacion con perceptron multicapa
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def entrenar_modelo(X ,y):
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)

    # Crear el modelo MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(3,1), max_iter=1000)

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
        
    return mlp


if __name__ == "__main__":
    print("MLP")