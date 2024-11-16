import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from music21 import converter, tempo, chord, note, instrument, stream
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from mapaNotasAcordes import cargar_notas_acordes_canciones
from procesarMidi import cargarPista, generar_cancion




def crear_secuencias(caracteristicas, longitud_secuencia):
    X, y = [], []
    for nota in range(len(caracteristicas) - longitud_secuencia):
        # Extrae una secuencia de notas de longitud especificada
        
        listanotasX = caracteristicas[nota:nota + longitud_secuencia]
        X.append(listanotasX)
        
        y.append(caracteristicas[nota + longitud_secuencia])
    return X, y


def entrenar_modelo(X, y, mlp):
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)


    # Entrenar el modelo
    mlp.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = mlp.predict(X_test)

    # Evaluar la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión del modelo:", accuracy)
        
    return mlp, y_pred, y_test

def predecir_sig_elem(elem_originales, modelo, cant_predicciones):
    n_predicciones = 0
    elementos = elem_originales
    
    
    while n_predicciones < cant_predicciones:
          
        elem_input = np.array(elementos[n_predicciones:n_predicciones+len(elem_originales)]).reshape(1, -1)
        print("Input para predicción:", elem_input)
        
        prediccion = modelo.predict(elem_input)
        prediccion = prediccion[0]
        print("Predicción:", prediccion)
        
        elementos.append(prediccion)
        n_predicciones += 1
    
    return elementos

def inicializar_modelo(carpeta_audios,longitud_secuencia, notasyacordes):
    mlp_pitch = RandomForestClassifier(n_estimators=100)
    mlp_velocity = RandomForestClassifier(n_estimators=100)
    mlp_duration = RandomForestClassifier(n_estimators=100)
    # mlp_pitch = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=10000)
    # mlp_velocity = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=10000)
    # mlp_duration = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=10000)
    
    print("\n=====Cargando caracteristicas=====")
    
    for nombre_archivo in os.listdir(carpeta_audios):
        archivo_midi = os.path.join(carpeta_audios, nombre_archivo)
        
        todos_caracteristicas = cargarPista(archivo_midi)
        for i, nota_acorde in enumerate(todos_caracteristicas[0]):
            indice = notasyacordes.index(sorted(nota_acorde))
            todos_caracteristicas[0][i] = indice
        
        X,y = crear_secuencias(todos_caracteristicas[0],longitud_secuencia)
        
        mlp_pitch, y_pred, y_test = entrenar_modelo(X,y,mlp_pitch)
        
        X,y = crear_secuencias(todos_caracteristicas[1],longitud_secuencia)
        
        mlp_velocity, y_pred, y_test = entrenar_modelo(X,y,mlp_velocity)
        
        X,y = crear_secuencias(todos_caracteristicas[2],longitud_secuencia)
        # Multiplicar cada valor dentro de X por 100 y convertir a int
        X = [[int(valor * 1000) for valor in sublista] for sublista in X]
        
        # Multiplicar cada valor en y por 100 y convertir a int
        y = [int(valor * 1000) for valor in y]
        
        mlp_duration, y_pred, y_test = entrenar_modelo(X,y,mlp_duration)
        
    return mlp_pitch, mlp_velocity, mlp_duration


def predecir_cancion(mlp_pitch, mlp_velocity, mlp_duration, longitud_secuencia, notasyacordes):
    #Predecir cancion:
    todos_caracteristicas = cargarPista("Audios/waldstein_2.mid")
    for i, nota_acorde in enumerate(todos_caracteristicas[0]):
        indice = notasyacordes.index(sorted(nota_acorde))
        todos_caracteristicas[0][i] = indice
    
    cant_predicciones = 100
    pitches_conprediccion = predecir_sig_elem(todos_caracteristicas[0][0:longitud_secuencia], mlp_pitch, cant_predicciones)
    velocities_conprediccion = predecir_sig_elem(todos_caracteristicas[1][0:longitud_secuencia], mlp_velocity, cant_predicciones)
    
    durations_originales = todos_caracteristicas[2][0:longitud_secuencia]
    
    for i in range(len(durations_originales)):
        durations_originales[i] *= 1000
    
    durations_conprediccion = predecir_sig_elem(durations_originales, mlp_duration, cant_predicciones)
    
    for i in range(len(durations_conprediccion)):
        durations_conprediccion[i] /= 1000
    
    
    for i in range(len(pitches_conprediccion)):
        pitches_conprediccion[i] = notasyacordes[pitches_conprediccion[i]]
    return pitches_conprediccion, velocities_conprediccion, durations_conprediccion





if __name__ == "__main__":
    l_s = 20
    c_a = "Audios/"
    
    mapa = cargar_notas_acordes_canciones(c_a)
    
    mlp_p, mlp_v, mlp_d = inicializar_modelo(c_a,l_s, mapa)
    
    p_conprediccion, v_conprediccion, d_conprediccion = predecir_cancion(mlp_p, mlp_v, mlp_d, l_s, mapa)

            
    cancion_generada = generar_cancion(p_conprediccion, v_conprediccion, d_conprediccion)

    # Guardar la canción en un archivo MIDI
    cancion_generada.write('midi', fp='cancion_generada.mid')
