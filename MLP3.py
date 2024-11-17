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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression




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

def inicializar_modelo(carpeta_audios,longitud_secuencia, notasyacordes, nombre_pieza):
    # mlp_pitch = LogisticRegression()
    # mlp_velocity = LogisticRegression()
    # mlp_duration = LogisticRegression()
    
    # mlp_pitch = KNeighborsClassifier(n_neighbors=3)
    # mlp_velocity = KNeighborsClassifier(n_neighbors=3)
    # mlp_duration = KNeighborsClassifier(n_neighbors=3)
    
    mlp_pitch = RandomForestClassifier(n_estimators=100)
    mlp_velocity = RandomForestClassifier(n_estimators=100)
    mlp_duration = RandomForestClassifier(n_estimators=100)
    
    # mlp_pitch = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=10000)
    # mlp_velocity = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=10000)
    # mlp_duration = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=10000)
    
    print("\n=====Cargando caracteristicas=====")
    
    for nombre_archivo in os.listdir(carpeta_audios):
        archivo_midi = os.path.join(carpeta_audios, nombre_archivo)
        
        todos_caracteristicas = cargarPista(archivo_midi, nombre_pieza)
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


def predecir_cancion(mlp_pitch, mlp_velocity, mlp_duration, longitud_secuencia, notasyacordes, cancion_inicial, nombre_pieza, cant_predicciones):
    #Predecir cancion:
    todos_caracteristicas = cargarPista(cancion_inicial, nombre_pieza)
    for i, nota_acorde in enumerate(todos_caracteristicas[0]):
        indice = notasyacordes.index(sorted(nota_acorde))
        todos_caracteristicas[0][i] = indice
    
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

def getTempo(midi_file):
    from music21 import converter, tempo

    # Cargar el archivo MIDI
    midi_data = converter.parse(midi_file)

    # Obtener todos los objetos de tipo MetronomeMark (tempo)
    tempos = midi_data.flat.getElementsByClass(tempo.MetronomeMark)

    # Si hay tempos definidos, devolver el primero; de lo contrario, usar un valor por defecto de 120 BPM
    tempo_bpm = tempos[0].number if len(tempos) > 0 else 120

    return tempo_bpm



if __name__ == "__main__":
    l_s = 20
    c_a = "Audios/"
    cancion_a_continuar = "Audios/mond_3.mid"
    cant_predicciones = 100
    
    tempo_bpm = getTempo(cancion_a_continuar)
    print(tempo_bpm) ##CORREGIR ESTO QUE DA MALLL
    
    
    mapa_right = cargar_notas_acordes_canciones(c_a, "piano right")
    mapa_left = cargar_notas_acordes_canciones(c_a, "piano left")
    
    mlp_p_r, mlp_v_r, mlp_d_r = inicializar_modelo(c_a,l_s, mapa_right, "piano right")
    mlp_p_l, mlp_v_l, mlp_d_l = inicializar_modelo(c_a,l_s, mapa_left, "piano left")
    
    p_conprediccion_r, v_conprediccion_r, d_conprediccion_r = predecir_cancion(mlp_p_r, mlp_v_r, mlp_d_r, l_s, mapa_right, cancion_a_continuar, "piano right", cant_predicciones)
    p_conprediccion_l, v_conprediccion_l, d_conprediccion_l = predecir_cancion(mlp_p_l, mlp_v_l, mlp_d_l, l_s, mapa_left, cancion_a_continuar, "piano left", cant_predicciones)

    cancion_generada = generar_cancion([[p_conprediccion_r, v_conprediccion_r, d_conprediccion_r],[p_conprediccion_l, v_conprediccion_l, d_conprediccion_l]], tempo_bpm)
    cancion_generada.write('midi', fp='cancion_generada.mid')
    
    fragmento = generar_cancion([[p_conprediccion_r[0:l_s], v_conprediccion_r[0:l_s], d_conprediccion_r[0:l_s]],[p_conprediccion_l[0:l_s], v_conprediccion_l[0:l_s], d_conprediccion_l[0:l_s]]], tempo_bpm)
    fragmento.write('midi', fp='fragmento.mid')
    
    

