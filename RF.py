import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


from mapaNotasAcordes import cargar_notas_acordes_canciones
from procesarMidi import cargarPista, generar_cancion,getTempo ,crear_secuencias





def entrenar_modelo_rf(X, y, rf):
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)


    # Entrenar el modelo
    rf.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = rf.predict(X_test)

    # Evaluar la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión del modelo:", accuracy)
        
    return rf, y_pred, y_test


def predecir_sig_elem_rf(elem_originales, modelo, cant_predicciones):
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

    
    rf_pitch = RandomForestClassifier(n_estimators=100)
    rf_velocity = RandomForestClassifier(n_estimators=100)
    rf_duration = RandomForestClassifier(n_estimators=100)
    
    
    
    print("\n=====Cargando caracteristicas=====")
    
    for nombre_archivo in os.listdir(carpeta_audios):
        archivo_midi = os.path.join(carpeta_audios, nombre_archivo)
        
        todos_caracteristicas = cargarPista(archivo_midi, nombre_pieza)
        for i, nota_acorde in enumerate(todos_caracteristicas[0]):
            indice = notasyacordes.index(sorted(nota_acorde))
            todos_caracteristicas[0][i] = indice
        
        X,y = crear_secuencias(todos_caracteristicas[0],longitud_secuencia)
        
        rf_pitch, y_pred, y_test = entrenar_modelo_rf(X,y,rf_pitch)
        
        X,y = crear_secuencias(todos_caracteristicas[1],longitud_secuencia)
        
        rf_velocity, y_pred, y_test = entrenar_modelo_rf(X,y,rf_velocity)
        
        X,y = crear_secuencias(todos_caracteristicas[2],longitud_secuencia)
        # Multiplicar cada valor dentro de X por 100 y convertir a int
        X = [[int(valor * 1000) for valor in sublista] for sublista in X]
        
        # Multiplicar cada valor en y por 100 y convertir a int
        y = [int(valor * 1000) for valor in y]
        
        rf_duration, y_pred, y_test = entrenar_modelo_rf(X,y,rf_duration)
        
    return rf_pitch, rf_velocity, rf_duration


def predecir_cancion(rf_pitch, rf_velocity, rf_duration, longitud_secuencia, notasyacordes, cancion_inicial, nombre_pieza, cant_predicciones):
    #Predecir cancion:
    todos_caracteristicas = cargarPista(cancion_inicial, nombre_pieza)
    
    
    #===== Para pitch =====
    #transformo en indice de mapa:
    for i, nota_acorde in enumerate(todos_caracteristicas[0]):
        indice = notasyacordes.index(sorted(nota_acorde))
        todos_caracteristicas[0][i] = indice
    
    #predigo elementos:
    print("\n====Prediccion de Pitches====")
    pitches_conprediccion = predecir_sig_elem_rf(todos_caracteristicas[0][0:longitud_secuencia], rf_pitch, cant_predicciones)
    #vuelvo a transformar de indice a notas y acordes
    for i in range(len(pitches_conprediccion)):
        pitches_conprediccion[i] = notasyacordes[pitches_conprediccion[i]]
    
    
    #===== Para velocities =====
    print("\n====Prediccion de Velocities====")
    velocities_conprediccion = predecir_sig_elem_rf(todos_caracteristicas[1][0:longitud_secuencia], rf_velocity, cant_predicciones)
    
    #===== Para durations =====
    print("\n====Prediccion de Durations====")
    #multiplico por 1000 y transformo en enteros
    durations_originales = todos_caracteristicas[2][0:longitud_secuencia]
    for i in range(len(durations_originales)):
        durations_originales[i] *= 1000
    durations_conprediccion = predecir_sig_elem_rf(durations_originales, rf_duration, cant_predicciones)
    for i in range(len(durations_conprediccion)):
        durations_conprediccion[i] /= 1000

    return pitches_conprediccion, velocities_conprediccion, durations_conprediccion




if __name__ == "__main__":
    from IPython import get_ipython
    get_ipython().magic('clear')
    
    l_s = 20
    c_a = "Audios2/"
    cancion_a_continuar = "Audios2/figaro.mid"
    cant_predicciones = 200
    nombre_pista1 = "right"
    nombre_pista2 = "left"
    
    
    # #Si haces Audios
    # nombre_pista1 = "piano right"
    # nombre_pista2 = "piano left"
    
    
    tempo_bpm = getTempo(cancion_a_continuar)
    
    print("\n=====Cargando acordes presentes en canciones=====")
    mapa_right, mapa_left = cargar_notas_acordes_canciones(c_a,nombre_pista1, nombre_pista2)
    
    rf_p_r, rf_v_r, rf_d_r = inicializar_modelo(c_a,l_s, mapa_right, nombre_pista1)
    rf_p_l, rf_v_l, rf_d_l = inicializar_modelo(c_a,l_s, mapa_left, nombre_pista2)
    
    p_conprediccion_r, v_conprediccion_r, d_conprediccion_r = predecir_cancion(rf_p_r, rf_v_r, rf_d_r, l_s, mapa_right, cancion_a_continuar, nombre_pista1, cant_predicciones)
    p_conprediccion_l, v_conprediccion_l, d_conprediccion_l = predecir_cancion(rf_p_l, rf_v_l, rf_d_l, l_s, mapa_left, cancion_a_continuar, nombre_pista2, cant_predicciones)

    cancion_generada = generar_cancion([[p_conprediccion_r, v_conprediccion_r, d_conprediccion_r],[p_conprediccion_l, v_conprediccion_l, d_conprediccion_l]], tempo_bpm)
    cancion_generada.write('midi', fp='cancion_generada_rf.mid')
    
    fragmento = generar_cancion([[p_conprediccion_r[0:l_s], v_conprediccion_r[0:l_s], d_conprediccion_r[0:l_s]],[p_conprediccion_l[0:l_s], v_conprediccion_l[0:l_s], d_conprediccion_l[0:l_s]]], tempo_bpm)
    fragmento.write('midi', fp='fragmento.mid')
    
        
    # ##comentar/descomentar para todas las canciones

    # for cancion in os.listdir(c_a):
    #     cancion_a_continuar = cancion
    #     path_cancion_a_continuar = os.path.join(c_a, cancion_a_continuar)
        
    #     tempo_bpm = getTempo(path_cancion_a_continuar)
        
    #     p_conprediccion_r, v_conprediccion_r, d_conprediccion_r = predecir_cancion(rf_p_r, rf_v_r, rf_d_r, l_s, mapa_right, path_cancion_a_continuar, nombre_pista1, cant_predicciones)
    #     p_conprediccion_l, v_conprediccion_l, d_conprediccion_l = predecir_cancion(rf_p_l, rf_v_l, rf_d_l, l_s, mapa_left, path_cancion_a_continuar, nombre_pista2, cant_predicciones)
    
    #     cancion_generada = generar_cancion([[p_conprediccion_r, v_conprediccion_r, d_conprediccion_r],[p_conprediccion_l, v_conprediccion_l, d_conprediccion_l]], tempo_bpm)
    #     path_cancion_generada = os.path.join("Ejemplos/", cancion_a_continuar)
    #     cancion_generada.write('midi', fp=path_cancion_generada)
        
    #     fragmento = generar_cancion([[p_conprediccion_r[0:l_s], v_conprediccion_r[0:l_s], d_conprediccion_r[0:l_s]],[p_conprediccion_l[0:l_s], v_conprediccion_l[0:l_s], d_conprediccion_l[0:l_s]]], tempo_bpm)
    #     path_fragmento = os.path.join("Ejemplos/", cancion_a_continuar.replace('.mid', '_fragmento.mid'))
    #     fragmento.write('midi', fp=path_fragmento)
    
    