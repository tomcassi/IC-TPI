import os
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


from mapaNotasAcordes import cargar_notas_acordes_canciones
from procesarMidi import cargarPista, generar_cancion,getTempo ,crear_secuencias,getTimeSignature,calcular_longitud_secuencia

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



def entrenar_modelo_rf(X, y, rf):
    # Entrenar el modelo
    rf.fit(X, y)

    # Hacer predicciones
    y_pred = rf.predict(X)

    # Evaluar la precisión del modelo
    accuracy = accuracy_score(y, y_pred)
    print("Precisión del modelo:", accuracy)
        
    return rf, y_pred, y


def predecir_sig_elem_rf(elem_originales, modelo, cant_predicciones):
    n_predicciones = 0
    elementos = elem_originales
    
    
    while n_predicciones < cant_predicciones:
          
        elem_input = np.array(elementos[n_predicciones:n_predicciones+len(elem_originales)]).reshape(1, -1)
       # print("Input para predicción:", elem_input)
        
        prediccion = modelo.predict(elem_input)
        prediccion = prediccion[0]
       # print("Predicción:", prediccion)
        
        elementos.append(prediccion)
        n_predicciones += 1
    
    return elementos



def inicializar_modelo(carpeta_audios,longitud_secuencia, notasyacordes, nombre_pieza):

    modelos = {
    # Variaciones extensivas de Random Forest
    "RandomForest_50": RandomForestClassifier(
        n_estimators=50, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=36
    ),
    "RandomForest_100": RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=4, min_samples_leaf=1, random_state=36
    ),
    "RandomForest_200_depth10": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=2, random_state=36
    ),
    "RandomForest_500_depth15": RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=3, min_samples_leaf=1, random_state=36
    ),
    "RandomForest_1000_minLeaf3": RandomForestClassifier(
        n_estimators=1000, max_depth=None, min_samples_split=2, min_samples_leaf=3, random_state=36
    ),
    "RandomForest_300_depth20_minSplit6": RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=6, min_samples_leaf=1, random_state=36
    ),
    "RandomForest_150_depth5": RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=36
    ),
    "RandomForest_50_minSplit8_minLeaf5": RandomForestClassifier(
        n_estimators=50, max_depth=None, min_samples_split=8, min_samples_leaf=5, random_state=36
    ),
    "RandomForest_400_minSplit10_depth30": RandomForestClassifier(
        n_estimators=400, max_depth=30, min_samples_split=10, min_samples_leaf=2, random_state=36
    ),
    "RandomForest_250_depthNone_minLeaf2": RandomForestClassifier(
        n_estimators=250, max_depth=None, min_samples_split=4, min_samples_leaf=2, random_state=36
    ),
    "RandomForest_700_depthNone": RandomForestClassifier(
        n_estimators=700, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=36
    ),
    "RandomForest_500_depth25_minLeaf4": RandomForestClassifier(
        n_estimators=500, max_depth=25, min_samples_split=5, min_samples_leaf=4, random_state=36
    ),
    "RandomForest_300_depth15_minLeaf6": RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=8, min_samples_leaf=6, random_state=36
    ),
    "RandomForest_100_depth20_minLeaf2": RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_split=3, min_samples_leaf=2, random_state=36
    ),
    "RandomForest_200_depth30_minLeaf3": RandomForestClassifier(
        n_estimators=200, max_depth=30, min_samples_split=4, min_samples_leaf=3, random_state=36
    ),

    # Variaciones de MLPClassifier
    "MLP_50": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
    "MLP_100": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "MLP_50_20": MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=500, random_state=42),
    "MLP_100_50": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    "MLP_150_relu": MLPClassifier(hidden_layer_sizes=(150,), activation='relu', max_iter=500, random_state=42),
    "MLP_50_50_adam": MLPClassifier(hidden_layer_sizes=(50, 50), solver='adam', max_iter=500, random_state=42),
    "MLP_200_tanh": MLPClassifier(hidden_layer_sizes=(200,), activation='tanh', max_iter=500, random_state=42),
    "MLP_100_25_relu": MLPClassifier(hidden_layer_sizes=(100, 25), activation='relu', max_iter=1000, random_state=42),
    "MLP_50_50_lbfgs": MLPClassifier(hidden_layer_sizes=(50, 50), solver='lbfgs', max_iter=1000, random_state=42),
    "MLP_100_adaptive": MLPClassifier(hidden_layer_sizes=(100,), solver='adam', learning_rate='adaptive', max_iter=500, random_state=42),
    "MLP_50_20_10": MLPClassifier(hidden_layer_sizes=(50, 20, 10), max_iter=1000, random_state=42),
    "MLP_200_relu_alpha1e-4": MLPClassifier(hidden_layer_sizes=(200,), activation='relu', alpha=1e-4, max_iter=500, random_state=42),
    "MLP_300_150_tanh": MLPClassifier(hidden_layer_sizes=(300, 150), activation='tanh', max_iter=500, random_state=42),
    "MLP_400_logistic": MLPClassifier(hidden_layer_sizes=(400,), activation='logistic', max_iter=1000, random_state=42),
    "MLP_250_100_50_adam": MLPClassifier(hidden_layer_sizes=(250, 100, 50), solver='adam', max_iter=500, random_state=42),
}
    
    accuracies = []
    
    print("\n=====Cargando caracteristicas y calculando acc=====")
    
    for nombre_archivo in os.listdir(carpeta_audios):
        archivo_midi = os.path.join(carpeta_audios, nombre_archivo)
        
        todos_caracteristicas = cargarPista(archivo_midi, nombre_pieza)
        for i, nota_acorde in enumerate(todos_caracteristicas[0]):
            indice = notasyacordes.index(sorted(nota_acorde))
            todos_caracteristicas[0][i] = indice
        
        X,y = crear_secuencias(todos_caracteristicas[0],longitud_secuencia)
        
        
        # Entrenar y evaluar cada modelo
        for nombre_modelo, modelo in modelos.items():
            # Entrenar el modelo y obtener predicciones
            modelo_entrenado, y_pred, y_test = entrenar_modelo_rf(X, y, modelo)

            # Calcular accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Almacenar el resultado
            accuracies.append((nombre_modelo, accuracy))
            print(f"Archivo: {nombre_archivo}, Modelo: {nombre_modelo}, Accuracy: {accuracy:.4f}")

    return accuracies


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
    
    #Segundos que se van a tomar para tomar secuencia
    tiempo_secuencia=10
    tiempo_a_predecir= 60
    
    c_a = "Audios/"
    cancion_a_continuar = "Audios/pkelite4.mid"
    
    

    nombre_pista1 = "right"
    nombre_pista2 = "left"
    
    
    # # #Si haces Audios
    # nombre_pista1 = "piano right"
    # nombre_pista2 = "piano left"
    
    
    l_s_r,l_s_l=calcular_longitud_secuencia(cancion_a_continuar, tiempo_secuencia,nombre_pista1,nombre_pista2)
    
    tempo_bpm = getTempo(cancion_a_continuar)
    firma_de_compas = getTimeSignature(cancion_a_continuar)


    cant_predicciones_r,cant_predicciones_l=calcular_longitud_secuencia(cancion_a_continuar, tiempo_a_predecir,nombre_pista1,nombre_pista2)
 
    


    
    print("\n=====Cargando acordes presentes en canciones=====")
    mapa_right, mapa_left = cargar_notas_acordes_canciones(c_a,nombre_pista1, nombre_pista2)
    
    accuracies_r = inicializar_modelo(c_a,l_s_r, mapa_right, nombre_pista1)
    accuracies_l = inicializar_modelo(c_a,l_s_l, mapa_left, nombre_pista2)
    
    
    archivo_csv = "resultados_modelos_r.csv"

    with open(archivo_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Escribir encabezado
        writer.writerow(["Modelo", "Accuracy"])
        # Escribir los resultados
        writer.writerows(accuracies_r)
        
    archivo_csv = "resultados_modelos_l.csv"

    with open(archivo_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Escribir encabezado
        writer.writerow(["Modelo", "Accuracy"])
        # Escribir los resultados
        writer.writerows(accuracies_l)

print(f"Resultados exportados a {archivo_csv}")
    
    # p_conprediccion_r, v_conprediccion_r, d_conprediccion_r = predecir_cancion(rf_p_r, rf_v_r, rf_d_r, l_s_r, mapa_right, cancion_a_continuar, nombre_pista1, cant_predicciones_r)
    # p_conprediccion_l, v_conprediccion_l, d_conprediccion_l = predecir_cancion(rf_p_l, rf_v_l, rf_d_l, l_s_l, mapa_left, cancion_a_continuar, nombre_pista2, cant_predicciones_l)

    # cancion_nombre= 'cancion_generada_rf.mid'
    # cancion_generada = generar_cancion([[p_conprediccion_r, v_conprediccion_r, d_conprediccion_r],[p_conprediccion_l, v_conprediccion_l, d_conprediccion_l]], tempo_bpm,firma_de_compas,cancion_nombre)
    # #cancion_generada.write('midi', fp='cancion_generada_rf.mid')
    
    # fragmento_nombre='fragmento_rf.mid'
    # fragmento = generar_cancion([[p_conprediccion_r[0:l_s_r], v_conprediccion_r[0:l_s_r], d_conprediccion_r[0:l_s_r]],[p_conprediccion_l[0:l_s_l], v_conprediccion_l[0:l_s_l], d_conprediccion_l[0:l_s_l]]], tempo_bpm,firma_de_compas,fragmento_nombre)
   
    
   
    
   
    #fragmento.write('midi', fp='fragmento.mid')
    
        
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
    
    
