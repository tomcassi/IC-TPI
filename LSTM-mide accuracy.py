import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from mapaNotasAcordes import cargar_notas_acordes_canciones
from procesarMidi import generar_cancion,cargarPista,crear_secuencias,getTempo,getTimeSignature,calcular_longitud_secuencia


from tensorflow.keras.callbacks import EarlyStopping

def entrenar_modelo_lstm(X, y, modelo):
    # Configuración de Early Stopping
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Dividir los datos para validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, shuffle=True)
    
    # Entrenar el modelo
    modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1000,  # Máximo de épocas
        batch_size=128,
        #callbacks=[early_stopping],  # Agregar el callback
        verbose=1
    )
    return modelo




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




def inicializar_modelo(carpeta_audios, longitud_secuencia, notasyacordes, nombre_pieza):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Activation, Input
    from keras.utils import to_categorical
    from sklearn.metrics import accuracy_score
    import os
    import numpy as np

    # Definición de múltiples configuraciones de LSTM
    modelos = {
        "LSTM_128_dropout0.3": {
            "units": 128,
            "dropout": 0.3,
            "dense_units": 128
        },
        "LSTM_256_dropout0.3": {
            "units": 256,
            "dropout": 0.3,
            "dense_units": 256
        },
        "LSTM_256_dropout0.5": {
            "units": 256,
            "dropout": 0.5,
            "dense_units": 256
        },
        "LSTM_512_dropout0.3": {
            "units": 512,
            "dropout": 0.3,
            "dense_units": 512
        },
        "LSTM_512_dropout0.5": {
            "units": 512,
            "dropout": 0.5,
            "dense_units": 512
        },
    }

    accuracies = []

    print("\n=====Cargando características y calculando accuracies=====")

    for nombre_archivo in os.listdir(carpeta_audios):
        archivo_midi = os.path.join(carpeta_audios, nombre_archivo)

        # Cargar características desde el archivo MIDI
        todos_caracteristicas = cargarPista(archivo_midi, nombre_pieza)
        for i, nota_acorde in enumerate(todos_caracteristicas[0]):
            indice = notasyacordes.index(sorted(nota_acorde))
            todos_caracteristicas[0][i] = indice

        # Crear secuencias para entrenamiento
        X, y = crear_secuencias(todos_caracteristicas[0], longitud_secuencia)
        X = np.reshape(X, (len(X), longitud_secuencia, 1))
        y = to_categorical(y, num_classes=len(notasyacordes))

        # Entrenar y evaluar cada modelo
        for nombre_modelo, config in modelos.items():
            # Crear modelo con configuración específica
            modelo = Sequential([
                Input(shape=(longitud_secuencia, 1)),
                LSTM(config["units"], return_sequences=True),
                Dropout(config["dropout"]),
                LSTM(config["units"]),
                Dense(config["dense_units"]),
                Dropout(config["dropout"]),
                Dense(len(notasyacordes)),
                Activation('softmax')
            ])
            modelo.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

            # Entrenar el modelo
            modelo.fit(X, y, epochs=5, batch_size=32, verbose=0)

            # Evaluar el modelo
            y_pred = np.argmax(modelo.predict(X), axis=1)
            y_true = np.argmax(y, axis=1)

            accuracy = accuracy_score(y_true, y_pred)
            accuracies.append((nombre_archivo, nombre_modelo, accuracy))
            print(f"Archivo: {nombre_archivo}, Modelo: {nombre_modelo}, Accuracy: {accuracy:.4f}")

    return accuracies



def predecir_cancion(lstm_pitch,rf_velocity,rf_duration,longitud_secuencia, notasyacordes, cancion_inicial, nombre_pieza, cant_predicciones):
    #Predecir cancion:
    todos_caracteristicas = cargarPista(cancion_inicial, nombre_pieza)
    
    
    #===== Para pitch =====
    #transformo en indice de mapa:
    for i, nota_acorde in enumerate(todos_caracteristicas[0]):
        indice = notasyacordes.index(sorted(nota_acorde))
        todos_caracteristicas[0][i] = indice
    
    #predigo elementos:
    print("\n====Prediccion de Pitches====")
    pitches_conprediccion = predecir_sig_elem_lstm(todos_caracteristicas[0][0:longitud_secuencia], lstm_pitch, cant_predicciones,longitud_secuencia,len(notasyacordes))
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

    return pitches_conprediccion,velocities_conprediccion,durations_conprediccion






def predecir_sig_elem_lstm(elem_originales, modelo, cant_predicciones,longitud_secuencia,n_vocab):
    n_predicciones = 0
    elementos = elem_originales
    
    
    while n_predicciones < cant_predicciones:
          
        ##elem_input = np.array(elementos[n_predicciones:n_predicciones+len(elem_originales)]).reshape(1, -1)
        elem_input=np.reshape(elementos[n_predicciones:n_predicciones+len(elem_originales)], (1, longitud_secuencia, 1))
        #print("Input para predicción:", elem_input)
        
        prediccion = modelo.predict(elem_input)
        prediccion = prediccion[0]
        
        prediccion = modelo.predict(elem_input, verbose=0).flatten()
        prediccion = np.random.choice(range(n_vocab), p=prediccion)
        
       # print("Predicción:", prediccion)
        
        elementos.append(prediccion)
        n_predicciones += 1
    
    return elementos


def predecir_sig_elem_rf(elem_originales, modelo, cant_predicciones):
    n_predicciones = 0
    elementos = elem_originales
    
    
    while n_predicciones < cant_predicciones:
          
        elem_input = np.array(elementos[n_predicciones:n_predicciones+len(elem_originales)]).reshape(1, -1)
        #print("Input para predicción:", elem_input)
        
        prediccion = modelo.predict(elem_input)
        prediccion = prediccion[0]
        #print("Predicción:", prediccion)
        
        elementos.append(prediccion)
        n_predicciones += 1
    
    return elementos



# Parámetros iniciales
if __name__ == "__main__":
    from IPython import get_ipython
    get_ipython().magic('clear')
    tiempo_secuencia=3
    tiempo_a_predecir=60

    c_a = "Audios/"
    cancion_a_continuar = "Audios/pkelite4.mid"
    
    ##Si haces Audios
    # nombre_pista1 = "piano right"
    # nombre_pista2 = "piano left"
    
    #Si es para Audios2 
    nombre_pista1 = "right"
    nombre_pista2 = "left"
    
    
    l_s_r,l_s_l=calcular_longitud_secuencia(cancion_a_continuar, tiempo_secuencia,nombre_pista1,nombre_pista2)

    
    tempo_bpm = getTempo(cancion_a_continuar)
    firma_de_compas = getTimeSignature(cancion_a_continuar)




    
    print("\n=====Cargando acordes presentes en canciones=====")
    mapa_right, mapa_left = cargar_notas_acordes_canciones(c_a,nombre_pista1, nombre_pista2)
    
    
    accuracies_lstm_r = inicializar_modelo(c_a,l_s_r, mapa_right, nombre_pista1)
    accuracies_lstm_l= inicializar_modelo(c_a,l_s_l, mapa_left, nombre_pista2)
    
    
    cant_predicciones_r,cant_predicciones_l=calcular_longitud_secuencia(cancion_a_continuar, tiempo_a_predecir,nombre_pista1,nombre_pista2)

    
    p_conprediccion_r, v_conprediccion_r, d_conprediccion_r = predecir_cancion(lstm_p_r, rf_velocity_r, rf_duration_r, l_s_r, mapa_right, cancion_a_continuar, nombre_pista1, cant_predicciones_r)
    p_conprediccion_l, v_conprediccion_l, d_conprediccion_l = predecir_cancion(lstm_p_l, rf_velocity_l, rf_duration_l, l_s_l, mapa_left, cancion_a_continuar, nombre_pista2, cant_predicciones_l)
    
    
    cancion_nombre= 'cancion_generada_lstm.mid'
    cancion_generada = generar_cancion([[p_conprediccion_r, v_conprediccion_r, d_conprediccion_r],[p_conprediccion_l, v_conprediccion_l, d_conprediccion_l]], tempo_bpm,firma_de_compas,cancion_nombre)
  
   
    fragmento_nombre='fragmento_lstm.mid'
    fragmento = generar_cancion([[p_conprediccion_r[0:l_s_r], v_conprediccion_r[0:l_s_r], d_conprediccion_r[0:l_s_r]],[p_conprediccion_l[0:l_s_l], v_conprediccion_l[0:l_s_l], d_conprediccion_l[0:l_s_l]]], tempo_bpm,firma_de_compas,fragmento_nombre)

      # ##comentar/descomentar para todas las canciones

    # for cancion in os.listdir(c_a):
    #     cancion_a_continuar = cancion
    #     path_cancion_a_continuar = os.path.join(c_a, cancion_a_continuar)
       
    #     tempo_bpm = getTempo(path_cancion_a_continuar)
       
    #     p_conprediccion_r, v_conprediccion_r, d_conprediccion_r = predecir_cancion(lstm_p_r, rf_velocity_r, rf_duration_r, l_s, mapa_right, path_cancion_a_continuar, nombre_pista1, cant_predicciones)
    #     p_conprediccion_l, v_conprediccion_l, d_conprediccion_l = predecir_cancion(lstm_p_l, rf_velocity_l, rf_duration_l, l_s, mapa_left, path_cancion_a_continuar, nombre_pista2, cant_predicciones)
       
    #     cancion_generada = generar_cancion([[p_conprediccion_r, v_conprediccion_r, d_conprediccion_r],[p_conprediccion_l, v_conprediccion_l, d_conprediccion_l]], tempo_bpm)
    #     path_cancion_generada = os.path.join("Ejemplos", cancion_a_continuar)
    #     cancion_generada.write('midi', fp=path_cancion_generada)
       
    #     fragmento = generar_cancion([[p_conprediccion_r[0:l_s], v_conprediccion_r[0:l_s], d_conprediccion_r[0:l_s]],[p_conprediccion_l[0:l_s], v_conprediccion_l[0:l_s], d_conprediccion_l[0:l_s]]], tempo_bpm)
    #     path_fragmento = os.path.join("Ejemplos", cancion_a_continuar.replace('.mid', '_fragmento.mid'))
    #     fragmento.write('midi', fp=path_fragmento)
   



