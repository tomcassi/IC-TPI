# Importamos las bibliotecas necesarias 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
import pretty_midi
import os
import matplotlib.pyplot as plt
import random

plt.close('all')

# Función para crear el modelo LSTM

# Parámetros principales
secuencia_len = 5  # Número de notas en cada secuencia de entrada

# Definir la carpeta donde estoy parado
carpeta_principal = r'C:\Users\Rama\Desktop\IC-TPI-LSTM'  # Ruta a la carpeta principal
carpeta_audios=carpeta_principal + '\Audios'
carpeta_graficas=carpeta_principal + '\Graficas'
carpeta_salida= carpeta_principal+ '\Pruebas'

# Función para extraer notas de un archivo MIDI
def extraer_notas_de_midi(archivo):
    midi_data = pretty_midi.PrettyMIDI(archivo)  # Cargamos el archivo MIDI
    notas = []  # Lista para almacenar las notas extraídas
    # Procesamos solo el primer instrumento
    if midi_data.instruments:
        instrumento = midi_data.instruments[0]  # Seleccionamos el primer instrumento
        for nota in instrumento.notes:
            notas.append((nota.pitch, nota.velocity, nota.end - nota.start))  # Guardamos pitch, velocidad y duración
    return notas

# Cargamos todas las canciones en una lista de notas
notas = []
for i in range(1, 10):  # De 1 a 10 para cargar los audios
    archivo_midi = os.path.join(carpeta_audios, f"beethoven{i}.mid")  # Ruta completa del archivo MIDI
    try:
        notas += extraer_notas_de_midi(archivo_midi)  # Agregamos las notas de cada archivo a la lista total
        print(f'Archivo cargado: {archivo_midi}')
    except FileNotFoundError:
        print(f'No se encontró el archivo: {archivo_midi}')

# Crear listas para las secuencias de entrada y sus salidas correspondientes
secuencias_entrada_pitch = []  # Lista para las secuencias de entrada para pitch
secuencias_entrada_velocity = []  # Lista para las secuencias de entrada para velocidad
secuencias_entrada_duration = []  # Lista para las secuencias de entrada para duración
secuencias_salida_pitch = []  # Lista para la salida de pitch
secuencias_salida_velocity = []  # Lista para la salida de velocidad
secuencias_salida_duration = []  # Lista para la salida de duración

# Creamos las secuencias para entrenamiento, dividiendo el conjunto de notas en secuencias de 5 notas
for i in range(len(notas) - secuencia_len):  
    secuencia = notas[i:i + secuencia_len]  # Cada secuencia de 5 notas
    salida = notas[i + secuencia_len]  # La nota que sigue a esta secuencia

    # Añadir a las listas correspondientes
    secuencias_entrada_pitch.append([nota[0] for nota in secuencia])  # Altura
    secuencias_entrada_velocity.append([nota[1] for nota in secuencia])  # Velocidad
    secuencias_entrada_duration.append([nota[2] for nota in secuencia])  # Duración

    # Añadimos la salida correspondiente, que es la siguiente nota
    secuencias_salida_pitch.append(salida[0])  # Salida de altura
    secuencias_salida_velocity.append(salida[1])  # Salida de velocidad
    secuencias_salida_duration.append(salida[2])  # Salida de duración

# Comprobar las longitudes de las listas
print(f'Longitud de secuencias de entrada: {len(secuencias_entrada_pitch)}')
print(f'Longitud de secuencias de salida: {len(secuencias_salida_pitch)}')



# Convertimos las listas a matrices numpy para su uso en la LSTM
X_pitch = np.array(secuencias_entrada_pitch)  # Matriz de secuencias de entrada para altura
X_velocity = np.array(secuencias_entrada_velocity)  # Matriz de secuencias de entrada para velocidad
X_duration = np.array(secuencias_entrada_duration)  # Matriz de secuencias de entrada para duración
y_pitch = np.array(secuencias_salida_pitch)  # Vector de salidas de altura
y_velocity = np.array(secuencias_salida_velocity)  # Vector de salidas de velocidad
y_duration = np.array(secuencias_salida_duration)  # Vector de salidas de duración


# Función para crear el modelo LSTM
def crear_modelo(input_shape):
    modelo = Sequential()
    
    # Primera capa LSTM con Dropout
    modelo.add(LSTM(128, input_shape=input_shape, return_sequences=True))  # Capa LSTM que devuelve secuencias
    modelo.add(Dropout(0.2))  # Aplicar un Dropout del 20% para prevenir el sobreajuste
    
    # Segunda capa LSTM
    modelo.add(LSTM(128))  # Capa LSTM que no devuelve secuencias
    modelo.add(Dropout(0.2))  # Aplicar un Dropout del 20% nuevamente
    
    # Capa de salida
    modelo.add(Dense(1))  # Capa densa de salida, ajusta según tus necesidades
    
    # Compilación del modelo
    modelo.compile(loss='mean_squared_error', optimizer='adam')  # Puedes ajustar la función de pérdida y optimizador según sea necesario

    return modelo




#Para pasarle los datos tiene que estar con el formato  (n_samples, n_timesteps, n_features).

n_timesteps = secuencia_len  # longitud de la secuencia
n_features = 1  # número de características (1 para cada pitch, velocity, duration)

# Reshape para X_pitch
X_pitch = X_pitch.reshape((X_pitch.shape[0], n_timesteps, n_features))

# Reshape para X_velocity
X_velocity = X_velocity.reshape((X_velocity.shape[0], n_timesteps, n_features))

# Reshape para X_duration
X_duration = X_duration.reshape((X_duration.shape[0], n_timesteps, n_features))


# Crear el modelo para pitch
modelo_pitch = crear_modelo((X_pitch.shape[1], 1))  # Modelo para altura
# Compilar el modelo para pitch
modelo_pitch.compile(loss='mean_squared_error', optimizer='adam')
# Entrenamos el modelo para pitch

# Entrenar el modelo y guardar el historial
historial_pitch = modelo_pitch.fit(X_pitch, y_pitch, epochs=100, batch_size=64)

# Extraer la pérdida y otras métricas del historial
perdida_pitch = historial_pitch.history['loss']
 





# Crear el modelo para velocidad
modelo_velocity = crear_modelo((X_velocity.shape[1], 1))  # Modelo para velocidad
# Compilar el modelo para velocidad
modelo_velocity.compile(loss='mean_squared_error', optimizer='adam')
# Entrenamos el modelo para velocidad
historial_velocity=modelo_velocity.fit(X_velocity, y_velocity, epochs=100, batch_size=64)


# Extraer la pérdida y otras métricas del historial
perdida_velocity = historial_velocity.history['loss']




# Crear el modelo para duración
modelo_duration = crear_modelo((X_duration.shape[1], 1))  # Modelo para duración
# Compilar el modelo para duración
modelo_duration.compile(loss='mean_squared_error', optimizer='adam')
# Entrenamos el modelo para duración


# Entrenar el modelo y guardar el historial
historial_duration = modelo_duration.fit(X_duration, y_duration, epochs=100, batch_size=64)

# Extraer la pérdida y otras métricas del historial
perdida_duration = historial_duration.history['loss']




plt.figure(1)
plt.plot(perdida_pitch)
plt.title('Pérdida durante el entrenamiento (Pitch)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.show()
plt.savefig(os.path.join(carpeta_graficas, 'perdida_pitch.png'))  # Guardar como PNG


plt.figure(2)
plt.plot(perdida_velocity)
plt.title('Pérdida durante el entrenamiento (Velocity)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.show()
plt.savefig(os.path.join(carpeta_graficas, 'perdida_velocity.png'))  # Guardar como PNG


plt.figure(3)
plt.plot(perdida_duration)
plt.title('Pérdida durante el entrenamiento (Duration)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.show()
plt.savefig(os.path.join(carpeta_graficas, 'perdida_duration.png'))  # Guardar como PNG





# Función para generar nuevas notas
def generar_nueva_secuencia(secuencia_inicial, modelo_pitch, modelo_velocity, modelo_duration, num_notas_a_predecir):
    nueva_secuencia = secuencia_inicial.copy()  # Hacemos una copia de la secuencia inicial
    
    # Generar nuevas notas
    for i in range(num_notas_a_predecir):  # Cambiamos el rango para evitar IndexError
        if len(nueva_secuencia) >= 5:  # Aseguramos que haya al menos 5 notas en la secuencia
            entrada = nueva_secuencia[i:i + 5]  # Obtener la ventana de 5 notas
            
            # Extraer características de la entrada
            entradas_pitch = np.array([pitch_aux[0] for pitch_aux in entrada]).reshape(1, 5, 1)  # Reshape para LSTM
            entradas_velocity = np.array([velocity_aux[1] for velocity_aux in entrada]).reshape(1, 5, 1)
            entradas_duration = np.array([duration_aux[2] for duration_aux in entrada]).reshape(1, 5, 1)
            
            # Predecir nuevas notas
            nuevo_pitch = modelo_pitch.predict(entradas_pitch, verbose=0)[0][0]
            nuevo_velocity = modelo_velocity.predict(entradas_velocity, verbose=0)[0][0]
            nuevo_duration = modelo_duration.predict(entradas_duration, verbose=0)[0][0]

            # Agregar las nuevas notas a la secuencia
            nueva_secuencia.append([nuevo_pitch, nuevo_velocity, nuevo_duration])

    return nueva_secuencia

# Secuencia inicial para generación
secuencia_len = 5  # Definir el tamaño de la secuencia inicial

inicio_random = random.randint(0, len(notas) - secuencia_len )


secuencia_inicial = notas[inicio_random:inicio_random+secuencia_len]  # Comenzamos con las primeras 'secuencia_len' notas
num_notas_a_predecir = 100

# Generar nuevas notas
nueva_musica = generar_nueva_secuencia(secuencia_inicial, modelo_pitch, modelo_velocity, modelo_duration, num_notas_a_predecir)


def guardar_en_midi(nueva_secuencia, nombre_archivo):
    
    
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo)

    nuevo_midi = pretty_midi.PrettyMIDI()  # Crear un nuevo objeto PrettyMIDI
    instrumento = pretty_midi.Instrument(program=0)  # Crear un nuevo instrumento (por ejemplo, piano)


    # Variable para almacenar el tiempo actual
    tiempo_actual = 0

    # Agregar las notas a nuestro instrumento
    for nota in nueva_secuencia:
        pitch_auxiliar = int(nota[0])  # Obtener el pitch (nota)
        velocity_auxiliar = int(nota[1])  # Obtener la velocidad
        duration_auxiliar = float(nota[2])  # Obtener la duración de la nota desde la secuencia

        # Crear una nueva nota MIDI
        nueva_nota = pretty_midi.Note(pitch=pitch_auxiliar,velocity=velocity_auxiliar, start=tiempo_actual, end=tiempo_actual + duration_auxiliar)
        instrumento.notes.append(nueva_nota)

        # Avanzar el tiempo actual
        tiempo_actual += duration_auxiliar  # Aumentar el tiempo por la duración de la nota

    # Añadir el instrumento al objeto MIDI
    nuevo_midi.instruments.append(instrumento)

    # Guardar el archivo MIDI
    nuevo_midi.write(ruta_salida)

# Usar la función para guardar las nuevas notas en un archivo MIDI
guardar_en_midi(nueva_musica, 'nueva_musica.mid')