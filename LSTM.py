# Importamos las bibliotecas necesarias
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pretty_midi
import os

# Parámetros principales
secuencia_len = 10  # Número de notas en cada secuencia de entrada
num_notas = 128     # Número total de notas posibles (usando 128 para representar notas MIDI)

# Definir la carpeta donde están los archivos MIDI
carpeta = r'C:\Users\Rama\Desktop\b\IC-TPI\Beto'  # Ruta a la carpeta con los archivos MIDI

# Función para extraer notas de un archivo MIDI
def extraer_notas_de_midi(archivo):
    midi_data = pretty_midi.PrettyMIDI(archivo)  # Cargamos el archivo MIDI
    notas = []  # Lista para almacenar las notas extraídas

    # Procesamos solo el primer instrumento
    if midi_data.instruments:
        instrumento = midi_data.instruments[0]  # Seleccionamos el primer instrumento
        for nota in instrumento.notes:
            notas.append(nota.pitch)  # Guardamos solo la altura de la nota (pitch)
    return notas

# Cargamos todas las canciones en una lista de notas
notas = []
for i in range(1, 10):  # De 1 a 30 para cargar 'beethoven1.mid' hasta 'beethoven30.mid'
    archivo_midi = os.path.join(carpeta, f"beethoven{i}.mid")  # Ruta completa del archivo MIDI
    try:
        notas += extraer_notas_de_midi(archivo_midi)  # Agregamos las notas de cada archivo a la lista total
        print(f'Archivo cargado: {archivo_midi}')
    except FileNotFoundError:
        print(f'No se encontró el archivo: {archivo_midi}')

# Crear listas para las secuencias de entrada y sus salidas correspondientes
secuencias_entrada = []  # Lista para las secuencias de entrada
secuencias_salida = []   # Lista para la nota siguiente (etiqueta) de cada secuencia

# Creamos las secuencias para entrenamiento, dividiendo el conjunto de notas en secuencias de 10 notas
for i in range(len(notas) - secuencia_len):
    secuencia = notas[i:i + secuencia_len]       # Cada secuencia de 10 notas
    salida = notas[i + secuencia_len]            # La nota que sigue a esta secuencia
    secuencias_entrada.append(secuencia)         # Añadir la secuencia a las entradas
    secuencias_salida.append(salida)             # Añadir la nota siguiente a las salidas

# Convertimos las listas a matrices numpy para su uso en la LSTM
X = np.array(secuencias_entrada)  # Matriz de secuencias de entrada
y = np.array(secuencias_salida)   # Vector de salidas correspondientes

# Reshape de X para que la LSTM lo interprete en formato (muestras, pasos, características)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Un canal de característica (1) por nota

# Normalizamos los valores de las notas a [0,1] dividiendo por el número total de notas posibles
X = X / float(num_notas)

# Convertimos las salidas (notas siguientes) en categorías usando one-hot encoding
y = to_categorical(y, num_classes=num_notas)  # Cada salida es un vector de longitud igual al número de notas posibles

# Definición del modelo LSTM
modelo = Sequential()  # Creamos un modelo secuencial
# Primera capa LSTM con 128 unidades y retorno de secuencias (para conectar con la siguiente capa LSTM)
modelo.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# - 128: Número de neuronas en esta capa LSTM.
# - input_shape: forma de la entrada, donde X.shape[1] es el número de pasos (10) y X.shape[2] es el número de características (1 en este caso).
# - return_sequences=True: significa que esta capa devolverá las secuencias completas en lugar de solo la última salida.

# Segunda capa LSTM con 128 unidades
modelo.add(LSTM(128))
# - Esta capa también tiene 128 neuronas, pero no devuelve las secuencias completas porque es la última capa LSTM.

# Capa de salida completamente conectada, con softmax para clasificación de la siguiente nota
modelo.add(Dense(num_notas, activation='softmax'))
# - Dense: Capa completamente conectada que toma la salida de la última capa LSTM.
# - num_notas: Número de neuronas en esta capa es igual al número total de notas posibles (128).
# - activation='softmax': Función de activación que convierte las salidas en probabilidades que suman 1, útil para clasificación.

# Compilación del modelo usando función de pérdida de entropía cruzada y optimizador Adam
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# - loss='categorical_crossentropy': Función de pérdida utilizada para problemas de clasificación multiclase.
# - optimizer='adam': Algoritmo de optimización que ajusta los pesos del modelo durante el entrenamiento.
# - metrics=['accuracy']: Métrica para evaluar el rendimiento del modelo durante el entrenamiento.

# Entrenamos el modelo con 100 épocas y un tamaño de lote de 64
modelo.fit(X, y, epochs=100, batch_size=64)
# - epochs=100: Número de veces que el modelo verá todo el conjunto de datos durante el entrenamiento.
# - batch_size=64: Número de muestras que se procesarán antes de actualizar los pesos del modelo.

# Secuencia inicial para generación (puede ser cualquiera de las secuencias de entrenamiento)
secuencia_inicial = notas[:secuencia_len]  # Comenzamos con las primeras 10 notas
nueva_musica = secuencia_inicial.copy()     # Lista donde se almacenará la música generada

# Generación de una nueva secuencia de 100 notas a partir de las notas iniciales
for _ in range(100):
    # Convertimos la secuencia actual en el formato de entrada adecuado para el modelo
    entrada = np.array(secuencia_inicial)  # Convertimos la secuencia inicial en un array
    entrada = np.reshape(entrada, (1, secuencia_len, 1))  # Redimensionar para LSTM
    entrada = entrada / float(num_notas)                  # Normalizar como en el entrenamiento

    # Predecimos la próxima nota
    prediccion = modelo.predict(entrada, verbose=0)  # Output es un vector de probabilidades
    siguiente_nota = np.argmax(prediccion)           # Seleccionamos la nota con mayor probabilidad

    # Añadimos la nota predicha a la nueva música
    nueva_musica.append(siguiente_nota)

    # Actualizamos la secuencia de entrada eliminando la primera nota y agregando la predicción
    secuencia_inicial = secuencia_inicial[1:] + [siguiente_nota]

# Mostrar la secuencia de música generada
print("Nueva secuencia de música generada:", nueva_musica)

# Función para guardar notas en un archivo MIDI
def guardar_en_midi(notas, nombre_archivo='nueva_secuencia.mid'):
    midi = pretty_midi.PrettyMIDI()  # Creamos un objeto PrettyMIDI
    instrumento = pretty_midi.Instrument(program=0)  # Usamos el piano (program 0)

    # Duración de cada nota (en segundos)
    duracion_nota = 0.5  # Cada nota sonará durante 0.5 segundos

    # Agregar las notas con tiempos específicos
    for i, nota in enumerate(notas):
        start_time = i * duracion_nota  # Tiempo de inicio basado en el índice de la nota
        nueva_nota = pretty_midi.Note(velocity=100, pitch=nota, start=start_time, end=start_time + duracion_nota)  # Definimos tiempo de inicio y fin
        instrumento.notes.append(nueva_nota)  # Añadir la nota al instrumento

    # Añadir el instrumento al objeto MIDI
    midi.instruments.append(instrumento)

    # Guardar el archivo MIDI
    midi.write(nombre_archivo)  # Guardamos el archivo MIDI con el nombre especificado

# Llamamos a la función para guardar la secuencia generada
guardar_en_midi(nueva_musica)

print(f"Secuencia guardada en el archivo MIDI: nueva_secuencia.mid")
