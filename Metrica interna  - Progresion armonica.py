from mapaNotasAcordes import cargar_notas_acordes_canciones
from music21 import converter, tempo, chord, note, instrument, stream,meter

from music21 import converter, meter

from music21 import converter, tempo, note, chord

from collections import Counter





def cargarPista2(archivo_midi):

    todos_caracteristicas = []
    try:
        # Llamada a la función para procesar todas las partes del archivo MIDI
        datos_partes = procesar_primera_pista2(archivo_midi)
        print(f'Archivo cargado: {archivo_midi}')
        todos_caracteristicas = datos_partes
    except FileNotFoundError:
        print(f'No se encontró el archivo: {archivo_midi}')
    return todos_caracteristicas


def procesar_primera_pista2(midi_file):
    

    try:
        midi_data = converter.parse(midi_file)
    except Exception as e:
        raise ValueError(f"No se pudo cargar el archivo MIDI: {e}")
    
    # Obtener todos los tempos del archivo MIDI
    tempos = midi_data.flatten().getElementsByClass(tempo.MetronomeMark)
    tempo_bpm = tempos[0].number if len(tempos) > 0 else 120  # Valor predeterminado: 120 BPM
    
    datos_partes = []
    
    # Procesar cada parte del archivo MIDI
    for parte in midi_data.parts:
        # Listas para almacenar los datos de la parte
        nombres = []       # Nombres de las notas, acordes o silencios
        pitches = []       # Alturas (pitches) en formato MIDI
        velocidades = []   # Velocidades de las notas/acordes
        duraciones = []    # Duraciones de los elementos en "quarterLength"
        
        # Recorrer los elementos de la parte
        for elemento in parte.flatten().notesAndRests:
            # if isinstance(elemento, chord.Chord):  # Si el elemento es un acorde
            #     nombres.append(f"Acorde: {', '.join(n.nameWithOctave for n in elemento.notes)}")
            #     pitches.append([n.pitch.midi for n in elemento.notes])
            #     velocidad_acorde = elemento.notes[0].volume.velocity if elemento.notes else 0
            #     velocidades.append(velocidad_acorde)
            #     duraciones.append(elemento.quarterLength)
            if isinstance(elemento, note.Note):  # Si el elemento es una nota
                nombres.append(elemento.nameWithOctave)
                pitches.append([elemento.pitch.midi])
                velocidades.append(elemento.volume.velocity)
                duraciones.append(elemento.quarterLength)
            # elif isinstance(elemento, note.Rest):  # Si el elemento es un silencio
            #     nombres.append("Silencio")
            #     pitches.append([-1])  # -1 para representar silencios
            #     velocidades.append(0)
            #     duraciones.append(elemento.quarterLength)
        
        # Agregar los datos de la parte procesada junto con el tempo
        datos_partes.append([nombres, pitches])
    
    return datos_partes




c_a = "Audios/"

  
  ##Si haces Audios
nombre_pista1 = "piano right"
nombre_pista2 = "piano left"



mapa_right, mapa_left = cargar_notas_acordes_canciones(c_a,nombre_pista1, nombre_pista2)


import os

def calcular_metrica_armonica_normalizada(pitches):
 
    if len(pitches) < 2:  # Si hay menos de dos notas, la métrica es 0
        return 0
    metrica = 0
    for i in range(len(pitches) - 1):
        intervalo = abs(pitches[i + 1] - pitches[i])
        metrica += intervalo
    return int(metrica / len(pitches))

import os
import csv


carpeta_midi = "_A"

# Diccionario para almacenar las métricas por archivo
metricas = {}

# Variable para contar las canciones
contador_canciones = 1

# Recorrer todos los archivos en la carpeta
for archivo in os.listdir(carpeta_midi):
    if archivo.endswith(".mid"):  # Solo procesar archivos MIDI
        archivo_midi = os.path.join(carpeta_midi, archivo)

        # Cargar características del archivo
        caracteristicas = cargarPista2(archivo_midi)

        # Extraer right y left
        caracteristicas_right = caracteristicas[0]
        caracteristicas_left = caracteristicas[1]

        # Convertir las alturas a índices en el mapa (right)
        for i, nota_acorde in enumerate(caracteristicas_right[1]):
            indice = mapa_right.index(sorted(nota_acorde))
            caracteristicas_right[1][i] = indice

        # Convertir las alturas a índices en el mapa (left)
        for i, nota_acorde in enumerate(caracteristicas_left[1]):
            indice = mapa_left.index(sorted(nota_acorde))
            caracteristicas_left[1][i] = indice

        # Calcular la métrica armónica para cada parte
        metrica_right = calcular_metrica_armonica_normalizada(caracteristicas_right[1])
        metrica_left = calcular_metrica_armonica_normalizada(caracteristicas_left[1])

        # Asignar un nombre a la canción (cancion1, cancion2, ...)
        nombre_cancion = f"cancion{contador_canciones}"

        # Almacenar las métricas en el diccionario con el nombre de la canción
        metricas[nombre_cancion] = {
            "metrica_right": metrica_right,
            "metrica_left": metrica_left
        }

        # Incrementar el contador para el siguiente archivo
        contador_canciones += 1

# Guardar las métricas en un archivo CSV
with open('metricas_canciones.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Escribir el encabezado (nombres de las columnas)
    writer.writerow(['Cancion', 'Metrica_Right', 'Metrica_Left'])
    
    # Escribir las métricas por cada canción
    for cancion, valores in metricas.items():
        writer.writerow([cancion, valores['metrica_right'], valores['metrica_left']])

# Imprimir las métricas calculadas
for cancion, valores in metricas.items():
    print(f"{cancion}: Right={valores['metrica_right']}, Left={valores['metrica_left']}")




##


import os
import csv
# Carpeta que contiene los archivos MIDI
carpeta_midi = "Audios/"

# Diccionario para almacenar las métricas por archivo
metricas = {}

# Recorrer todos los archivos en la carpeta
for archivo in os.listdir(carpeta_midi):
    if archivo.endswith(".mid"):  # Solo procesar archivos MIDI
        archivo_midi = os.path.join(carpeta_midi, archivo)

        # Cargar características del archivo
        caracteristicas = cargarPista2(archivo_midi)

        # Extraer right y left
        caracteristicas_right = caracteristicas[0]
        caracteristicas_left = caracteristicas[1]

        # Convertir las alturas a índices en el mapa (right)
        for i, nota_acorde in enumerate(caracteristicas_right[1]):
            indice = mapa_right.index(sorted(nota_acorde))
            caracteristicas_right[1][i] = indice

        # Convertir las alturas a índices en el mapa (left)
        for i, nota_acorde in enumerate(caracteristicas_left[1]):
            indice = mapa_left.index(sorted(nota_acorde))
            caracteristicas_left[1][i] = indice

        # Calcular la métrica armónica para cada parte
        metrica_right = calcular_metrica_armonica_normalizada(caracteristicas_right[1])
        metrica_left = calcular_metrica_armonica_normalizada(caracteristicas_left[1])

        # Almacenar las métricas en el diccionario
        metricas[archivo] = {
            "metrica_right": metrica_right,
            "metrica_left": metrica_left
        }

# Guardar las métricas en un archivo CSV
with open('metricas_midi.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Escribir el encabezado (nombres de las columnas)
    writer.writerow(['Archivo', 'Metrica_Right', 'Metrica_Left'])
    
    # Escribir las métricas por cada archivo MIDI
    for archivo, valores in metricas.items():
        writer.writerow([archivo, valores['metrica_right'], valores['metrica_left']])

# Imprimir las métricas calculadas
for archivo, valores in metricas.items():
    print(f"{archivo}: Right={valores['metrica_right']}, Left={valores['metrica_left']}")
    
    










