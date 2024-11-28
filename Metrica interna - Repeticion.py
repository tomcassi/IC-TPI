import os
import csv
from music21 import converter, note, chord
from collections import Counter

# Función para extraer notas y acordes de una pista
def extraer_notas_y_acordes(pista):
    eventos = []
    for elemento in pista.flat.notes:
        if isinstance(elemento, note.Note):
            eventos.append((elemento.pitch.midi,))  # Nota individual como tupla de un elemento
        elif isinstance(elemento, chord.Chord):
            eventos.append(tuple(sorted([p.midi for p in elemento.pitches])))  # Acorde como tupla ordenada
    return eventos

# Función para cargar el archivo MIDI y procesar sus pistas
def cargar_midi(file_path):
    midi = converter.parse(file_path)
    repeticiones_por_pista = {}

    for i, part in enumerate(midi.parts):
        eventos = extraer_notas_y_acordes(part)
        repeticiones = contar_repeticiones_secuencias(eventos, 2)  # Contamos las secuencias de longitud 2
        repeticiones_por_pista[f'Pista {i+1}'] = repeticiones  # Guardamos las repeticiones por pista
    
    return repeticiones_por_pista

# Función para contar las repeticiones de secuencias de longitud n
def contar_repeticiones_secuencias(eventos, n):
    if len(eventos) < n:
        return {}
    secuencias = [tuple(eventos[i:i + n]) for i in range(len(eventos) - n + 1)]
    repeticiones = Counter(secuencias)
    return repeticiones

# Función para calcular la calificación basada en las repeticiones
def calcular_calificacion_repeticiones(repeticiones):
    total_repeticiones = sum(repeticiones.values())
    num_secuencias = len(repeticiones)
    
    if num_secuencias == 0:
        return 0  # Evitar división por cero si no hay secuencias
    
    promedio_repeticiones = total_repeticiones / num_secuencias
    return promedio_repeticiones

# Ruta de la carpeta donde se encuentran los archivos MIDI
carpeta_midi = 'Audios/'

# Lista de archivos MIDI en la carpeta
archivos_midi = [f for f in os.listdir(carpeta_midi) if f.endswith('.mid')]

# Lista para almacenar los resultados
resultados = []

# Procesar cada archivo MIDI
pistas_maximas = 0  # Para saber cuántas columnas de pista se necesitan
for archivo in archivos_midi:
    ruta_midi = os.path.join(carpeta_midi, archivo)
    print(f"Procesando: {ruta_midi}")
    
    # Obtener repeticiones por pista
    repeticiones_por_pista = cargar_midi(ruta_midi)
    
    # Calcular calificaciones por pista
    calificaciones = {pista: calcular_calificacion_repeticiones(repeticiones)
                      for pista, repeticiones in repeticiones_por_pista.items()}
    
    # Guardar los resultados en una fila
    fila = [archivo] + [calificaciones.get(f'Pista {i+1}', 0) for i in range(len(calificaciones))]
    resultados.append(fila)
    pistas_maximas = max(pistas_maximas, len(calificaciones))  # Actualizar el número máximo de pistas

# Guardar los resultados en un archivo CSV
ruta_csv = 'resultados_repeticiones_canciones_originales.csv'

with open(ruta_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Crear encabezados dinámicamente según el número máximo de pistas
    encabezados = ['Archivo'] + [f'Pista {i+1}' for i in range(pistas_maximas)]
    writer.writerow(encabezados)
    writer.writerows(resultados)

print(f"Resultados guardados en: {ruta_csv}")
