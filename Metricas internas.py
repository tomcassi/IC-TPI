from mapaNotasAcordes import cargar_notas_acordes_canciones,añadir_acordes_mapa
from procesarMidi import generar_cancion,cargarPista,crear_secuencias,getTempo,getTimeSignature,calcular_longitud_secuencia,procesar_primera_pista
from music21 import converter, tempo, chord, note, instrument, stream,meter
from music21 import converter, note, stream

from music21 import converter, meter

from music21 import converter, tempo, note, chord

def cargarPista2(archivo_midi):
    """
    Función para cargar un archivo MIDI y procesar todas las partes contenidas en él.
    
    Parámetros:
        archivo_midi (str): Ruta del archivo MIDI.
    
    Retorna:
        todos_caracteristicas (list): Una lista con las características [pitches, velocidades, duraciones, tempo_bpm]
                                      de todas las partes procesadas.
    """
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
    """
    Función para procesar todas las partes de un archivo MIDI, extrayendo notas, acordes, velocidades y duraciones.
    
    Parámetros:
        midi_file (str): Ruta del archivo MIDI.
    
    Retorna:
        datos_partes (list): Lista de listas donde cada elemento contiene los datos de una parte del archivo MIDI
                             en el formato [nombres, pitches, velocidades, duraciones, tempo_bpm].
    """
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


def calcular_metrica_armonica(pitches):
    """
    Calcula la métrica armónica sumando las diferencias absolutas entre notas consecutivas.

    Parámetros:
        pitches (list): Lista de alturas (en valores MIDI) de las notas.

    Retorna:
        metrica (int): Valor entero que representa el movimiento tonal total.
    """
    metrica = 0
    for i in range(len(pitches) - 1):
        intervalo = abs(pitches[i + 1] - pitches[i])
        metrica += intervalo
    return metrica



c_a = "Audios/"

  
  ##Si haces Audios
nombre_pista1 = "piano right"
nombre_pista2 = "piano left"



mapa_right, mapa_left = cargar_notas_acordes_canciones(c_a,nombre_pista1, nombre_pista2)


import os

def calcular_metrica_armonica_normalizada(pitches):
    """
    Calcula la métrica armónica normalizada sumando las diferencias absolutas 
    entre notas consecutivas y dividiendo por la cantidad de notas.

    Parámetros:
        pitches (list): Lista de alturas (en valores MIDI) de las notas.

    Retorna:
        metrica_normalizada (float): Valor de la métrica normalizada.
    """
    if len(pitches) < 2:  # Si hay menos de dos notas, la métrica es 0
        return 0
    metrica = 0
    for i in range(len(pitches) - 1):
        intervalo = abs(pitches[i + 1] - pitches[i])
        metrica += intervalo
    return int(metrica / len(pitches))

# Carpeta que contiene los archivos MIDI
carpeta_midi = "A"

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

# Imprimir las métricas calculadas
for archivo, valores in metricas.items():
    print(f"{archivo}: Right={valores['metrica_right']}, Left={valores['metrica_left']}")



##


# Carpeta que contiene los archivos MIDI
carpeta_midi = "B"

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

# Imprimir las métricas calculadas
for archivo, valores in metricas.items():
    print(f"{archivo}: Right={valores['metrica_right']}, Left={valores['metrica_left']}")

