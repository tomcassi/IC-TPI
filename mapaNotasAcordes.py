import os
from music21 import converter, chord

def cargar_notas_acordes_canciones(carpeta_audios, nombre_pieza):
    notasyacordes = []
    # for i in range(-1, 128):  # Corrige el rango para incluir -1 hasta 127
    #     notasyacordes.append([i])  # Agregar sublistas con el valor correspondiente
    for i in range(0, 128):  # Corrige el rango para incluir -1 hasta 127
        notasyacordes.append([i])  # Agregar sublistas con el valor correspondiente
    
    
    print("\n=====Cargando acordes presentes en canciones=====")
    for nombre_archivo in os.listdir(carpeta_audios):
        print(nombre_archivo)
        archivo_midi = os.path.join(carpeta_audios, nombre_archivo)
        
        try:
            pitches = cargar_acordes(archivo_midi, nombre_pieza)
            print(f'Acordes cargados de {nombre_pieza}: {archivo_midi}')
        except ValueError as e:
            print(e)
            continue
        
        for pitch in pitches:
            if pitch not in notasyacordes:
                notasyacordes.append(pitch)
    return notasyacordes


def cargar_acordes(midi_file, nombre_pieza):
    # Cargar el archivo MIDI
    try:
        midi_data = converter.parse(midi_file)
    except Exception as e:
        raise ValueError(f"No se pudo cargar el archivo MIDI: {e}")

    # Filtrar la parte de "Piano Right"
    piano_right = None
    for part in midi_data.parts:
        # Verificar si el nombre de la parte contiene "Piano Right"
        if part.partName and nombre_pieza in part.partName.lower():
            piano_right = part
            break

    if not piano_right:
        print("No se encontro "+ nombre_pieza)

    # Recoger los acordes (pitches)
    pitches = []
    for elemento in piano_right.flat.notesAndRests:
        if isinstance(elemento, chord.Chord):  # Si el elemento es un acorde
            pitches.append(sorted([n.pitch.midi for n in elemento.notes]))  # Alturas de las notas, ordenadas
    return pitches