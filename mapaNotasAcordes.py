import os
from music21 import converter, chord, note
import sys

def cargar_notas_acordes_canciones(carpeta_audios, nombre_pieza_1, nombre_pieza_2):
    notasyacordes_1 = []
    notasyacordes_2 = []

    ##notasyacordes_1.append([-1])
    ##notasyacordes_2.append([-1])
    
    for i in range(0, 128):  # Corrige el rango para incluir 0 hasta 127
        notasyacordes_1.append([i])  # Agregar sublistas con el valor correspondiente
        notasyacordes_2.append([i])  # Agregar sublistas con el valor correspondiente
    
    for nombre_archivo in os.listdir(carpeta_audios):
        archivo_midi = os.path.join(carpeta_audios, nombre_archivo)
        
        try:
            # Cargar el archivo MIDI
            midi_data = converter.parse(archivo_midi)
        except Exception as e:
            print(f"No se pudo cargar el archivo MIDI: {e}")
            continue

        # Filtrar las partes de las dos piezas
        parte_1 = None
        parte_2 = None
        for part in midi_data.parts:
            if part.partName and nombre_pieza_1 in part.partName.lower():
                parte_1 = part
            elif part.partName and nombre_pieza_2 in part.partName.lower():
                parte_2 = part

        if not parte_1:
            print(f"No se encontró {nombre_pieza_1} en {archivo_midi}")
            sys.exit()
            
        else:
            # Recoger los acordes (pitches) para la primera pieza
            for elemento in parte_1.flatten().notesAndRests:
                if isinstance(elemento, chord.Chord):  # Si el elemento es un acorde
                    pitch = sorted([n.pitch.midi for n in elemento.notes])  # Alturas de las notas, ordenadas
                    if pitch not in notasyacordes_1:
                        notasyacordes_1.append(pitch)
                elif isinstance(elemento, note.Note):  # Si el elemento es una nota
                    pitch = [elemento.pitch.midi]  # Altura de la nota
                    if pitch not in notasyacordes_1:
                        notasyacordes_1.append(pitch)

        if not parte_2:
            print(f"No se encontró {nombre_pieza_2} en {archivo_midi}")
            sys.exit()
        else:
            # Recoger los acordes (pitches) para la segunda pieza
            for elemento in parte_2.flatten().notesAndRests:
                if isinstance(elemento, chord.Chord):  # Si el elemento es un acorde
                    pitch = sorted([n.pitch.midi for n in elemento.notes])  # Alturas de las notas, ordenadas
                    if pitch not in notasyacordes_2:
                        notasyacordes_2.append(pitch)
                elif isinstance(elemento, note.Note):  # Si el elemento es una nota
                    pitch = [elemento.pitch.midi]  # Altura de la nota
                    if pitch not in notasyacordes_2:
                        notasyacordes_2.append(pitch)

        print(f'Acordes y notas cargados de {nombre_pieza_1} y {nombre_pieza_2}: {archivo_midi}')
    
    return notasyacordes_1, notasyacordes_2




def añadir_acordes_mapa(notasyacordes_1,notasyacordes_2,archivo_midi,nombre_pieza_1,nombre_pieza_2):
    
   
    midi_data = converter.parse(archivo_midi)
  
    
    # Filtrar las partes de las dos piezas
    parte_1 = None
    parte_2 = None
    for part in midi_data.parts:
        if part.partName and nombre_pieza_1 in part.partName.lower():
            parte_1 = part
        elif part.partName and nombre_pieza_2 in part.partName.lower():
            parte_2 = part
    
    if not parte_1:
        print(f"No se encontró {nombre_pieza_1} en {archivo_midi}")
        sys.exit()
        
    else:
        # Recoger los acordes (pitches) para la primera pieza
        for elemento in parte_1.flatten().notesAndRests:
            if isinstance(elemento, chord.Chord):  # Si el elemento es un acorde
                pitch = sorted([n.pitch.midi for n in elemento.notes])  # Alturas de las notas, ordenadas
                if pitch not in notasyacordes_1:
                    notasyacordes_1.append(pitch)
            elif isinstance(elemento, note.Note):  # Si el elemento es una nota
                pitch = [elemento.pitch.midi]  # Altura de la nota
                if pitch not in notasyacordes_1:
                    notasyacordes_1.append(pitch)
    
    if not parte_2:
        print(f"No se encontró {nombre_pieza_2} en {archivo_midi}")
        sys.exit()
    else:
        # Recoger los acordes (pitches) para la segunda pieza
        for elemento in parte_2.flatten().notesAndRests:
            if isinstance(elemento, chord.Chord):  # Si el elemento es un acorde
                pitch = sorted([n.pitch.midi for n in elemento.notes])  # Alturas de las notas, ordenadas
                if pitch not in notasyacordes_2:
                    notasyacordes_2.append(pitch)
            elif isinstance(elemento, note.Note):  # Si el elemento es una nota
                pitch = [elemento.pitch.midi]  # Altura de la nota
                if pitch not in notasyacordes_2:
                    notasyacordes_2.append(pitch)
    
    print(f'Acordes y notas cargados de {nombre_pieza_1} y {nombre_pieza_2}: {archivo_midi}')
    
    
    
    return notasyacordes_1,notasyacordes_2