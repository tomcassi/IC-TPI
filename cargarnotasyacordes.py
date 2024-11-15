from music21 import converter, tempo, chord, note, instrument
import os

notasyacordes = []


for i in range(-1, 128):  # Corrige el rango para incluir -1 hasta 127
    notasyacordes.append([i])  # Agregar sublistas con el valor correspondiente



def cargar_acordes(midi_file):
    # Cargar el archivo MIDI
    try:
        midi_data = converter.parse(midi_file)
    except Exception as e:
        raise ValueError(f"No se pudo cargar el archivo MIDI: {e}")

    # Filtrar la parte de "Piano Right"
    piano_right = None
    for part in midi_data.parts:
        # Verificar si el nombre de la parte contiene "Piano Right"
        if part.partName and "Piano right" in part.partName:
            piano_right = part
            break
        # Alternativamente, verificar si el instrumento es Piano
        elif any(isinstance(instr, instrument.Piano) for instr in part.getElementsByClass(instrument.Instrument)):
            piano_right = part
            break

    # Usar la primera pista si no se encuentra "Piano Right"
    if not piano_right:
        print("No se encontró una parte etiquetada como 'Piano Right'. Usando la primera pista disponible.")
        piano_right = midi_data.parts[0]

    # Recoger los acordes (pitches)
    pitches = []
    for elemento in piano_right.flat.notesAndRests:
        if isinstance(elemento, chord.Chord):  # Si el elemento es un acorde
            pitches.append(sorted([n.pitch.midi for n in elemento.notes]))  # Alturas de las notas, ordenadas
    return pitches

if __name__ == "__main__":
    carpeta_audios = "Audios/"
    for nombre_archivo in os.listdir(carpeta_audios):
        archivo_midi = os.path.join(carpeta_audios, nombre_archivo)
        
        try:
            pitches = cargar_acordes(archivo_midi)
            print(f'Archivo cargado: {archivo_midi}')
        except ValueError as e:
            print(e)
            continue
        
        for pitch in pitches:
            if pitch not in notasyacordes:
                notasyacordes.append(pitch)

    # Mostrar los acordes únicos
    print("Acordes únicos (pitches):")
    for acorde in notasyacordes:
        print(acorde)