from music21 import converter, tempo, chord, note, instrument

def procesar_primera_pista(midi_file):
    # Cargar el archivo MIDI
    try:
        midi_data = converter.parse(midi_file)
    except Exception as e:
        raise ValueError(f"No se pudo cargar el archivo MIDI: {e}")

    # Obtener el tempo del archivo MIDI (si está definido, se toma el primero encontrado)
    tempos = midi_data.flat.getElementsByClass(tempo.MetronomeMark)
    tempo_bpm = tempos[0].number if len(tempos) > 0 else 120  # Valor predeterminado: 120 BPM

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

    # Validar si se encontró la parte de "Piano Right"
    if piano_right:
        print("Parte 'Piano Right' encontrada.")
    else:
        print("No se encontró una parte etiquetada como 'Piano Right'. Usando la primera pista disponible.")
        piano_right = midi_data.parts[0]  # Usar la primera pista si no se encuentra "Piano Right"

    # Listas para almacenar los datos de la pista
    nombres = []       # Nombres de los acordes
    pitches = []       # Alturas (pitches) en formato MIDI (notas numéricas)
    velocidades = []   # Velocidades (volúmenes de las notas/acordes)
    duraciones = []    # Duraciones de los elementos en "quarterLength" (unidad relativa a negras)

    # Recorrer los elementos de la pista (solo acordes)
    for elemento in piano_right.flat.notesAndRests:
        if isinstance(elemento, chord.Chord):  # Si el elemento es un acorde
            nombres.append(f"Acorde: {', '.join(n.nameWithOctave for n in elemento.notes)}")  # Nombres de las notas del acorde
            pitches.append([n.pitch.midi for n in elemento.notes])  # Alturas de las notas
            velocidades.append([n.volume.velocity or 0 for n in elemento.notes])  # Velocidades de cada nota
            duraciones.append(elemento.quarterLength)  # Duración del acorde

    # Retornar los datos procesados junto con el tempo
    return nombres, pitches, velocidades, duraciones, tempo_bpm

if __name__ == "__main__":
    archivo_midi = "Audios\elise.mid"
    todos_caracteristicas = []
    try:
           # Llamada a la función para extraer nombres, pitches, velocidades y duraciones de cada archivo
            nombres, pitches, velocidades, duraciones, tempo_bpm = procesar_primera_pista(archivo_midi)
            print(f'Archivo cargado: {archivo_midi}')
            # Extender la lista con los valores del archivo actual
            auxiliar = [pitches,velocidades,duraciones]
            todos_caracteristicas=auxiliar
        
    except FileNotFoundError:
            print(f'No se encontró el archivo: {archivo_midi}')
    
    