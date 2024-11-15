from music21 import converter, note, chord, stream, tempo

# Función para procesar la primera pista de un archivo MIDI
def procesar_primera_pista(midi_file):
    """
    Procesa la primera pista de un archivo MIDI, extrayendo notas, acordes, silencios y tempo.
    
    Args:
        midi_file (str): Ruta al archivo MIDI que se desea procesar.
    
    Returns:
        tuple: Contiene las listas de nombres, pitches, velocidades, duraciones y el tempo en BPM.
    """
    # Cargar el archivo MIDI
    midi_data = converter.parse(midi_file)
    
    # Obtener el tempo del archivo MIDI (si está definido, se toma el primero encontrado)
    tempos = midi_data.flat.getElementsByClass(tempo.MetronomeMark)
    tempo_bpm = tempos[0].number if len(tempos) > 0 else 120  # Valor predeterminado: 120 BPM
    
    # Seleccionar la primera pista del archivo MIDI
    # Si tiene partes (tracks), selecciona la primera; si no, toma todo el archivo
    primera_pista = midi_data.parts[0] if len(midi_data.parts) > 0 else midi_data
    
    # Listas para almacenar los datos de la pista
    nombres = []       # Nombres de las notas, acordes o silencios
    pitches = []       # Alturas (pitches) en formato MIDI (notas numéricas)
    velocidades = []   # Velocidades (volúmenes de las notas/acordes)
    duraciones = []    # Duraciones de los elementos en "quarterLength" (unidad relativa a negras)
    
    # Recorrer los elementos de la pista (notas, acordes, silencios)
    for elemento in primera_pista.flat.notesAndRests:
        if isinstance(elemento, chord.Chord):  # Si el elemento es un acorde
            nombres.append(f"Acorde: {', '.join(n.nameWithOctave for n in elemento.notes)}")  # Nombres de las notas del acorde
            pitches.append([n.pitch.midi for n in elemento.notes])  # Alturas de las notas
            velocidades.append([n.volume.velocity or 0 for n in elemento.notes])  # Velocidades de cada nota
            duraciones.append(elemento.quarterLength)  # Duración del acorde
        elif isinstance(elemento, note.Note):  # Si el elemento es una nota
            nombres.append(elemento.nameWithOctave)  # Nombre de la nota (con octava)
            pitches.append([elemento.pitch.midi])  # Altura MIDI de la nota
            velocidades.append([elemento.volume.velocity or 0])  # Velocidad de la nota
            duraciones.append(elemento.quarterLength)  # Duración de la nota
        elif isinstance(elemento, note.Rest):  # Si el elemento es un silencio
            nombres.append("Silencio")  # Indica que es un silencio
            pitches.append([0])  # Un silencio no tiene altura (pitch)
            velocidades.append([0])  # Un silencio no tiene velocidad
            duraciones.append(float(elemento.quarterLength))  # Duración del silencio
    
    # Retornar los datos procesados junto con el tempo
    return nombres, pitches, velocidades, duraciones, tempo_bpm

# Función para guardar un nuevo archivo MIDI a partir de los datos procesados
def guardar_midi_con_tempo(nombres, pitches, velocidades, duraciones, tempo_bpm, archivo_salida):
    """
    Crea y guarda un nuevo archivo MIDI utilizando los datos procesados.
    
    Args:
        nombres (list): Lista con los nombres de notas/acordes/silencios.
        pitches (list): Lista de alturas (en formato MIDI) para cada elemento.
        velocidades (list): Lista de velocidades para cada elemento.
        duraciones (list): Lista de duraciones (en quarterLength) para cada elemento.
        tempo_bpm (float): Tempo en BPM a agregar al inicio del archivo.
        archivo_salida (str): Ruta donde se guardará el nuevo archivo MIDI.
    """
    nuevo_stream = stream.Stream()  # Crear un nuevo stream para construir el MIDI
    
    # Agregar el tempo al inicio del stream
    marca_tempo = tempo.MetronomeMark(number=tempo_bpm)  # Crear la marca de tempo
    nuevo_stream.append(marca_tempo)  # Agregar al stream
    
    offset = 0.0  # Posición inicial en tiempo
    
    # Recorrer los datos procesados para reconstruir el MIDI
    for nombre, pitch, velocidad, duracion in zip(nombres, pitches, velocidades, duraciones):
        if "Acorde" in nombre:  # Si es un acorde
            acorde = chord.Chord(pitch)  # Crear un objeto Chord con las alturas MIDI
            acorde.quarterLength = duracion  # Asignar la duración
            acorde.offset = offset  # Establecer el tiempo inicial del acorde
            nuevo_stream.append(acorde)  # Agregar el acorde al stream
        elif "Silencio" in nombre:  # Si es un silencio
            silencio = note.Rest()  # Crear un objeto Rest
            silencio.quarterLength = duracion  # Asignar la duración
            silencio.offset = offset  # Establecer el tiempo inicial del silencio
            nuevo_stream.append(silencio)  # Agregar el silencio al stream
        else:  # Si es una nota
            for p, v in zip(pitch, velocidad):  # Recorrer cada nota (para soportar acordes simples)
                if p != 0:  # Evitar procesar silencios
                    n = note.Note(p)  # Crear un objeto Note con la altura MIDI
                    n.quarterLength = duracion  # Asignar la duración
                    n.offset = offset  # Establecer el tiempo inicial de la nota
                    nuevo_stream.append(n)  # Agregar la nota al stream
        
        offset += duracion  # Avanzar el tiempo para el siguiente elemento
    
    # Guardar el stream en un archivo MIDI
    nuevo_stream.write('midi', fp=archivo_salida)
    print(f"Archivo MIDI guardado en: {archivo_salida}")

# Rutas de archivo
midi_file = r'C:\Users\Rama\Desktop\b\IC-TPI\Audios\beethoven1.mid'  # Archivo MIDI de entrada
archivo_salida = r'C:\Users\Rama\Desktop\b\IC-TPI\Audios\beethoven1_output.mid'  # Archivo MIDI de salida

# Procesar el archivo MIDI
nombres, pitches, velocidades, duraciones, tempo_bpm = procesar_primera_pista(midi_file)

# Guardar los datos procesados en un nuevo archivo MIDI con el tempo
guardar_midi_con_tempo(nombres, pitches, velocidades, duraciones, tempo_bpm, archivo_salida)
