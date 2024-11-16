from music21 import converter, tempo, chord, note, instrument, stream

def cargarPista (archivo_midi, nombre_pieza):
    # Listas grandes para almacenar todos los datos de los archivos MIDI
    todos_caracteristicas = []
    try:
           # Llamada a la función para extraer nombres, pitches, velocidades y duraciones de cada archivo
            nombres, pitches, velocidades, duraciones, tempo_bpm = procesar_primera_pista(archivo_midi, nombre_pieza)
            print(f'Archivo cargado: {archivo_midi}')
            # Extender la lista con los valores del archivo actual
            auxiliar = [pitches,velocidades,duraciones]
            todos_caracteristicas=auxiliar
        
    except FileNotFoundError:
            print(f'No se encontró el archivo: {archivo_midi}')
    return todos_caracteristicas


def procesar_primera_pista(midi_file, nombre_pieza):
    try:
        midi_data = converter.parse(midi_file)
    except Exception as e:
        raise ValueError(f"No se pudo cargar el archivo MIDI: {e}")

    # Obtener el tempo del archivo MIDI (si está definido, se toma el primero encontrado)
    tempos = midi_data.flat.getElementsByClass(tempo.MetronomeMark)
    tempo_bpm = tempos[0].number if len(tempos) > 0 else 120  # Valor predeterminado: 120 BPM

    # Filtrar la parte de "Piano Right"
    parte = None
    for part in midi_data.parts:
        # Verificar si el nombre de la parte contiene "Piano Right"
        if part.partName and nombre_pieza in part.partName.lower():
            parte = part
            break

    # Validar si se encontró la parte
    if parte:
        print(f"Parte {nombre_pieza} encontrada.")
    else:
        print(f"No se encontró una parte etiquetada como {nombre_pieza}.")

    # Listas para almacenar los datos de la pista
    nombres = []       # Nombres de las notas, acordes o silencios
    pitches = []       # Alturas (pitches) en formato MIDI (notas numéricas)
    velocidades = []   # Velocidades (volúmenes de las notas/acordes)
    duraciones = []    # Duraciones de los elementos en "quarterLength" (unidad relativa a negras)

    # Recorrer los elementos de la pista (notas, acordes, silencios)
    for elemento in parte.flat.notesAndRests:
        if isinstance(elemento, chord.Chord):  # Si el elemento es un acorde
            nombres.append(f"Acorde: {', '.join(n.nameWithOctave for n in elemento.notes)}")  # Nombres de las notas del acorde
            pitches.append([n.pitch.midi for n in elemento.notes])  # Alturas de las notas
            velocidad_acorde = elemento.notes[0].volume.velocity if elemento.notes else 0
            velocidades.append(velocidad_acorde)
            duraciones.append(elemento.quarterLength)  # Duración del acorde
        elif isinstance(elemento, note.Note):  # Si el elemento es una nota
            nombres.append(elemento.nameWithOctave)  # Nombre de la nota (con octava)
            pitches.append([elemento.pitch.midi])  # Altura MIDI de la nota
            velocidades.append(elemento.volume.velocity)  # Velocidad de la nota
            duraciones.append(elemento.quarterLength)  # Duración de la nota
        elif isinstance(elemento, note.Rest):  # Si el elemento es un silencio
            nombres.append("Silencio")  # Indica que es un silencio
            pitches.append([-1])  # Un silencio no tiene altura (pitch)
            velocidades.append(0)  # Un silencio no tiene velocidad
            duraciones.append(float(elemento.quarterLength))  # Duración del silencio

    # Retornar los datos procesados junto con el tempo
    return nombres, pitches, velocidades, duraciones, tempo_bpm


# def generar_cancion(pitches_conprediccion, velocities_conprediccion, durations_conprediccion):
#     # Crear una nueva secuencia de música
#     cancion = stream.Stream()

#     # Asegurarse de que las listas tengan el mismo tamaño
#     if len(pitches_conprediccion) == len(velocities_conprediccion) == len(durations_conprediccion):
#         for i in range(len(pitches_conprediccion)):
#             pitch = pitches_conprediccion[i]
#             velocity = velocities_conprediccion[i]
#             duration = durations_conprediccion[i]

#             # Si el pitch es -1, entonces es un silencio
#             if pitch == -1:
#                 # Crear un silencio con la duración especificada
#                 silencio = note.Rest(quarterLength=duration)
#                 cancion.append(silencio)
#             # Si el pitch tiene más de una nota (acorde)
#             elif isinstance(pitch, list):
#                 # Crear un acorde con las notas
#                 notas = [note.Note(p, quarterLength=duration) for p in pitch]
#                 for n in notas:
#                     n.volume.velocity = velocity
#                 acord = chord.Chord(notas)
#                 cancion.append(acord)
#             else:
#                 # Crear una nota individual
#                 n = note.Note(pitch, quarterLength=duration)
#                 n.volume.velocity = velocity
#                 cancion.append(n)

#     else:
#         print("Las listas de predicciones no tienen el mismo tamaño. No se puede generar la canción.")

#     return cancion


def generar_cancion(lista_de_canales):
    from music21 import stream, note, chord, instrument

    # Crear una nueva secuencia de música para la canción completa
    cancion = stream.Score()

    # Crear las partes de Piano Right y Piano Left
    piano_right = stream.Part()
    piano_right.insert(0, instrument.Piano())
    piano_right.partName = "Piano Right"

    piano_left = stream.Part()
    piano_left.insert(0, instrument.Piano())
    piano_left.partName = "Piano Left"

    # Iterar sobre los canales (se espera una lista de listas, una para cada canal)
    for canal_idx, canal in enumerate(lista_de_canales):
        # Asegurarse de que cada canal tenga listas válidas para pitches, velocities y durations
        if len(canal) == 3:
            pitches_conprediccion, velocities_conprediccion, durations_conprediccion = canal

            if len(pitches_conprediccion) == len(velocities_conprediccion) == len(durations_conprediccion):
                for pitch, velocity, duration in zip(pitches_conprediccion, velocities_conprediccion, durations_conprediccion):
                    # Si el pitch es -1, crear un silencio
                    if pitch == -1:
                        silencio = note.Rest(quarterLength=duration)
                        if canal_idx == 0:  # Primer canal: Piano Right
                            piano_right.append(silencio)
                        elif canal_idx == 1:  # Segundo canal: Piano Left
                            piano_left.append(silencio)
                    # Si el pitch tiene más de una nota (acorde)
                    elif isinstance(pitch, list):
                        notas = [note.Note(p, quarterLength=duration) for p in pitch]
                        for n in notas:
                            n.volume.velocity = velocity
                        acord = chord.Chord(notas)
                        if canal_idx == 0:  # Primer canal: Piano Right
                            piano_right.append(acord)
                        elif canal_idx == 1:  # Segundo canal: Piano Left
                            piano_left.append(acord)
                    else:
                        # Crear una nota individual
                        n = note.Note(pitch, quarterLength=duration)
                        n.volume.velocity = velocity
                        if canal_idx == 0:  # Primer canal: Piano Right
                            piano_right.append(n)
                        elif canal_idx == 1:  # Segundo canal: Piano Left
                            piano_left.append(n)
            else:
                print(f"Las listas de predicciones no coinciden en tamaño en el canal {canal_idx}.")
        else:
            print(f"El canal {canal_idx} no tiene un formato válido. Se necesitan tres listas: pitches, velocities y durations.")

    # Añadir las partes al score
    cancion.append(piano_right)
    cancion.append(piano_left)

    return cancion

