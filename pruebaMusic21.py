from music21 import converter, chord, note, stream

# Función para descomponer el archivo MIDI en nombres de acordes/notas, números de pitch MIDI, velocidad y duración
def descomponer_midi_en_acordes_y_notas(midi_file):
    # Cargar el archivo MIDI
    midi_data = converter.parse(midi_file)
    
    # Obtener todas las pistas en el archivo MIDI
    pistas = midi_data.getElementsByClass('Stream')
    
    # Listas para almacenar nombres, pitch, velocidad y duración
    nombres_acordes_y_notas = []
    numeros_pitch = []
    velocidades = []
    duraciones = []
    
    # Recorrer todas las pistas en el archivo
    for pista in pistas:
        # Listas temporales para los datos de cada pista
        nombres_pista = []
        pitches_pista = []
        velocidades_pista = []
        duraciones_pista = []
        
        # Recorrer todos los elementos de la pista en orden cronológico
        for elemento in pista.flat:
            # Si el elemento es un acorde
            if isinstance(elemento, chord.Chord):
                # Obtener los nombres, números de pitch y velocidades de las notas en el acorde
                acorde_nombres = [n.nameWithOctave for n in elemento.notes]
                acorde_pitches = [n.pitch.midi for n in elemento.notes if n.pitch is not None]
                acorde_velocidades = [n.volume.velocity for n in elemento.notes if n.volume.velocity is not None]
                
                # Asegurarnos de que el pitch tenga siempre 5 elementos, completando con ceros si es necesario
                while len(acorde_pitches) < 5:
                    acorde_pitches.append(0)
                acorde_pitches = acorde_pitches[:5]
                
                # Asegurarnos de que las velocidades también tengan 5 elementos
                while len(acorde_velocidades) < 5:
                    acorde_velocidades.append(0)
                acorde_velocidades = acorde_velocidades[:5]
                
                # Guardar el nombre, pitch, velocidad y duración del acorde
                nombres_pista.append(f"Acorde: {', '.join(acorde_nombres)}")
                pitches_pista.append(acorde_pitches)
                velocidades_pista.append(acorde_velocidades)
                duraciones_pista.append(float(elemento.quarterLength))  # Convertir duración a float
                
            # Si el elemento es una nota individual
            elif isinstance(elemento, note.Note):
                # Guardar el nombre, pitch y velocidad de la nota
                nombres_pista.append(elemento.nameWithOctave)
                pitch_list = [elemento.pitch.midi] if elemento.pitch is not None else [0]
                velocity = [elemento.volume.velocity if elemento.volume.velocity is not None else 0]
                
                # Asegurarnos de que pitch y velocidad tengan siempre 5 elementos
                while len(pitch_list) < 5:
                    pitch_list.append(0)
                while len(velocity) < 5:
                    velocity.append(0)
                    
                pitches_pista.append(pitch_list)
                velocidades_pista.append(velocity)
                duraciones_pista.append(float(elemento.quarterLength))  # Convertir duración a float
        
        # Añadir los datos de esta pista a las listas generales
        nombres_acordes_y_notas.append(nombres_pista)
        numeros_pitch.append(pitches_pista)
        velocidades.append(velocidades_pista)
        duraciones.append(duraciones_pista)
    
    return nombres_acordes_y_notas, numeros_pitch, velocidades, duraciones


# Modificar la función para incluir un desplazamiento temporal
def guardar_como_midi(nombres, pitches, velocidades, duraciones, archivo_salida):
    nuevo_midi = stream.Score()
    
    for i in range(len(nombres)):
        nueva_pista = stream.Part()
        
        offset = 0.0  # Lleva la cuenta del tiempo en la pista
        
        for nombre, pitch, velocidad, duracion in zip(nombres[i], pitches[i], velocidades[i], duraciones[i]):
            if isinstance(nombre, str) and 'Acorde' in nombre:
                acorde = chord.Chord(pitch)
                acorde.quarterLength = duracion
                acorde.offset = offset  # Asigna el offset al acorde
                nueva_pista.append(acorde)
            else:
                for p, v in zip(pitch, velocidad):
                    if p != 0:
                        n = note.Note(p)
                        n.quarterLength = duracion
                        n.offset = offset  # Asigna el offset a la nota
                        nueva_pista.append(n)
            
            # Aumenta el offset según la duración de la última nota/acorde añadida
            offset += duracion
        
        nuevo_midi.append(nueva_pista)
    
    nuevo_midi.write('midi', fp=archivo_salida)
    print(f"Archivo MIDI guardado en: {archivo_salida}")


# Usar la función con tu archivo MIDI
midi_file = r'C:\Users\Rama\Desktop\b\IC-TPI\Audios\beethoven1.mid'
archivo_salida = r'C:\Users\Rama\Desktop\b\IC-TPI\Audios\beethoven1_output.mid'

# Obtener los nombres de acordes/notas, números de pitch, velocidades y duraciones
nombres_acordes_y_notas, numeros_pitch, velocidades, duraciones = descomponer_midi_en_acordes_y_notas(midi_file)

# Guardar los datos en un nuevo archivo MIDI
guardar_como_midi(nombres_acordes_y_notas, numeros_pitch, velocidades, duraciones, archivo_salida)

