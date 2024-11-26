from music21 import converter, tempo, chord, note, instrument, stream,meter

import sys

def cargarPista (archivo_midi, nombre_pieza):
    # Listas grandes para almacenar todos los datos de los archivos MIDI
    todos_caracteristicas = []
    try:
           # Llamada a la función para extraer nombres, pitches, velocidades y duraciones de cada archivo
            nombres, pitches, velocidades, duraciones, tempo_bpm = procesar_primera_pista(archivo_midi, nombre_pieza)
            print(f'Archivo cargado: {archivo_midi}')
            # Extender la lista con los valores del archivo actual
            auxiliar = [pitches,velocidades,duraciones,tempo_bpm]
            todos_caracteristicas=auxiliar
        
    except FileNotFoundError:
            print(f'No se encontró el archivo: {archivo_midi}')
    return todos_caracteristicas


def procesar_primera_pista(midi_file, nombre_pieza):
    try:
        midi_data = converter.parse(midi_file)
    except Exception as e:
        raise ValueError(f"No se pudo cargar el archivo MIDI: {e}")

       # Obtener todos los tempos del archivo MIDI y almacenarlos en una lista
    tempos = midi_data.flatten().getElementsByClass(tempo.MetronomeMark)
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
        print(f"No se encontró una parte etiquetada como {nombre_pieza} en {midi_file}.")
        sys.exit(0)

    # Listas para almacenar los datos de la pista
    nombres = []       # Nombres de las notas, acordes o silencios
    pitches = []       # Alturas (pitches) en formato MIDI (notas numéricas)
    velocidades = []   # Velocidades (volúmenes de las notas/acordes)
    duraciones = []    # Duraciones de los elementos en "quarterLength" (unidad relativa a negras)

    # Recorrer los elementos de la pista (notas, acordes, silencios)
    for elemento in parte.flatten().notesAndRests:
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

    #Retornar los datos procesados junto con el tempo
    return nombres, pitches, velocidades, duraciones, tempo_bpm


def generar_cancion(lista_de_canales, tempo_bpm,time_signature,nombre_archivo):
    # Crear una nueva secuencia de música para la canción completa
    cancion = stream.Score()

    # Crear las partes de Piano Right y Piano Left
    piano_right = stream.Part()
    piano_right.append(instrument.Piano())  # Agrega el instrumento al final
    
    piano_left = stream.Part()
    piano_left.append(instrument.Piano())  # También lo agrega al final

    # Establecer el tempo de la canción
    tempo_indicacion = tempo.MetronomeMark(number=tempo_bpm)
    cancion.insert(0, tempo_indicacion)
    
    
    # Establecer la firma de compás para cada parte
    cancion.insert(0, meter.TimeSignature(time_signature))  # Nueva instancia de TimeSignature
  
    

    # Iterar sobre los canales (se espera una lista de listas, una para cada canal)
    for canal_idx, canal in enumerate(lista_de_canales):
        # Asegurarse de que cada canal tenga listas válidas para pitches, velocities y durations
        if len(canal) == 3:
            pitches_conprediccion, velocities_conprediccion, durations_conprediccion = canal

            if len(pitches_conprediccion) == len(velocities_conprediccion) == len(durations_conprediccion):
                for pitch, velocity, duration in zip(pitches_conprediccion, velocities_conprediccion, durations_conprediccion):
                    # Si el pitch es -1, crear un silencio
                    if pitch == [-1]:
                        silencio = note.Rest(quarterLength=duration)
                        if canal_idx == 0:  # Primer canal: Piano Right
                            piano_right.append(silencio)
                        elif canal_idx == 1:  # Segundo canal: Piano Left
                            piano_left.append(silencio)
                    # Si el pitch tiene más de una nota (acorde)
                   # elif isinstance(pitch, list):
                    elif len(pitch)>1: 
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
                        n = note.Note(pitch[0], quarterLength=duration)
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

    cancion.write('midi', fp=nombre_archivo)  # Guardar como archivo MIDI


    return cancion


def getTempo(midi_file):
    # Cargar el archivo MIDI
    midi_data = converter.parse(midi_file)

    # Obtener todos los objetos de tipo MetronomeMark (tempo)
    tempos = midi_data.flatten().getElementsByClass(tempo.MetronomeMark)

    # Si hay tempos definidos, devolver el primero; de lo contrario, usar un valor por defecto de 120 BPM
    tempo_bpm = tempos[0].number if len(tempos) > 0 else 120

    return tempo_bpm


def getTimeSignature(midi_file):

    try:
        # Cargar el archivo MIDI
        midi_data = converter.parse(midi_file)

        # Buscar objetos de tipo TimeSignature
        time_signatures = midi_data.flatten().getElementsByClass(meter.TimeSignature)

        # Si hay firmas de compás, devolver la primera; de lo contrario, usar un valor por defecto
        if time_signatures:
            time_signature = time_signatures[0]
            return f"{time_signature.numerator}/{time_signature.denominator}"
        else:
            return "4/4"  # Valor por defecto si no se encuentra ninguna firma
    except Exception as e:
        print(f"Error al procesar la firma de compás del archivo MIDI: {e}")
        return "4/4"  # Valor por defecto en caso de error




from music21 import converter, note, stream

from music21 import converter, meter

def calcular_longitud_secuencia(path_cancion, duracion_en_segundos,nombre_pieza1,nombre_pieza2):

    # Cargar la partitura desde el archivo
    partitura = converter.parse(path_cancion)

    # Determinar el BPM (si no hay, asumir 120 BPM por defecto)
    bpm = partitura.metronomeMarkBoundaries()[0][2].number if partitura.metronomeMarkBoundaries() else 120
    segundos_por_negra = 60 / bpm

    # Calcular el tiempo en negras
    tiempo_total_en_negras = duracion_en_segundos / segundos_por_negra

    # Inicializar contadores para Piano Right y Piano Left
    cuenta_piano_right = 0
    cuenta_piano_left = 0

    # Iterar sobre las partes y filtrar los elementos dentro del tiempo
    for part in partitura.parts:
            
        if part.partName and nombre_pieza1 in part.partName.lower():
            elementos = part.flat.notesAndRests
            elementos_dentro_del_tiempo = [
                elem for elem in elementos if elem.offset < tiempo_total_en_negras
            ]
            cuenta_piano_right = len(elementos_dentro_del_tiempo)
        elif part.partName and nombre_pieza2 in part.partName.lower():
            elementos = part.flat.notesAndRests
            elementos_dentro_del_tiempo = [
                elem for elem in elementos if elem.offset < tiempo_total_en_negras
            ]
            cuenta_piano_left = len(elementos_dentro_del_tiempo)

    # Devolver un vector con los resultados
    return [cuenta_piano_right, cuenta_piano_left]




def crear_secuencias(caracteristicas, longitud_secuencia):
    X, y = [], []
    for nota in range(len(caracteristicas) - longitud_secuencia):
        # Extrae una secuencia de notas de longitud especificada
        
        listanotasX = caracteristicas[nota:nota + longitud_secuencia]
        X.append(listanotasX)
        
        y.append(caracteristicas[nota + longitud_secuencia])
    return X, y



