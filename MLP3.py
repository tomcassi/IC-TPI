import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from music21 import converter, tempo, chord, note, instrument


def cargar_notas_acordes_canciones(carpeta_audios="Audios/"):
    print("\n=====Cargando notas=====")
    notasyacordes = []
    for i in range(-1, 128):  # Corrige el rango para incluir -1 hasta 127
        notasyacordes.append([i])  # Agregar sublistas con el valor correspondiente
    
    print("\n=====Cargando acordes presentes en canciones=====")
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
    return notasyacordes


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



def procesar_primera_pista(midi_file):
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
    nombres = []       # Nombres de las notas, acordes o silencios
    pitches = []       # Alturas (pitches) en formato MIDI (notas numéricas)
    velocidades = []   # Velocidades (volúmenes de las notas/acordes)
    duraciones = []    # Duraciones de los elementos en "quarterLength" (unidad relativa a negras)

    # Recorrer los elementos de la pista (notas, acordes, silencios)
    for elemento in piano_right.flat.notesAndRests:
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

def cargarPista (archivo_midi):
    # Listas grandes para almacenar todos los datos de los archivos MIDI
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
            
            
    return todos_caracteristicas


def crear_secuencias(caracteristicas, longitud_secuencia):
    X, y = [], []
    for nota in range(len(caracteristicas) - longitud_secuencia):
        # Extrae una secuencia de notas de longitud especificada
        
        listanotasX = caracteristicas[nota:nota + longitud_secuencia]
        X.append(listanotasX)
        
        y.append(caracteristicas[nota + longitud_secuencia])
    return X, y


def entrenar_modelo(X, y, mlp):
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)


    # Entrenar el modelo
    mlp.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = mlp.predict(X_test)

    # Evaluar la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión del modelo:", accuracy)
        
    return mlp, y_pred, y_test

def predecir_sig_elem(elem_originales, modelo, cant_predicciones):
    n_predicciones = 0
    elementos = elem_originales
    
    
    while n_predicciones < cant_predicciones:
          
        elem_input = np.array(elementos[n_predicciones:n_predicciones+len(elem_originales)]).reshape(1, -1)
        print("Input para predicción:", elem_input)
        
        prediccion = modelo.predict(elem_input)
        prediccion = prediccion[0]
        print("Predicción:", prediccion)
        
        elementos.append(prediccion)
        n_predicciones += 1
    
    return elementos

from music21 import stream, note, chord

def generar_cancion(pitches_conprediccion, velocities_conprediccion, durations_conprediccion):
    # Crear una nueva secuencia de música
    cancion = stream.Stream()

    # Asegurarse de que las listas tengan el mismo tamaño
    if len(pitches_conprediccion) == len(velocities_conprediccion) == len(durations_conprediccion):
        for i in range(len(pitches_conprediccion)):
            pitch = pitches_conprediccion[i]
            velocity = velocities_conprediccion[i]
            duration = durations_conprediccion[i]

            # Si el pitch es -1, entonces es un silencio
            if pitch == -1:
                # Crear un silencio con la duración especificada
                silencio = note.Rest(quarterLength=duration)
                cancion.append(silencio)
            # Si el pitch tiene más de una nota (acorde)
            elif isinstance(pitch, list):
                # Crear un acorde con las notas
                notas = [note.Note(p, quarterLength=duration) for p in pitch]
                for n in notas:
                    n.volume.velocity = velocity
                acord = chord.Chord(notas)
                cancion.append(acord)
            else:
                # Crear una nota individual
                n = note.Note(pitch, quarterLength=duration)
                n.volume.velocity = velocity
                cancion.append(n)

    else:
        print("Las listas de predicciones no tienen el mismo tamaño. No se puede generar la canción.")

    return cancion




if __name__ == "__main__":
    longitud_secuencia = 20
    carpeta_audios = "Audios/"
    notasyacordes = cargar_notas_acordes_canciones(carpeta_audios)

    print("\n=====Cargando caracteristicas=====")
                
    mlp_pitch = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=10000)
    mlp_velocity = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=10000)
    mlp_duration = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=10000)
    

    for nombre_archivo in os.listdir(carpeta_audios):
        archivo_midi = os.path.join(carpeta_audios, nombre_archivo)
        
        todos_caracteristicas = cargarPista(archivo_midi)
        for i, nota_acorde in enumerate(todos_caracteristicas[0]):
            indice = notasyacordes.index(sorted(nota_acorde))
            todos_caracteristicas[0][i] = indice
        
        X,y = crear_secuencias(todos_caracteristicas[0],longitud_secuencia)
        
        mlp_pitch, y_pred, y_test = entrenar_modelo(X,y,mlp_pitch)
        
        # y_test = np.array(y_test)
        
        X,y = crear_secuencias(todos_caracteristicas[1],longitud_secuencia)
        
        mlp_velocity, y_pred, y_test = entrenar_modelo(X,y,mlp_velocity)
        
        X,y = crear_secuencias(todos_caracteristicas[2],longitud_secuencia)
        # Multiplicar cada valor dentro de X por 100 y convertir a int
        X = [[int(valor * 1000) for valor in sublista] for sublista in X]
        
        # Multiplicar cada valor en y por 100 y convertir a int
        y = [int(valor * 1000) for valor in y]
        
        mlp_duration, y_pred, y_test = entrenar_modelo(X,y,mlp_duration)
        
        y_test = np.array(y_test)
    
    
    
    #Predecir cancion:
    todos_caracteristicas = cargarPista("Audios/mond_3.mid")
    for i, nota_acorde in enumerate(todos_caracteristicas[0]):
        indice = notasyacordes.index(sorted(nota_acorde))
        todos_caracteristicas[0][i] = indice
    
    cant_predicciones = 100
    pitches_conprediccion = predecir_sig_elem(todos_caracteristicas[0][0:longitud_secuencia], mlp_pitch, cant_predicciones)
    velocities_conprediccion = predecir_sig_elem(todos_caracteristicas[1][0:longitud_secuencia], mlp_velocity, cant_predicciones)
    
    durations_originales = todos_caracteristicas[2][0:longitud_secuencia]
    
    for i in range(len(durations_originales)):
        durations_originales[i] *= 1000
    
    durations_conprediccion = predecir_sig_elem(durations_originales, mlp_duration, cant_predicciones)
    
    for i in range(len(durations_conprediccion)):
        durations_conprediccion[i] /= 1000
    
    
    for i in range(len(pitches_conprediccion)):
        pitches_conprediccion[i] = notasyacordes[pitches_conprediccion[i]]
            
    cancion_generada = generar_cancion(pitches_conprediccion, velocities_conprediccion, durations_conprediccion)

    # Guardar la canción en un archivo MIDI
    cancion_generada.write('midi', fp='cancion_generada.mid')