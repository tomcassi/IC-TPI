from cargarnotasyacordes import *
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np



errores = []  # Lista para almacenar los errores en cada época

def procesar_primera_pista(midi_file):
    """
    Procesa la primera pista de un archivo MIDI, extrayendo notas, acordes, silencios y tempo.
    
    Args:
        midi_file (str): Ruta al archivo MIDI que se desea procesar.
    
    Returns:
        tuple: Contiene las listas de nombres, pitches, velocidades, duraciones y el tempo en BPM.
    """
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
    nombres = []       # Nombres de las notas, acordes o silencios
    pitches = []       # Alturas (pitches) en formato MIDI (notas numéricas)
    velocidades = []   # Velocidades (volúmenes de las notas/acordes)
    duraciones = []    # Duraciones de los elementos en "quarterLength" (unidad relativa a negras)

    # Recorrer los elementos de la pista (notas, acordes, silencios)
    for elemento in piano_right.flat.notesAndRests:
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
            pitches.append([-1])  # Un silencio no tiene altura (pitch)
            velocidades.append([0])  # Un silencio no tiene velocidad
            duraciones.append(float(elemento.quarterLength))  # Duración del silencio

    # Retornar los datos procesados junto con el tempo
    return nombres, pitches, velocidades, duraciones, tempo_bpm

def cargarPista (archivo_midi):
    # Definir la carpeta donde estoy parado
    
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



def entrenar_modelo(X, y, modelo1):
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True, random_state=42)

    # Entrenar el modelo
    modelo1.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = modelo1.predict(X_test)

    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])-1):
            y_pred=y_pred.astype(int)
            if y_pred[i][j]<10:
                y_pred[i][j]=0


    # Evaluar el modelo usando el Error Cuadrático Medio (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print("Error cuadrático medio (MSE):", mse)
    
    # Almacenar el error en la lista global
    errores.append(mse)

    

    return modelo1, y_pred, y_test ,X_test



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

    # Imprimir algunas predicciones
    # for i in range(len(y_test)):
    #     print(f"Predicción: {y_pred[i]}, Real: {y_test[i]}")
        
    return mlp, y_pred, y_test


if __name__ == "__main__":
    longitud_secuencia = 5
    carpeta_audios = "Audios/"
    print("\n=====Cargando acordes=====")
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
    print("\n=====Cargando caracteristicas=====")
                
    mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=10000)
    

    for nombre_archivo in os.listdir(carpeta_audios):
        archivo_midi = os.path.join(carpeta_audios, nombre_archivo)
        
        todos_caracteristicas = cargarPista(archivo_midi)
        for i, nota_acorde in enumerate(todos_caracteristicas[0]):
            indice = notasyacordes.index(sorted(nota_acorde))
            todos_caracteristicas[0][i] = indice
        
        X,y = crear_secuencias(todos_caracteristicas[0],longitud_secuencia)
        
        mlp, y_pred, y_test = entrenar_modelo(X,y,mlp)
        y_test = np.array(y_test)
        


