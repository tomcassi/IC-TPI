from pruebaMusic21 import descomponer_midi_en_acordes_y_notas
import os

# Parámetros principales
secuencia_len = 10  # Número de notas en cada secuencia de entrada

# Definir la carpeta donde estoy parado
carpeta_principal = r'C:\Users\Rama\Desktop\b\IC-TPI'  # Ruta a la carpeta principal
carpeta_audios = carpeta_principal + '\Audios'

# Listas grandes para almacenar todos los datos de los archivos MIDI
todos_caracteristicas = []

# Cargar múltiples archivos MIDI y extraer sus características
for i in range(10):  # De 1 a 3 para cargar los audios
    archivo_midi = os.path.join(carpeta_audios, f"beethoven{i}.mid")  # Ruta completa del archivo MIDI
    
    try:
        # Llamada a la función para extraer nombres, pitches, velocidades y duraciones de cada archivo
        nombres_acordes_y_notas, numeros_pitch, velocidades, duraciones = descomponer_midi_en_acordes_y_notas(archivo_midi)
        print(f'Archivo cargado: {archivo_midi}')
        # Extender la lista con los valores del archivo actual
        auxiliar = [numeros_pitch,velocidades,duraciones]
        todos_caracteristicas.append(auxiliar)
    
    except FileNotFoundError:
        print(f'No se encontró el archivo: {archivo_midi}')
