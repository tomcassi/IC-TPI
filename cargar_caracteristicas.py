from pruebaMusic21 import procesar_primera_pista
import os


#sarchivo_midi = os.path.join(carpeta_audios, f"beethoven{indice}.mid")  # Ruta completa del archivo MIDI
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

