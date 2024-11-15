from pruebaMusic21 import procesar_primera_pista
from cargar_caracteristicas import cargarPista
import os
import copy


# Parámetros principales
secuencia_len = 5  # Número de notas en cada secuencia de entrada

# Definir la carpeta donde estoy parado
carpeta_audios = r'C:\Users\Rama\Desktop\b\IC-TPI\Audios\beethoven1.mid'


#Agarramos de a una cancion
todos_caracteristicas = cargarPista(carpeta_audios)

maximo_tamaño_acorde=5
#Le hacemos el MLP

x,y=[],[]
pitch=[]
velocidad=[]
for i in range(len(todos_caracteristicas[0]) - secuencia_len):
    # Crear una copia profunda para evitar modificar los datos originales
    pitch_auxiliar = copy.deepcopy(todos_caracteristicas[0][i:i+secuencia_len])
    velocidad_auxiliar = copy.deepcopy(todos_caracteristicas[1][i:i+secuencia_len])
    
    for j in range(len(pitch_auxiliar)):
        while len(pitch_auxiliar[j]) < maximo_tamaño_acorde:
            pitch_auxiliar[j].append(0)  # Agregar ceros al final de la sublista
            
        while len(velocidad_auxiliar[j]) < maximo_tamaño_acorde:
            velocidad_auxiliar[j].append(0)  # Agregar ceros al final de la sublista
            
    pitch.append(pitch_auxiliar)
    velocidad.append(velocidad_auxiliar)
    
    duracion = todos_caracteristicas[2][i:i+secuencia_len]
    
    y_auxiliar = todos_caracteristicas[0][i + secuencia_len]
    x.append(pitch_auxiliar)
    y.append(y_auxiliar)

    

   

    
    
    
    