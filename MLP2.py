from pruebaMusic21 import procesar_primera_pista
from cargar_caracteristicas import cargarPista
from crear_secuencias import crear_secuencia
import os
import copy


# Parámetros principales
secuencia_len = 5  # Número de notas en cada secuencia de entrada

# Definir la carpeta donde estoy parado
carpeta_audios = r'C:\Users\Rama\Desktop\b\IC-TPI\Audios\beethoven1.mid'


#Agarramos de a una cancion
todos_caracteristicas = cargarPista(carpeta_audios)
maximo_tamaño_acorde=5

x,y = crear_secuencia(todos_caracteristicas,maximo_tamaño_acorde,secuencia_len)

    

x_aplanado = []

# Suponiendo que x tiene 3 listas y quieres recorrer sus elementos
for i in range(len(x[0])):  # Asumiendo que x[0], x[1], x[2] tienen la misma longitud
    x_auxiliar = []  # Crear una lista vacía para almacenar los elementos en cada iteración
    x_auxiliar.extend([x[0][i]])  # Agregar el elemento de x[0][i]
    x_auxiliar.extend([x[1][i]])  # Agregar el elemento de x[1][i]
    x_auxiliar.extend([x[2][i]])  # Agregar el elemento de x[2][i]
    
    x_aplanado.append(x_auxiliar)  # Agregar la lista auxiliar a la lista final


