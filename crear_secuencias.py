import copy

def crear_secuencia(todos_caracteristicas,maximo_tamaño_acorde,secuencia_len):

    x,y=[[]],[[]]
    pitch=[]
    # velocidad=[]
    for i in range(len(todos_caracteristicas[0]) - secuencia_len):
        # Crear una copia profunda para evitar modificar los datos originales
        pitch_auxiliar = copy.deepcopy(todos_caracteristicas[0][i:i+secuencia_len])
        # velocidad_auxiliar = copy.deepcopy(todos_caracteristicas[1][i:i+secuencia_len])
        
        for j in range(len(pitch_auxiliar)):
            while len(pitch_auxiliar[j]) < maximo_tamaño_acorde:
                pitch_auxiliar[j].append(0)  # Agregar ceros al final de la sublista
                
        pitch.append(pitch_auxiliar)
        # velocidad.append(velocidad_auxiliar)
        
        # duracion = todos_caracteristicas[2][i:i+secuencia_len]
        
        y_auxiliar_pitch = todos_caracteristicas[0][i + secuencia_len]
        # y_auxiliar_velocidad = todos_caracteristicas[1][i + secuencia_len]
        # y_auxiliar_duracion = todos_caracteristicas[2][i + secuencia_len]
        
      
        while len(y_auxiliar_pitch) < maximo_tamaño_acorde:
            y_auxiliar_pitch.append(0)  # Agregar ceros al final de la sublista
            
            
            
        # while len(y_auxiliar_velocidad) < maximo_tamaño_acorde:
        #     y_auxiliar_velocidad.append(0)  # Agregar ceros al final de la sublista
        
        x[0].append(pitch_auxiliar)
        y[0].append(y_auxiliar_pitch)
        
        # x[1].append(velocidad_auxiliar)
        # y[1].append(y_auxiliar_velocidad)
    
    
        # x[2].append(duracion)
        # y[2].append(y_auxiliar_duracion)
        
        
    return x,y



def aplanar_secuencia(x,y):
    x_aplanado=[]
    y_aplanado=[]

    # Suponiendo que x tiene 3 listas y quieres recorrer sus elementos
    for i in range(len(x[0])):  # Asumiendo que x[0], x[1], x[2] tienen la misma longitud
        x_auxiliar = []  # Crear una lista vacía para almacenar los elementos en cada iteración
        
        
        x_auxiliar.extend(x[0][i])  # Agregar todos los elementos de x[0][i] a x_auxiliar
        # x_auxiliar.extend(x[1][i])  # Agregar todos los elementos de x[1][i] a x_auxiliar
        # x_auxiliar.extend(x[2][i])  # Agregar todos los elementos de x[2][i] a x_auxiliar
        
        def aplanar(lista):
            resultado = []
            for elemento in lista:
                if isinstance(elemento, list):
                    resultado.extend(aplanar(elemento))
                else:
                    resultado.append(elemento)
        
            return resultado
        
        resultado=aplanar(x_auxiliar)
        
        x_aplanado.append(resultado)  # Agregar los elementos de x_auxiliar a la lista final (esto aplana la lista)
        
       ###### 
        
        y_auxiliar = []  # Crear una lista vacía para almacenar los elementos en cada iteración
        
        
        y_auxiliar.extend(y[0][i])  # Agregar todos los elementos de x[0][i] a x_auxiliar
        # y_auxiliar.extend(y[1][i])  # Agregar todos los elementos de x[1][i] a x_auxiliar
        # y_auxiliar.append(y[2][i])  # Agregar todos los elementos de x[2][i] a x_auxiliar
        
        resultado=aplanar(y_auxiliar)
        
        y_aplanado.append(resultado)  # Agregar los elementos de x_auxiliar a la lista final (esto aplana la lista)
    
    return x_aplanado,y_aplanado


