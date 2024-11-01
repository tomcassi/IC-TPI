# Implementacion con perceptron multicapa
import numpy as np

def crear_secuencias(midi_data, longitud_secuencia = 2):
    X, y = [],[]
    
    for nota in range(0,len(midi_data.instruments[0].notes),longitud_secuencia):
        print(nota)
        
        if nota+longitud_secuencia < len(midi_data.instruments[0].notes):
            listanotasX = midi_data.instruments[0].notes[nota:nota+longitud_secuencia]
            for i in range(len(listanotasX)):
                X.append(listanotasX[i].pitch)

            y.append(midi_data.instruments[0].notes[nota+longitud_secuencia].pitch)            


    return np.array(X),np.array(y)


if __name__ == "__main__":
    print("MLP")