""" 
============ Pitch ============
Rango 0-127
Representa la nota que se toca, 0=C1, 127= G9.
Cada incremento es un semitono
En bateria o elementos de percusion no representa la nota, sino el sonido de percusion

============ Velocity ============
0-127
Representa la fuerza con la que se toca la nota

====== Valores a extraer de estructura de midi_data ======

midi_data.instruments[instrumento].notes[nota].duration
midi_data.instruments[instrumento].notes[nota].start
midi_data.instruments[instrumento].notes[nota].end
midi_data.instruments[instrumento].notes[nota].velocity
midi_data.instruments[instrumento].notes[nota].pitch
=========================================================
"""

import pretty_midi
import numpy as np
import MLP
import RNN

def cargar_cancion(file_path):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    return midi_data

def reemplazar_nota(midi_data, cant_originales=5, cant_reemplazos=1):
    # Implementar
    return midi_data

def crear_secuencias(midi_data, longitud_secuencia = 5):
    X, y = [],[]
    
    for nota in range(0,len(midi_data.instruments[0].notes),longitud_secuencia):
        listanotasX = []
        listapitchX = []
        
        if nota+longitud_secuencia <= len(midi_data.instruments[0].notes):
            listanotasX = midi_data.instruments[0].notes[nota:nota+longitud_secuencia]
            # print("X",nota,nota+longitud_secuencia)
            # print("y",nota+longitud_secuencia)
            for n in listanotasX:
                listapitchX.append(n.pitch)
            y.append(midi_data.instruments[0].notes[nota+longitud_secuencia].pitch) 
            X.append(listapitchX)
    return X,y


if __name__ == "__main__":
    print("Main")
    # midi_data = cargar_cancion("Happy Birthday MIDI.mid")
    midi_data = cargar_cancion("youre only lonely L.mid")

    
    X,y = MLP.crear_secuencias(midi_data, longitud_secuencia = 5)
    
    y_pred = MLP.entrenamiento(X,y)
    
    X_y_concatenado =[]
    for patron in range(len(X)):
        for pitch in X[patron-1]:
            X_y_concatenado.append(pitch)
        X_y_concatenado.append(y_pred[patron])
        
        
    for i in range(len(Xfinal)):
        X_y_concatenado.append(Xfinal[i])
    
    X_y_concatenado = np.array(X_y_concatenado)
    print(X_y_concatenado)
    
    lista_notas = midi_data.instruments[0].notes[:]
    for nota in range(len(lista_notas)-1):
        lista_notas[nota].pitch = X_y_concatenado[nota]
    
    midi_data.instruments[0].notes = lista_notas

    midi_data.instruments = [midi_data.instruments[0]]
    
    print("")
    midi_data.write("Bateria modificada.mid")
    
    
    #Este es para comparar
    midi2_data = cargar_cancion("youre only lonely L.mid")
    midi2_data.instruments = [midi2_data.instruments[0]]  
    midi2_data.write("Bateria original.mid")