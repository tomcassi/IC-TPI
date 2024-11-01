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




if __name__ == "__main__":
    print("Main")
    # midi_data = cargar_cancion("Happy Birthday MIDI.mid")
    midi_data = cargar_cancion("youre only lonely L.mid")
    X,y,Xfinal = MLP.crear_secuencias(midi_data)
    
    y_pred = MLP.entrenamiento(X,y)
    
    X_y_concatenado =[]
    for patron in range(len(X)):
        for pitch in X[patron-1]:
            X_y_concatenado.append(pitch)
        X_y_concatenado.append(y_pred[patron])
    X_y_concatenado.append(Xfinal)
    
    X_y_concatenado = np.array(X_y_concatenado)
    print(X_y_concatenado)
    
    lista_notas = midi_data.instruments[0].notes[:]
    for nota in range(len(lista_notas)):
        lista_notas[nota].pitch = X_y_concatenado[nota]
    
    midi_data.instruments[0].notes = lista_notas

    midi_data.instruments = [midi_data.instruments[0]]
    
    print("")
    midi_data.write("midi modificado.mid")
    
