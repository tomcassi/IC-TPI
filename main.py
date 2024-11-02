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
import sys

def cargar_cancion(file_path):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    return midi_data


# def escribir_cancion(notas):
#     # midi_generado = pretty_midi.PrettyMIDI()
#     # midi_generado.instruments.append(pretty_midi.Instrument(0))
    
#     # for pitch in pitches:
#     #     note = pretty_midi.Note(velocity=100, pitch=pitch, start=0, end=1)
#     #     midi_generado.instruments[0].notes.append(note)
    
#     return midi_generado

def escribir_midi(midi_data, file_path):
    
    midi_data.write(file_path)
    return


if __name__ == "__main__":
    longitud_secuencia = 10
    cant_predicciones = 10

    print("Main")
    # midi_data = cargar_cancion("Happy Birthday MIDI.mid")
    midi_data = cargar_cancion("youre only lonely L.mid")
    
    
    #chequear longitud de la secuencia
    if longitud_secuencia >= len(midi_data.instruments[0].notes)-1:
        print("Error: La longitud de la secuencia es mayor o igual a la cantidad de notas disponibles.")
        sys.exit(1)  # Termina el programa con un código de error opcional (1)

    X,y = MLP.crear_secuencias(midi_data, longitud_secuencia)
    
    mlp = MLP.entrenar_modelo(X,y)
    
    pitches_pred = np.array(MLP.predecir_sig_pitches(midi_data.instruments[0].notes[0:longitud_secuencia],mlp,cant_predicciones))
   
    print(pitches_pred)
   
   
   
   
   
    # y_pred = MLP.entrenamiento(X,y)
    
    # X_y_concatenado =[]
    # for patron in range(len(X)):
    #     for pitch in X[patron-1]:
    #         X_y_concatenado.append(pitch)
    #     X_y_concatenado.append(y_pred[patron])
        
        
    # for i in range(len(Xfinal)):
    #     X_y_concatenado.append(Xfinal[i])
    
    # X_y_concatenado = np.array(X_y_concatenado)
    # print(X_y_concatenado)
    
    # lista_notas = midi_data.instruments[0].notes[:]
    # for nota in range(len(lista_notas)-1):
    #     lista_notas[nota].pitch = X_y_concatenado[nota]
    
    # midi_data.instruments[0].notes = lista_notas

    # midi_data.instruments = [midi_data.instruments[0]]
    
    # print("")
    # midi_data.write("Bateria modificada.mid")
    
    
    # #Este es para comparar
    # midi2_data = cargar_cancion("youre only lonely L.mid")
    # midi2_data.instruments = [midi2_data.instruments[0]]  
    # midi2_data.write("Bateria original.mid")