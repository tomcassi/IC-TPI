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

Para saber que instrumento es se utiliza el midi_data.instruments[instrumento].program()

0-7: Pianos (ej., 0 = Piano acústico, 4 = Clavicordio)
8-15: Instrumentos de percusión cromática (ej., 9 = Glockenspiel, 11 = Vibrafono)
16-23: Órganos (ej., 16 = Órgano de percusión, 19 = Órgano de iglesia)
24-31: Guitarras (ej., 24 = Guitarra acústica de nylon, 27 = Guitarra eléctrica limpia)
32-39: Bajos (ej., 32 = Bajo acústico, 36 = Bajo eléctrico de púa)
40-47: Cuerdas (ej., 40 = Violín, 44 = Contrabajo)
48-55: Instrumentos de cuerda (ej., 48 = Ensamble de cuerdas, 50 = Pizzicato)
56-63: Trompetas y otros metales (ej., 56 = Trompeta, 60 = Trompeta de madera)
64-71: Instrumentos de viento de madera (ej., 64 = Soprano Sax, 66 = Fagot)
72-79: Flautas y otros instrumentos de viento (ej., 73 = Flauta, 74 = Flautín)
80-87: Sintetizadores y sonidos varios
88-127: Efectos de sonido y percusión (ej., 117 = Tambor de madera, 120-127 = Efectos de sonido).

el instrumento del canal 10 esta reservado para percusion, ignora el program
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

# def escribir_cancion(instrumentos?, notas?, etc):
#     # midi_generado = pretty_midi.PrettyMIDI()
#     # midi_generado.instruments.append(pretty_midi.Instrument(0))
    
#     # for pitch in pitches:
#     #     note = pretty_midi.Note(velocity=100, pitch=pitch, start=0, end=1)
#     #     midi_generado.instruments[0].notes.append(note)
    
#     return midi_generado


if __name__ == "__main__":
    longitud_secuencia = 10
    cant_predicciones = 10

    print("Main")
    midi_data = cargar_cancion("Audios/beethoven1.mid")
    
    
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