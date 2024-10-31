import pretty_midi


def cargar_cancion(file_path):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    return midi_data


""" 
============ Pitch ============
Rango 0-127
Representa la nota que se toca, 0=C1, 127= G9.
Cada incremento es un semitono
En bateria o elementos de percusion no representa la nota, sino el sonido de percusion

============ Velocity ============
0-127
Representa la fuerza con la que se toca la nota
"""

 


midi_data = cargar_cancion("Happy Birthday MIDI.mid")

# print(midi_data.instruments[0].notes)


# midi_data.write("midi modificado.mid")