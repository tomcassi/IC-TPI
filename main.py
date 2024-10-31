import pretty_midi


def cargar_cancion(file_path):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    return midi_data
    


midi_data = cargar_cancion("Happy Birthday MIDI.mid")

# print(midi_data.instruments[0].notes)


# midi_data.write("midi modificado.mid")