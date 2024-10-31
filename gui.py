#Interfaz de la aplicacion
import customtkinter

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("my app")
        self.geometry("1280x720")

# Crear y ejecutar la aplicación
app = App()
app.mainloop()
