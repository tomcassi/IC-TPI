#Interfaz de la aplicacion
import customtkinter

class file_frame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        self.grid_columnconfigure(0,weight=1)
        self.grid_rowconfigure(0,weight=0) #esto hace falta?
        
        self.entry = customtkinter.CTkEntry(self)
        self.entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.button_seleccionar_archivo = customtkinter.CTkButton(self, text="Seleccionar archivo .mid")
        self.button_seleccionar_archivo.grid(row=0,column=2, padx=10, pady=10)
        

class model_frame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        



class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("my app")
        self.geometry("800x200")
        
        self.grid_columnconfigure(0,weight=1)
        self.grid_rowconfigure(0,weight=0)  
        self.grid_rowconfigure(1,weight=1)  
        
        self.file_frame = file_frame(self)
        self.file_frame.grid(row=0, column=0, sticky="new")
        
        
      
        self.model_frame = model_frame(self)
        self.model_frame.grid(row=1, column=0, sticky="nsw")


# Crear y ejecutar la aplicación
app = App()
app.mainloop()
