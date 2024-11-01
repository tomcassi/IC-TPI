#Interfaz de la aplicacion
import customtkinter

class file_frame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        self.grid_columnconfigure(0,weight=1)
        
        self.entry = customtkinter.CTkEntry(self)
        self.entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.button_seleccionar_archivo = customtkinter.CTkButton(self, text="Seleccionar archivo .mid")
        self.button_seleccionar_archivo.grid(row=0,column=2, padx=10, pady=10)
        

class model_frame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        self.variable = customtkinter.IntVar(value=0)
        
        self.radiobutton_modelo1 = customtkinter.CTkRadioButton(self, text="modelo 1", variable=self.variable, value=1)
        self.radiobutton_modelo1.grid(row=0, column=0, padx=10, pady=10)
        self.radiobutton_modelo2 = customtkinter.CTkRadioButton(self, text="modelo 2", variable=self.variable, value=2)
        self.radiobutton_modelo2.grid(row=1, column=0, padx=10, pady=10)

        
        
class progress_frame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        self.variable = customtkinter.Variable(value=0)
        
        self.grid_columnconfigure(0,weight=1)
        
        self.progressbar = customtkinter.CTkProgressBar(self, variable=self.variable)
        self.progressbar.grid(row=0, column=0, padx=10, pady=10, sticky="ew")


class button_frame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        self.rowconfigure(0,weight=1)
        
        self.button = customtkinter.CTkButton(self, text="Procesar", fg_color="green")
        self.button.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("my app")
        self.geometry("400x150")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  
        
        self.file_frame = file_frame(self)
        self.file_frame.grid(row=0, column=0, sticky="new", columnspan=2)
        
        self.model_frame = model_frame(self)
        self.model_frame.grid(row=1, column=0, sticky="nswe")
        
        self.button_frame = button_frame(self)
        self.button_frame.grid(row=1, column=1, sticky="ns")
        
        self.progress_frame = progress_frame(self)
        self.progress_frame.grid(row=2, column=0, sticky="sew", columnspan=2)
        
        
        # self.file_frame.configure(fg_color="transparent")
        # self.model_frame.configure(fg_color="transparent")
        # self.button_frame.configure(fg_color="transparent")
        # self.progress_frame.configure(fg_color="transparent")


# Crear y ejecutar la aplicación
app = App()
app.mainloop()
