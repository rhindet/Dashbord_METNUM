import tkinter as tk
from tkinter import ttk

# Crear una instancia de la ventana principal
ventana = tk.Tk()
ventana.title("Metodos Numericos - Dashboard")

# Establecer el tamaño fijo de la ventana (ancho x alto)
ventana.geometry("800x600")

# Crear un objeto Notebook (pestañas)
notebook = ttk.Notebook(ventana)

# Pestaña 1
pestaña1 = ttk.Frame(notebook)
notebook.add(pestaña1, text="Clima")

etiqueta1 = tk.Label(pestaña1, text="Contenido de la Pestaña 1")
etiqueta1.pack(fill="both", expand=True)

# Pestaña 2
pestaña2 = ttk.Frame(notebook)
notebook.add(pestaña2, text="Economia")

etiqueta2 = tk.Label(pestaña2, text="Contenido de la Pestaña 2")
etiqueta2.pack(fill="both", expand=True)

# Pestaña 3
pestaña3 = ttk.Frame(notebook)
notebook.add(pestaña3, text="Medicina")

etiqueta3 = tk.Label(pestaña3, text="Contenido de la Pestaña 3")
etiqueta3.pack(fill="both", expand=True)

# Empacar el objeto Notebook con expansión en ambas direcciones
notebook.pack(fill="both", expand=True)

# Ejecutar el bucle principal de la aplicación
ventana.mainloop()
