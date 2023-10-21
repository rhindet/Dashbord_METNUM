import tkinter as tk
from tkinter import ttk

# Función para calcular la temperatura estimada
def calcular_temperatura():
    hora_interes = float(entry_hora.get())
    # Datos de temperatura horaria (horas y temperaturas)
    datos = [(0, 10), (1, 12), (2, 15), (3, 18), (4, 20)]


    # Calcular las diferencias finitas hacia adelante
    n = len(datos)
    tabla_diferencias = [[0] * n for _ in range(n)]
    for i in range(n):
        tabla_diferencias[i][0] = datos[i][1]
    for j in range(1, n):
        for i in range(n - j):
            tabla_diferencias[i][j] = tabla_diferencias[i + 1][j - 1] - tabla_diferencias[i][j - 1]

    # Calcular el término de Newton
    resultado = tabla_diferencias[0][0]
    producto = 1
    for i in range(1, n):
        producto *= (hora_interes - datos[i - 1][0]) / i
        resultado += producto * tabla_diferencias[0][i]

    print(f"La temperatura estimada en la hora {hora_interes} es {resultado} °C")

    resultado = tabla_diferencias[0][0]
    producto = 1
    for i in range(1, n):
        producto *= (hora_interes - datos[i - 1][0]) / i
        resultado += producto * tabla_diferencias[0][i]
    resultado_label.config(text=f"La temperatura estimada en la hora {hora_interes} es {resultado:.2f} °C")

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

# Marco con estilo de botón más pequeño para contener los elementos
marco_boton = ttk.LabelFrame(pestaña1, text="Clima", padding=10, width=300, height=150)
marco_boton.pack(padx=20, pady=20, fill="both", expand=True)

# Cuadro de texto para ingresar la hora
etiqueta_hora = tk.Label(marco_boton, text="Ingrese la hora para estimar la temperatura:")
etiqueta_hora.pack()
entry_hora = tk.Entry(marco_boton)
entry_hora.pack()
calcular_button = tk.Button(marco_boton, text="Calcular Temperatura", command=calcular_temperatura)
calcular_button.pack()

# Etiqueta para mostrar el resultado
resultado_label = tk.Label(marco_boton, text="")
resultado_label.pack()

# Resto del código (interpolación)...

# Pestaña 2
pestaña2 = ttk.Frame(notebook)
notebook.add(pestaña2, text="Economia")

etiqueta2 = tk.Label(pestaña2, text="Contenido de la Pestaña 2")
etiqueta2.pack()

# Pestaña 3
pestaña3 = ttk.Frame(notebook)
notebook.add(pestaña3, text="Medicina")

etiqueta3 = tk.Label(pestaña3, text="Contenido de la Pestaña 3")
etiqueta3.pack()

# Personalizar el tamaño de fuente para el título de la pestaña "Clima"
estilo = ttk.Style()
estilo.configure("TNotebook.Tab", font=("Arial", 20))

# Empacar el objeto Notebook con expansión en ambas direcciones
notebook.pack(fill="both", expand=True)

# Ejecutar el bucle principal de la aplicación
ventana.mainloop()
