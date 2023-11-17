import tkinter as tk
from tkinter import ttk
import numpy as np
import math

def crearVentana():
    ventana = tk.Tk()
    ventana.title("Metodos Numericos - Dashboard")
    ventana.geometry("800x600")

    notebook = ttk.Notebook(ventana)

    pestaña1 = ttk.Frame(notebook)
    notebook.add(pestaña1, text="Clima")

    pestaña2 = ttk.Frame(notebook)
    notebook.add(pestaña2, text="Economia")


    pestaña4 = ttk.Frame(notebook)
    notebook.add(pestaña4, text="Geologia")

    pestaña5 = ttk.Frame(notebook)
    notebook.add(pestaña5, text="Mediciones")

    pestaña6 = ttk.Frame(notebook)
    notebook.add(pestaña6, text="Viviendas")


    estilo = ttk.Style()
    estilo.configure("TNotebook.Tab", font=("Arial", 20))

    marco_boton = ttk.LabelFrame(pestaña1, text="Temperatura/horas", padding=10, width=100, height=150)
    marco_boton.pack(padx=10, pady=20, fill="both", expand=True)

    marco_boton2 = ttk.LabelFrame(pestaña2, text="Tasa de interés efectiva a la tasa de interés compuesta de un préstamo o inversión ", padding=10, width=100, height=150)
    marco_boton2.pack(padx=10, pady=20, fill="both", expand=True)



    etiqueta_hora = tk.Label(marco_boton, text="Ingrese la hora para estimar la temperatura:")
    etiqueta_hora.grid(row=0, column=0, columnspan=2)  # Usar grid en lugar de pack para disposición en cuadrícula

    # Crear etiquetas para las columnas
    etiqueta_columna_hora = tk.Label(marco_boton, text="Hora")
    etiqueta_columna_hora.grid(row=1, column=0, padx=5, pady=5)

    etiqueta_columna_tmp = tk.Label(marco_boton, text="Temperatura")
    etiqueta_columna_tmp.grid(row=1, column=1, padx=5, pady=5)

    # Crear cuadros de texto sin bucle for
    hora1 = tk.Entry(marco_boton)
    hora1.grid(row=2, column=0, padx=5, pady=5)

    hora2 = tk.Entry(marco_boton)
    hora2.grid(row=3, column=0, padx=5, pady=5)

    hora3 = tk.Entry(marco_boton)
    hora3.grid(row=4, column=0, padx=5, pady=5)

    hora4 = tk.Entry(marco_boton)
    hora4.grid(row=5, column=0, padx=5, pady=5)

    hora5 = tk.Entry(marco_boton)
    hora5.grid(row=6, column=0, padx=5, pady=5)

    tmp1 = tk.Entry(marco_boton)
    tmp1.grid(row=2, column=1, padx=5, pady=5)

    tmp2 = tk.Entry(marco_boton)
    tmp2.grid(row=3, column=1, padx=5, pady=5)

    tmp3 = tk.Entry(marco_boton)
    tmp3.grid(row=4, column=1, padx=5, pady=5)

    tmp4 = tk.Entry(marco_boton)
    tmp4.grid(row=5, column=1, padx=5, pady=5)

    tmp5 = tk.Entry(marco_boton)
    tmp5.grid(row=6, column=1, padx=5, pady=5)


    horaBuscada = tk.Entry(marco_boton)
    horaBuscada.grid(row=7, column=0, columnspan=2, pady=10)

    resultado_label = tk.Label(marco_boton, text="")
    resultado_label.grid(row=8, column=0, columnspan=2)

    # Crear las opciones para el combobox
    opciones2 = ["Interpolación lineal", "Newton Hacia Adelante", "Newton Hacia Atrás", "Newton Diferencias Divididas", "Lagrange"]

    # Crear un StringVar para almacenar la opción seleccionada
    selected_option2 = tk.StringVar()

    # Crear el combobox y asociar las opciones y el StringVar
    combo2 = ttk.Combobox(marco_boton, textvariable=selected_option2, values=opciones2, state="normal")
    combo2.grid(row=4, column=5, padx=5, pady=5)

    # Configurar una función de devolución de llamada para manejar la selección
    combo2.bind("<<ComboboxSelected>>", lambda event: on_select(event, combo2))

    # Inicializar la opción seleccionada (puedes establecer un valor predeterminado si lo deseas)
    selected_option2.set(opciones2[0])


    calcular_button = tk.Button(marco_boton, text="Calcular Temperatura", command=lambda: calcular_temperatura(hora1.get(),hora2.get(),hora3.get(),hora4.get(),hora5.get(),tmp1.get(),tmp2.get(),tmp3.get(),tmp4.get(),tmp5.get(), horaBuscada.get(),resultado_label))
    calcular_button.grid(row=9, column=0, columnspan=2)





    #------ GEOLOGIA ----------

    marco_boton4 = ttk.LabelFrame(pestaña4,
                                  text="Distribución de la densidad del suelo",
                                  padding=10, width=100, height=150)
    marco_boton4.pack(padx=10, pady=20, fill="both", expand=True)

    etiqueta_hora4 = tk.Label(marco_boton4, text="variaciones de las densidades")
    etiqueta_hora4.grid(row=0, column=0, columnspan=2)  # Usar grid en lugar de pack para disposición en cuadrícula



    # Crear cuadros de texto sin bucle for
    campo11 = tk.Entry(marco_boton4)
    campo11.grid(row=2, column=0, padx=5, pady=5)

    campo12 = tk.Entry(marco_boton4)
    campo12.grid(row=3, column=0, padx=5, pady=5)

    campo13 = tk.Entry(marco_boton4)
    campo13.grid(row=4, column=0, padx=5, pady=5)


    campo21 = tk.Entry(marco_boton4)
    campo21.grid(row=2, column=1, padx=5, pady=5)

    campo22 = tk.Entry(marco_boton4)
    campo22.grid(row=3, column=1, padx=5, pady=5)

    campo23 = tk.Entry(marco_boton4)
    campo23.grid(row=4, column=1, padx=5, pady=5)


    campo31 = tk.Entry(marco_boton4)
    campo31.grid(row=2, column=4, padx=5, pady=5)

    campo32 = tk.Entry(marco_boton4)
    campo32.grid(row=3, column=4, padx=5, pady=5)

    campo33 = tk.Entry(marco_boton4)
    campo33.grid(row=4, column=4, padx=5, pady=5)

    separator1 = ttk.Separator(marco_boton4, orient="vertical")
    separator1.grid(row=2, column=3, rowspan=3, sticky="ns", padx=10)

    campo41 = tk.Entry(marco_boton4)
    campo41.grid(row=2, column=2, padx=5, pady=5)

    campo42 = tk.Entry(marco_boton4)
    campo42.grid(row=3, column=2, padx=5, pady=5)

    campo43 = tk.Entry(marco_boton4)
    campo43.grid(row=4, column=2, padx=5, pady=5)

    # Crear las opciones para el combobox
    opciones = ["Eliminación Gaussiana", "Gauss-Jordan", "Montante", "Gauss-Seidel","Jacobi"]

    # Crear un StringVar para almacenar la opción seleccionada
    selected_option = tk.StringVar()



    # Crear el combobox y asociar las opciones y el StringVar
    combo = ttk.Combobox(marco_boton4, textvariable=selected_option, values=opciones, state="normal")
    combo.grid(row=5, column=2, padx=5, pady=5)

    # Configurar una función de devolución de llamada para manejar la selección
    combo.bind("<<ComboboxSelected>>", lambda event: on_select(event, combo))

    # Inicializar la opción seleccionada (puedes establecer un valor predeterminado si lo deseas)
    selected_option.set(opciones[0])




    resultado_label4 = tk.Label(marco_boton4, text="")
    resultado_label4.grid(row=8, column=0, columnspan=2)

    calcular_button4 = tk.Button(marco_boton4, text="Calcular Densidad",
                                command=lambda: elimGauss(campo11.get(), campo12.get(), campo13.get(), campo21.get(),
                                                                     campo22.get(), campo23.get(), campo31.get(), campo32.get(),campo33.get(),
                                                                     campo41.get(), campo42.get(), campo43.get(),
                                                                     resultado_label))
    calcular_button4.grid(row=9, column=1, columnspan=2)

    # ------ Mediciones ----------
    marco_boton5 = ttk.LabelFrame(pestaña5,
                                  text="Ingrese los datos",
                                  padding=10, width=100, height=150)
    marco_boton5.pack(padx=10, pady=20, fill="both", expand=True)

    etiquetaRadio = tk.Label(marco_boton5, text="Radio del tanque")
    etiquetaRadio.grid(row=0, column=5, columnspan=2)  # Usar grid en lugar de pack para disposición en cuadrícula

    # Crear cuadros de texto sin bucle for
    radioCampo = tk.Entry(marco_boton5)
    radioCampo.grid(row=2, column=5, padx=5, pady=5)

    etiquetaLargo = tk.Label(marco_boton5, text="Largo del tanque")
    etiquetaLargo.grid(row=4, column=5, columnspan=2)  # Usar grid en lugar de pack para disposición en cuadrícula

    # Crear cuadros de texto sin bucle for
    largoCampo = tk.Entry(marco_boton5)
    largoCampo.grid(row=5, column=5, padx=5, pady=5)

    etiquetaIntervalos = tk.Label(marco_boton5, text="Numero de intervalos")
    etiquetaIntervalos.grid(row=6, column=5, columnspan=2)  # Usar grid en lugar de pack para disposición en cuadrícula

    # Crear cuadros de texto sin bucle for
    intervalosCampo = tk.Entry(marco_boton5)
    intervalosCampo.grid(row=7, column=5, padx=5, pady=5)

    resultado_label5 = tk.Label(marco_boton5, text="")
    resultado_label5.grid(row=8, column=5, columnspan=4)

    # Crear las opciones para el combobox
    opciones3 = ["Interpolación lineal", "Newton Hacia Adelante", "Newton Hacia Atrás", "Newton Diferencias Divididas",
                 "Lagrange"]

    # Crear un StringVar para almacenar la opción seleccionada
    selected_option3 = tk.StringVar()

    # Crear el combobox y asociar las opciones y el StringVar
    combo3 = ttk.Combobox(marco_boton, textvariable=selected_option3, values=opciones3, state="normal")
    combo3.grid(row=7, column=0, padx=5, pady=5)

    # Configurar una función de devolución de llamada para manejar la selección
    combo3.bind("<<ComboboxSelected>>", lambda event: on_select(event, combo3))

    # Inicializar la opción seleccionada (puedes establecer un valor predeterminado si lo deseas)
    selected_option3.set(opciones3[0])


    calcular_button5 = tk.Button(marco_boton5, text="Calcular Volumen",
                                 command=lambda: reglaTresOctavosSimpson(radioCampo.get(),largoCampo.get(),intervalosCampo.get(),resultado_label5))
    calcular_button5.grid(row=14, column=5, columnspan=4)




    notebook.pack(fill="both", expand=True)
    ventana.mainloop()


def on_select(event,combo):
    # Use the get method on the StringVar to retrieve the selected option
    selected_option = combo.get()
    print(f"Seleccionado: {selected_option}")

def calcular_temperatura(hora1,hora2,hora3,hora4,hora5,tmp1,tmp2,tmp3,tmp4,tmp5,horaBuscada,resultado_label):
    hora_interes1 = float(hora1)
    hora_interes2 = float(hora2)
    hora_interes3 = float(hora3)
    hora_interes4 = float(hora4)
    hora_interes5 = float(hora5)
    tmp_interes1 = float(tmp1)
    tmp_interes2 = float(tmp2)
    tmp_interes3 = float(tmp3)
    tmp_interes4 = float(tmp4)
    tmp_interes5 = float(tmp5)

    horaIngresada = float(horaBuscada)


    # Datos de temperatura horaria (horas y temperaturas)
    datos = [(hora_interes1, tmp_interes1), (hora_interes2, tmp_interes2), (hora_interes3, tmp_interes3), (hora_interes4, tmp_interes4), (hora_interes5, tmp_interes5)]

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
        producto *= (horaIngresada - datos[i - 1][0]) / i
        resultado += producto * tabla_diferencias[0][i]

    print(f"La temperatura estimada en la hora {horaIngresada} es {resultado} °C")

    resultado = tabla_diferencias[0][0]
    producto = 1
    for i in range(1, n):
        producto *= (horaIngresada - datos[i - 1][0]) / i
        resultado += producto * tabla_diferencias[0][i]
    resultado_label.config(text=f"La temperatura estimada en la hora {horaIngresada} es {resultado:.2f} °C")

def elimGauss(campo11, campo12, campo13, campo21,campo22, campo23, campo31, campo32,campo33, campo41, campo42, campo43,resultado_label):
    campo_11 = float(campo11)
    campo_12 = float(campo12)
    campo_13 = float(campo13)
    campo_21 = float(campo21)
    campo_22 = float(campo22)
    campo_23 = float(campo23)
    campo_31 = float(campo31)
    campo_32 = float(campo32)
    campo_33 = float(campo33)

    campo_41 = float(campo41)
    campo_42 = float(campo42)
    campo_43 = float(campo43)


    # INGRESO DE DATOS-------------------------------
    # Se ingreserá primero los valores de la matriz y después los valores del vector según la forma Ax = B, escrita como arreglos.
    m = 3
    n = 3
    # La función np.zeros() es una función de la biblioteca NumPy en Python que se utiliza para crear un array() de ceros con las dimensiones especificadas.
    # En este caso la matriz A sera de 3x3, y el vector B de 3x1


    A = np.array([[campo_11, campo_21, campo_41],
                  [campo_12, campo_22, campo_42],
                  [campo_13, campo_23, campo_43]], dtype=float)

    B = np.array([[campo_31], [campo_32], [campo_33]], dtype=float)


    # PROCEDIMIENTO-------------------------------
    casicero = 1e-15  # Considerar como 0

    # HACER MATRIZ AUMENTADA
    # Se realiza al juntar la matriz A con el vector B en forma de columnas (axis=1).
    AB = np.concatenate((A, B), axis=1)
    AB0 = np.copy(AB)  # Copia de la matriz aumntada

    # INTERCAMBIO PARCIAL POR FILA
    # El Intercambio se realiza si la posición donde se encuentra el valor de mayor magnitud NO corresponde a la diagonal de la matriz, es decir la posición 0 de la columna.

    # Conoce el tamaño de la matriz aumentada y revisa la primera columna desde la diagonal en adelante
    tamano = np.shape(AB)
    n = tamano[0]
    m = tamano[1]

    # La función np.argmax() es una función de la biblioteca NumPy en Python que se utiliza para encontrar el índice del valor máximo en un array NumPy unidimensional o a lo largo de un eje específico en un array multidimensional. En este caso encontrará el valor máximo de la columna.
    #            dmax = np.argmax(columna)

    # Para cada fila en AB que se encuentren entre la primera y la penultima fila, avanzando en 1
    for i in range(0, n - 1, 1):
        # columna desde diagonal i en adelante
        columna = abs(AB[i:, i])  # Obtiene la columna de la matriz AB de todas las filas pero solo la columna i
        dmax = np.argmax(columna)

        # dmax no está en diagonal, es decir, no es 0
        if (dmax != 0):
            # Intercambio de filas
            # Se saca una copia de la fila i con todas sus columnas
            temporal = np.copy(AB[i, :])
            # Se reemplaza la fila i por la fila donde se encontró el valor máximo con todas sus columnas
            AB[i, :] = AB[dmax + i, :]
            # Y ahora la fila donde se encontró el valor máxima se reemplaza por la copia temporal
            AB[dmax + i, :] = temporal

    AB1 = np.copy(AB)  # Copia de la matriz con intercambio

    # ELIMINACIÓN GAUSSIANA
    # Se realizan operaciones con las filas inferiores para convertir los elementos por debajo de la diagonal en cero. Las operaciones incluyen el vector B debido a que se trabaja sobre la matriz aumentada AB.
    for i in range(0, n - 1, 1):
        pivote = AB[i, i]  # Utiliza el elemento en la diagonal principal como un pivote
        adelante = i + 1
        # k referencia las filas que están adelante de la diagonal
        for k in range(adelante, n, 1):
            factor = AB[
                         k, i] / pivote  # Obtener el factor escalar, como el valor del primer elemento de la fila k entre el pivote
            AB[k, :] = AB[k, :] - AB[i,
                                  :] * factor  # Se obtiene el resultado de la fila en adelante menos la fila de referencia multiplicada por un factor

    # SUSTITUCIÓN HACIA ATRÁS
    # Para una fila i, el vector b[i] representa el valor de la constante en la fila i de la matriz aumentada, a[i] se refiere los valores de los coeficientes de la ecuación, de los que se usan los que se encuentran a la derecha de la diagonal.

    # Las iteraciones empiezan desde la última fila y la última columna de la matriz resultante del procedimiento de ELIMINACION GAUSSIANA
    ultfila = n - 1
    ultcolumna = m - 1

    # X es el vector donde se definirán los resultados
    X = np.zeros(n, dtype=float)

    i = ultfila
    for i in range(ultfila, 0 - 1, -1):
        suma = 0
        for j in range(i + 1, ultcolumna, 1):
            suma = suma + AB[i, j] * X[
                j]  # Acumula los valores de los coeficientes en i,j multiplicados por los valores de las variables encontradas hacia la derecha de la diagonal
        b = AB[i, ultcolumna]  # La constante es el valor en la matriz con la posición de fila i y la ultima columna
        X[i] = (b - suma) / AB[
            i, i]  # El valor de la variable en la posición i es la constante menos el valor de la suma dividido entre el valor correspondiente a la diagonal

    # La función np.transpose() de la biblioteca NumPy de Python se utiliza para transponer matrices y cambiar sus dimensiones. La transposición de una matriz implica intercambiar filas por columnas.
    X = np.transpose([X])

    # SALIDA------------------------------------------
    print("Matriz inicial")
    print(AB0)
    print("Matriz resultante")
    print(AB)
    print("Solución")
    print(X)

def gaussJordan():

    # INGRESO DE DATOS-------------------------------
    # Se ingreserá primero los valores de la matriz y después los valores del vector según la forma Ax = B, escrita como arreglos.
    m = 3
    n = 3
    # La función np.zeros() es una función de la biblioteca NumPy en Python que se utiliza para crear un array() de ceros con las dimensiones especificadas.
    # En este caso la matriz A sera de 3x3, y el vector B de 3x1
    A = np.zeros((m, n), dtype=float)
    B = np.zeros((m, 1), dtype=float)

    # Ciclo for para ingresar los coeficientes en la matriz
    print("Introduce los coeficientes de la matriz:")
    for r in range(m):
        for c in range(n):
            valorM = float(input(f"Ingrese el coeficiente para la pocisión [{r + 1},{c + 1}]: "))
            A[r, c] = valorM

    print("Introduce los valores del vector:")
    for r in range(m):
        valorV = float(input(f"Ingrese el valor para la pocisión {r + 1}: "))
        B[r] = valorV

    # PROCEDIMIENTO-------------------------------
    # Parámetro para controlar los casos en que la diferencia entre los ordenes de magnitud son por ejemplo menores a 15 ordenes de magnitud 10-15. e implementarlo en el algoritmo.
    casicero = 1e-15  # Considerar como 0

    # HACER MATRIZ AUMENTADA
    # Se realiza al juntar la matriz A con el vector B en forma de columnas (axis=1).
    AB = np.concatenate((A, B), axis=1)
    AB0 = np.copy(AB)  # Copia de la matriz aumntada

    # INTERCAMBIO PARCIAL POR FILA
    # El Intercambio se realiza si la posición donde se encuentra el valor de mayor magnitud NO corresponde a la diagonal de la matriz, es decir la posición 0 de la columna.

    # Conoce el tamaño de la matriz aumentada y revisa la primera columna desde la diagonal en adelante
    tamano = np.shape(AB)
    n = tamano[0]
    m = tamano[1]

    # La función np.argmax() es una función de la biblioteca NumPy en Python que se utiliza para encontrar el índice del valor máximo en un array NumPy unidimensional o a lo largo de un eje específico en un array multidimensional. En este caso encontrará el valor máximo de la columna.
    #            dmax = np.argmax(columna)

    # Para cada fila en AB que se encuentren entre la primera y la penultima fila, avanzando en 1
    for i in range(0, n - 1, 1):
        # columna desde diagonal i en adelante
        columna = abs(AB[i:, i])  # Obtiene la columna de la matriz AB de todas las filas pero solo la columna i
        dmax = np.argmax(columna)

        # dmax no está en diagonal, es decir, no es 0
        if (dmax != 0):
            # Intercambio de filas
            # Se saca una copia de la fila i con todas sus columnas
            temporal = np.copy(AB[i, :])
            # Se reemplaza la fila i por la fila donde se encontró el valor máximo con todas sus columnas
            AB[i, :] = AB[dmax + i, :]
            # Y ahora la fila donde se encontró el valor máxima se reemplaza por la copia temporal
            AB[dmax + i, :] = temporal

    AB1 = np.copy(AB)  # Copia de la matriz con intercambio

    # ELIMINACIÓN HACIA ADELANTE (GAUSS)
    # Se realizan operaciones con las filas inferiores para convertir los elementos por debajo de la diagonal en cero. Las operaciones incluyen el vector B debido a que se trabaja sobre la matriz aumentada AB.
    for i in range(0, n - 1, 1):
        pivote = AB[i, i]  # Utiliza el elemento en la diagonal principal como un pivote
        adelante = i + 1
        # k referencia las filas que están adelante de la diagonal
        for k in range(adelante, n, 1):
            factor = AB[
                         k, i] / pivote  # Obtener el factor escalar, como el valor del primer elemento de la fila k entre el pivote
            AB[k, :] = AB[k, :] - AB[i,
                                  :] * factor  # Se obtiene el resultado de la fila en adelante menos la fila de referencia multiplicada por un factor

    AB2 = np.copy(AB)  # Copia de la matriz con intercambio

    # ELIMINACIÓN HACIA ATRÁS (JORDAN)
    # Realiza el mismo proceso que la eliminación hacia adelante, con la diferencia de que se inicia en la última fila hasta la primera.
    ultfila = n - 1
    ultcolumna = m - 1
    # Se aplica desde la última fila i, para todas las otras filas k que se encuentran hacia atrás.
    for i in range(ultfila, 0 - 1, -1):
        pivote = AB[i, i]  # Utiliza el elemento en la diagonal principal como un pivote
        atras = i - 1
        # k referencia las filas que están atrás de la diagonal
        for k in range(atras, 0 - 1, -1):
            factor = AB[
                         k, i] / pivote  # Obtener el factor escalar, como el valor del primer elemento de la fila k entre el pivote
            AB[k, :] = AB[k, :] - AB[i,
                                  :] * factor  # Se obtiene el resultado de la fila en adelante menos la fila de referencia multiplicada por un factor
        # Convertir el valor que está en la diagonal en uno
        AB[i, :] = AB[i, :] / AB[i, i]

    # El vector con las soluciones se obtiene de la última columna de la matriz aumentada con todas sus filas.
    X = np.copy(AB[:, ultcolumna])

    # La función np.transpose() de la biblioteca NumPy de Python se utiliza para transponer matrices y cambiar sus dimensiones. La transposición de una matriz implica intercambiar filas por columnas.
    X = np.transpose([X])

    # SALIDA------------------------------------------
    print("Matriz inicial")
    print(AB0)
    print("Matriz resultante")
    print(AB)
    print("Solución")
    print(X)

def gaussSeidel():

    # INGRESO DE DATOS-------------------------------
    # Se ingreserá primero los valores de la matriz y después los valores del vector según la forma Ax = B, escrita como arreglos.
    m = 3
    n = 3
    # La función np.zeros() es una función de la biblioteca NumPy en Python que se utiliza para crear un array() de ceros con las dimensiones especificadas.
    # En este caso la matriz A sera de 3x3, y el vector B de 3x1
    A = np.zeros((m, n), dtype=float)
    B = np.zeros((m, 1), dtype=float)

    # Ciclo for para ingresar los coeficientes en la matriz
    print("Introduce los coeficientes de la matriz:")
    for r in range(m):
        for c in range(n):
            valorM = float(input(f"Ingrese el coeficiente para la pocisión [{r + 1},{c + 1}]: "))
            A[r, c] = valorM

    print("Introduce los valores del vector:")
    for r in range(m):
        valorV = float(input(f"Ingrese el valor para la pocisión {r + 1}: "))
        B[r] = valorV

    # PROCEDIMIENTO-------------------------------

    # HACER MATRIZ AUMENTADA
    # Se realiza al juntar la matriz A con el vector B en forma de columnas (axis=1).
    AB = np.concatenate((A, B), axis=1)
    AB0 = np.copy(AB)  # Copia de la matriz aumntada

    # INTERCAMBIO PARCIAL POR FILA
    # El Intercambio se realiza si la posición donde se encuentra el valor de mayor magnitud NO corresponde a la diagonal de la matriz, es decir la posición 0 de la columna.

    # Conoce el tamaño de la matriz aumentada y revisa la primera columna desde la diagonal en adelante
    tamano = np.shape(AB)
    n = tamano[0]
    m = tamano[1]

    # La función np.argmax() es una función de la biblioteca NumPy en Python que se utiliza para encontrar el índice del valor máximo en un array NumPy unidimensional o a lo largo de un eje específico en un array multidimensional. En este caso encontrará el valor máximo de la columna.
    #            dmax = np.argmax(columna)

    # Para cada fila en AB que se encuentren entre la primera y la penultima fila, avanzando en 1
    for i in range(0, n - 1, 1):
        # columna desde diagonal i en adelante
        columna = abs(AB[i:, i])  # Obtiene la columna de la matriz AB de todas las filas pero solo la columna i
        dmax = np.argmax(columna)

        # dmax no está en diagonal, es decir, no es 0
        if (dmax != 0):
            # Intercambio de filas
            # Se saca una copia de la fila i con todas sus columnas
            temporal = np.copy(AB[i, :])
            # Se reemplaza la fila i por la fila donde se encontró el valor máximo con todas sus columnas
            AB[i, :] = AB[dmax + i, :]
            # Y ahora la fila donde se encontró el valor máxima se reemplaza por la copia temporal
            AB[dmax + i, :] = temporal

    AB1 = np.copy(AB)  # Copia de la matriz con intercambio

    A = AB[:, :-1]  # Selecciona todas las columnas excepto la última

    # GAUSS-SEIDEL
    X0 = np.array([0.0, 0, 0])  # X0 es el valor de origen
    val_tolera = 0.001  # valor que se tolera como error
    iteramax = 100  # Número máximo de iteraciones posibles

    # Conoce el tamaño de la matriz y revisa la primera columna desde la diagonal en adelante
    tamano = np.shape(A)
    n = tamano[0]
    m = tamano[1]

    # Valores iniciales
    X = np.copy(X0)  # Vector X es una copia del vector X0
    vec_dif = np.ones(n,
                      dtype=float)  # El vector diferencia no puede ser cero, ya que es el parámetro que permite comparar con el error
    val_error = 2 * val_tolera  # Se inicializa el valor de error con 2 veces el valor que se tolera como error

    itera = 0  # La iteración comienza en 0

    # Se repetirá el procedimiento hasta que el valor de error sea menor o igual al valor que se tolera o cuando el numero de iterraciones es mayor al de iteraciones máximas
    while not (val_error <= val_tolera or itera > iteramax):
        for i in range(0, n, 1):
            suma = 0  # Acumulador que empieza en 0
            for j in range(0, n, 1):
                # Los valores de j no pueden ser iguales a los de i
                if (j != i):
                    # Se suman las multiplicaciones de la matriz A en i,j por los valores del vector X en j
                    suma = suma + A[i, j] * X[j]
            nuevo = (B[i] - suma) / A[i, i]
            vec_dif[i] = np.abs(nuevo - X[
                i])  # Obtener el valor absoluto de la diferencia de el valor nuevo menor el valor que antes se tenía en el vector
            X[i] = nuevo  # nuevo será una variable para asignar los nuevos valores del vector X
        val_error = np.max(vec_dif)  # El valor errado es el valor máximo en el vector diferencia
        itera = itera + 1  # Aumenta en 1 la iteración

    # Advertencia de No convergencia
    if (itera > iteramax):
        X = 0
        print("AVISO: EL SISTEMA NO ES CONVERGENTE!!!")

    # SALIDA------------------------------------------
    print("Matriz inicial")
    print(AB0)
    print("Matriz resultante")
    print(AB)
    print("Solución")
    print(X)

def jacobi():


    # INGRESO DE DATOS-------------------------------
    # Se ingreserá primero los valores de la matriz y después los valores del vector según la forma Ax = B, escrita como arreglos.
    m = 3
    n = 3
    # La función np.zeros() es una función de la biblioteca NumPy en Python que se utiliza para crear un array() de ceros con las dimensiones especificadas.
    # En este caso la matriz A sera de 3x3, y el vector B de 3x1
    A = np.zeros((m, n), dtype=float)
    B = np.zeros((m, 1), dtype=float)

    # Ciclo for para ingresar los coeficientes en la matriz
    print("Introduce los coeficientes de la matriz:")
    for r in range(m):
        for c in range(n):
            valorM = float(input(f"Ingrese el coeficiente para la pocisión [{r + 1},{c + 1}]: "))
            A[r, c] = valorM

    print("Introduce los valores del vector:")
    for r in range(m):
        valorV = float(input(f"Ingrese el valor para la pocisión {r + 1}: "))
        B[r] = valorV

    # PROCEDIMIENTO-------------------------------

    # HACER MATRIZ AUMENTADA
    # Se realiza al juntar la matriz A con el vector B en forma de columnas (axis=1).
    AB = np.concatenate((A, B), axis=1)
    AB0 = np.copy(AB)  # Copia de la matriz aumntada

    # INTERCAMBIO PARCIAL POR FILA
    # El Intercambio se realiza si la posición donde se encuentra el valor de mayor magnitud NO corresponde a la diagonal de la matriz, es decir la posición 0 de la columna.

    # Conoce el tamaño de la matriz aumentada y revisa la primera columna desde la diagonal en adelante
    tamano = np.shape(AB)
    n = tamano[0]
    m = tamano[1]

    # La función np.argmax() es una función de la biblioteca NumPy en Python que se utiliza para encontrar el índice del valor máximo en un array NumPy unidimensional o a lo largo de un eje específico en un array multidimensional. En este caso encontrará el valor máximo de la columna.
    #            dmax = np.argmax(columna)

    # Para cada fila en AB que se encuentren entre la primera y la penultima fila, avanzando en 1
    for i in range(0, n - 1, 1):
        # columna desde diagonal i en adelante
        columna = abs(AB[i:, i])  # Obtiene la columna de la matriz AB de todas las filas pero solo la columna i
        dmax = np.argmax(columna)

        # dmax no está en diagonal, es decir, no es 0
        if (dmax != 0):
            # Intercambio de filas
            # Se saca una copia de la fila i con todas sus columnas
            temporal = np.copy(AB[i, :])
            # Se reemplaza la fila i por la fila donde se encontró el valor máximo con todas sus columnas
            AB[i, :] = AB[dmax + i, :]
            # Y ahora la fila donde se encontró el valor máxima se reemplaza por la copia temporal
            AB[dmax + i, :] = temporal

    AB1 = np.copy(AB)  # Copia de la matriz con intercambio

    A = AB[:, :-1]  # Selecciona todas las columnas excepto la última

    # JACOBI
    X0 = np.array([0.0, 0, 0])  # X0 es el valor de origen
    val_tolera = 0.001  # valor que se tolera como error
    iteramax = 100  # Número máximo de iteraciones posibles

    # Conoce el tamaño de la matriz y revisa la primera columna desde la diagonal en adelante
    tamano = np.shape(A)
    n = tamano[0]
    m = tamano[1]

    # Valores iniciales
    X = np.copy(X0)  # Vector X es una copia del vector X0
    vec_dif = np.ones(n,
                      dtype=float)  # El vector diferencia no puede ser cero, ya que es el parámetro que permite comparar con el error
    val_error = 2 * val_tolera  # Se inicializa el valor de error con 2 veces el valor que se tolera como error

    itera = 0  # La iteración comienza en 0

    # Se repetirá el procedimiento hasta que el valor de error sea menor o igual al valor que se tolera o cuando el numero de iterraciones es mayor al de iteraciones máximas
    while not (val_error <= val_tolera or itera > iteramax):
        for i in range(0, n, 1):
            suma = 0  # Acumulador que empieza en 0
            for j in range(0, n, 1):
                # Los valores de j no pueden ser iguales a los de i
                if (j != i):
                    # Se suman las multiplicaciones de la matriz A en i,j por los valores del vector X en j
                    suma = suma + A[i, j] * X[j]
            nuevo = (B[i] - suma) / A[i, i]
            vec_dif[i] = np.abs(nuevo - X[
                i])  # Obtener el valor absoluto de la diferencia de el valor nuevo menor el valor que antes se tenía en el vector
            X[i] = nuevo  # nuevo será una variable para asignar los nuevos valores del vector X
        val_error = np.max(vec_dif)  # El valor errado es el valor máximo en el vector diferencia
        itera = itera + 1  # Aumenta en 1 la iteración

    # Advertencia de No convergencia
    if (itera > iteramax):
        X = 0
        print("AVISO: EL SISTEMA NO ES CONVERGENTE!!!")

    # SALIDA------------------------------------------
    print("Matriz inicial")
    print(AB0)
    print("Matriz resultante")
    print(AB)
    print("Solución")
    print(X)

def montante():

    # INGRESO DE DATOS-------------------------------
    # Se ingreserá primero los valores de la matriz y después los valores del vector según la forma Ax = B, escrita como arreglos.
    m = 3
    n = 3
    # La función np.zeros() es una función de la biblioteca NumPy en Python que se utiliza para crear un array() de ceros con las dimensiones especificadas.
    # En este caso la matriz A sera de 3x3, y el vector B de 3x1
    A = np.zeros((m, n), dtype=float)
    B = np.zeros((m, 1), dtype=float)

    # Ciclo for para ingresar los coeficientes en la matriz
    print("Introduce los coeficientes de la matriz:")
    for r in range(m):
        for c in range(n):
            valorM = float(input(f"Ingrese el coeficiente para la pocisión [{r + 1},{c + 1}]: "))
            A[r, c] = valorM

    print("Introduce los valores del vector:")
    for r in range(m):
        valorV = float(input(f"Ingrese el valor para la pocisión {r + 1}: "))
        B[r] = valorV

    # PROCEDIMIENTO-------------------------------
    # Parámetro para controlar los casos en que la diferencia entre los ordenes de magnitud son por ejemplo menores a 15 ordenes de magnitud 10-15. e implementarlo en el algoritmo.
    casicero = 1e-15  # Considerar como 0

    # HACER MATRIZ AUMENTADA
    # Se realiza al juntar la matriz A con el vector B en forma de columnas (axis=1).
    AB = np.concatenate((A, B), axis=1)
    AB0 = np.copy(AB)  # Copia de la matriz aumntada

    # INTERCAMBIO PARCIAL POR FILA
    # El Intercambio se realiza si la posición donde se encuentra el valor de mayor magnitud NO corresponde a la diagonal de la matriz, es decir la posición 0 de la columna.

    # Conoce el tamaño de la matriz aumentada y revisa la primera columna desde la diagonal en adelante
    tamano = np.shape(AB)
    n = tamano[0]
    m = tamano[1]

    # La función np.argmax() es una función de la biblioteca NumPy en Python que se utiliza para encontrar el índice del valor máximo en un array NumPy unidimensional o a lo largo de un eje específico en un array multidimensional. En este caso encontrará el valor máximo de la columna.
    #            dmax = np.argmax(columna)

    # Para cada fila en AB que se encuentren entre la primera y la penultima fila, avanzando en 1
    for i in range(0, n - 1, 1):
        # columna desde diagonal i en adelante
        columna = abs(AB[i:, i])  # Obtiene la columna de la matriz AB de todas las filas pero solo la columna i
        dmax = np.argmax(columna)

        # dmax no está en diagonal, es decir, no es 0
        if (dmax != 0):
            # Intercambio de filas
            # Se saca una copia de la fila i con todas sus columnas
            temporal = np.copy(AB[i, :])
            # Se reemplaza la fila i por la fila donde se encontró el valor máximo con todas sus columnas
            AB[i, :] = AB[dmax + i, :]
            # Y ahora la fila donde se encontró el valor máxima se reemplaza por la copia temporal
            AB[dmax + i, :] = temporal

    # MONTANTE
    # Se realizan operaciones con las filas inferiores para convertir los elementos por debajo de la diagonal en cero. Las operaciones incluyen el vector B debido a que se trabaja sobre la matriz aumentada AB.
    for i in range(0, n - 1, 1):
        pivote = AB[i, i]  # Utiliza el elemento en la diagonal principal como un pivote
        adelante = i + 1
        # k referencia las filas que están adelante de la diagonal
        for k in range(adelante, n, 1):
            factor = AB[
                         k, i] / pivote  # Obtener el factor escalar, como el valor del primer elemento de la fila k entre el pivote
            AB[k, :] = AB[k, :] - AB[i,
                                  :] * factor  # Se obtiene el resultado de la fila en adelante menos la fila de referencia multiplicada por un factor

    # Realiza el mismo proceso que la eliminación hacia adelante, con la diferencia de que se inicia en la última fila hasta la primera.
    ultfila = n - 1
    ultcolumna = m - 1
    # Se aplica desde la última fila i, para todas las otras filas k que se encuentran hacia atrás.
    for i in range(ultfila, 0 - 1, -1):
        pivote = AB[i, i]  # Utiliza el elemento en la diagonal principal como un pivote
        atras = i - 1
        # k referencia las filas que están atrás de la diagonal
        for k in range(atras, 0 - 1, -1):
            factor = AB[
                         k, i] / pivote  # Obtener el factor escalar, como el valor del primer elemento de la fila k entre el pivote
            AB[k, :] = AB[k, :] - AB[i,
                                  :] * factor  # Se obtiene el resultado de la fila en adelante menos la fila de referencia multiplicada por un factor
        # Convertir el valor que está en la diagonal en uno
        AB[i, :] = AB[i, :] / AB[i, i]

    # El vector con las soluciones se obtiene de la última columna de la matriz aumentada con todas sus filas.
    X = np.copy(AB[:, ultcolumna])

    # La función np.transpose() de la biblioteca NumPy de Python se utiliza para transponer matrices y cambiar sus dimensiones. La transposición de una matriz implica intercambiar filas por columnas.
    X = np.transpose([X])

    # SALIDA------------------------------------------
    print("Matriz inicial")
    print(AB0)
    print("Matriz resultante")
    print(AB)
    print("Solución")
    print(X)

def NC_cerrada():
    # Tabla de constantes para las fórmulas Cerradas de Newton-Cotes
    con = [[3 / 2, 0, 1, 1, 0],
           [4 / 3, 0, 2, -1, 2, 0],
           [5 / 24, 0, 11, 1, 1, 11, 0],
           [6 / 20, 0, 11, -14, 26, -14, 11, 0],
           [7 / 1440, 0, 611, -453, 562, 562, -453, 611, 0],
           [8 / 945, 0, 460, -954, 2196, -2459, 2196, -954, 460, 0]]
    # limite inferior de la integral
    a = 0
    # limite superior de la integral
    b = 1
    # radio del tanque cilíndrico
    r = 2.5
    # cantidad de subintervalos
    n = 4

    """
    La forma de la integral es /a-b pi(r(x))^2 dx
    """
    # obtener valor de h
    h = (b - a) / (n + 2)

    # Calcular el valor de la sumatoria de la formula
    f_sum = 0
    for i, j in enumerate(con[n - 1], 1):
        f_sum += j * np.pi * math.pow(r * (a + (i - 1) * h), 2)

    # Calcula el resultado en base a la formula: alpha sum wi f(a+ih)
    res = con[n - 1][0] * h * f_sum
    print(res)

    return res

def NC_cerrada2():
    # Tabla de constantes para las fórmulas Cerradas de Newton-Cotes
    con = [[1 / 2, 1, 1],
           [1 / 3, 1, 4, 1],
           [3 / 8, 1, 3, 3, 1],
           [2 / 45, 7, 32, 12, 32, 7],
           [5 / 288, 19, 75, 50, 50, 75, 19],
           [1 / 140, 41, 216, 27, 272, 27, 216, 41],
           [7 / 17280, 751, 3577, 1323, 2989, 2989, 1323, 3577, 751],
           [14 / 14175, 989, 5888, -928, 10946, -4540, 10946, -928, 5888, 989],
           [9 / 89600, 2857, 15741, 1080, 19344, 5788, 5788, 19344, 1080, 15741, 2857],
           [5 / 299376, 16067, 106300, -48525, 272400, -260550, 427368, -260550, 272400, -48525, 106300, 16067]]
    # limite inferior de la integral
    a = 0
    # limite superior de la integral
    b = 1
    # radio del tanque cilíndrico
    r = 2.5
    # cantidad de subintervalos
    n = 4

    """
    La forma de la integral es /a-b pi(r(x))^2 dx
    """
    # obtener valor de h
    h = (b - a) / n

    # Calcular el valor de la sumatoria de la formula
    f_sum = 0
    for i, j in enumerate(con[n - 1], 1):
        f_sum += j * np.pi * math.pow(r * (a + (i - 1) * h), 2)

    # Calcula el resultado en base a la formula: alpha sum wi f(a+ih)
    res = con[n - 1, 0] * h * f_sum

    return res

def reglaTresOctavosSimpson(radio,largo,intervalos,resultado_label):

    radioCampo = float(radio)
    largoCampo = int(largo)
    intervalosCampo = int(intervalos)


    # limite inferior de la integral
    a = 0
    # limite superior de la integral
    b = largoCampo
    # radio del tanque cilíndrico
    r = radioCampo
    # cantidad de subintervalos
    n = intervalosCampo

    """
    La forma de la integral es /a-b pi(r(x))^2 dx
    """
    # obtener valor de h
    h = (b - a) / n

    """
    debido a que la formula contiene [f(a)+2 Sumatoria f(a+ih) + f(b)]
    se calculan de manera individual cada uno de los sumandos
    """
    # valor de la función en a
    fa = np.pi * math.pow(r * a, 2)

    # cálculo de la sumatoria de los subintervalos
    fsum = 0
    for i in range(1, n):
        fsum += np.pi * math.pow(r * (a + i * h), 2)

    # valor de la función en b
    fb = np.pi * math.pow(r * b, 2)

    # cálculo con la formula de la regla trapezoidal
    res = (3 / 8) * h * (fa + 3 * fsum + fb)
    resultado_label.config(text=f"El volumen es :  {res} ")
    print(res)
    return res

def reglaTercioSimpson():
    # limite inferior de la integral
    a = 0
    # limite superior de la integral
    b = 1
    # radio del tanque cilíndrico
    r = 2.5
    # cantidad de subintervalos
    n = 4

    """
    La forma de la integral es /a-b pi(r(x))^2 dx
    """
    # obtener valor de h
    h = (b - a) / n

    """
    debido a que la formula contiene [f(a)+ 4 Sumatoria [i=1,n-1] f(a+ih) + 2 Sumatoria [i=2,n-2] f(a+ih) + f(b)]
    se calculan de manera individual cada uno de los sumandos
    """
    # valor de la función en a
    fa = np.pi * math.pow(r * a, 2)

    # cálculo de la sumatoria de los subintervalos para los valores impares
    f_impar = 0
    for i in range(1, n, 2):
        f_impar += np.pi * math.pow(r * (a + i * h), 2)

    # cálculo de la sumatoria de los subintervalos para los valores pares
    f_par = 0
    for i in range(2, n, 2):
        f_par += np.pi * math.pow(r * (a + i * h), 2)

    # valor de la función en b
    fb = np.pi * math.pow(r * b, 2)

    # cálculo con la formula de la regla de 1/3 de simpson
    res = (h / 3) * (fa + 4 * f_impar + 2 * f_par + fb)
    print(res)
    return res

def reglaTrapezoidal():
    # limite inferior de la integral
    a = 0
    # limite superior de la integral
    b = 1
    # radio del tanque cilíndrico
    r = 2.5
    # cantidad de subintervalos
    n = 4

    """
    La forma de la integral es /a-b pi(r(x))^2 dx
    """
    # obtener valor de h
    h = (b - a) / n

    """
    debido a que la formula contiene [f(a)+2 Sumatoria f(a+ih) + f(b)]
    se calculan de manera individual cada uno de los sumandos
    """
    # valor de la función en a
    fa = np.pi * math.pow(r * a, 2)

    # cálculo de la sumatoria de los subintervalos
    fsum = 0
    for i in range(1, n):
        fsum += np.pi * math.pow(r * (a + i * h), 2)

    # valor de la función en b
    fb = np.pi * math.pow(r * b, 2)

    # cálculo con la formula de la regla trapezoidal
    res = (h / 2) * (fa + 2 * fsum + fb)
    print(res)
    return res



crearVentana()



