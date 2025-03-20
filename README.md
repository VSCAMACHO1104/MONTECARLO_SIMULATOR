# MONTECARLO_SIMULATOR
Una app creada con código python para la generación de simulaciones Montecarlo para evaluar cantidad de pedido óptimo

Instalación: Instalar instancias en tu equipo ejecutando "pip install -r requirements.txt"
Ejecutar la app "Python -m streamlit run appMontecarlo.py"

Paso #1: Ingresar los valores de las variables deterministas no controlables, tal como el precio de venta y los volumenes de venta promedio correlacionados a estos precios como consecuencia de la elaticidad de la demanda del producto.
Paso #2: Ingresar las probabilidades no acumuladas asignadas entre los valores de precio y volumenes de venta. 
Paso #3: Ajustar la probabilidad acumulada mediante la sumatoria acumulativa de las probabilidades del paso 2. Asi mismo ajustar los límites inferior y superior según corresponda. 
Paso #4: Ingresar la desviación estándar para cada conjunto de datos correspondiente a cada volumen de ventas (media).
Paso #5: Ingresar los datos de costo (Costo de compra, costo de faltante o venta perdida, valor de rescate para venta de sobrantes) 
Paso #6: Inserta las cantidades de pedido a evaluar.
Paso #7: Inserta la cantidad de simulaciones deseada por cada cantidad de pedido a evaluar. 
