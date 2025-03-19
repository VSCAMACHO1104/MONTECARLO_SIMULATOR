import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import json
from streamlit_lottie import st_lottie

# Set page configuration
st.set_page_config(
    page_title="ORDER QTY OPTIMIZATION",
    layout="wide"
)

# Function to load Lottie animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# App title
st.title(":gray[Simulación Monte Carlo - Optimización sobre Cantidad de Pedido]")
st.markdown("Esta aplicación determina la cantidad óptima de pedido que maximiza el beneficio basado en simulaciones Monte Carlo.")
st.markdown("Creado por:red[ VICTOR CAMACHO]")

# Create tabs for different sections
tab1, tab2 = st.tabs(["Datos de Entrada", "Resultados de Simulación"])

with tab1:
    st.header("Variables Deterministas No Controlables")
    
    # Create input for table data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Datos de Precio y Volumen")
        
        # Default values based on the provided table
        default_data = {
            'Precio de Venta ($)': [70, 72, 75, 77, 80],
            'Probabilidad': [0.15, 0.35, 0.25, 0.15, 0.10],
            'Volumen Ventas (Media)': [380, 420, 430, 400, 350],
            'Desv. Standard': [55, 65, 70, 65, 55],
            'Acumulada': [0.15, 0.50, 0.75, 0.90, 1.00],
            'Lim Inf': [0, 0.15, 0.5, 0.75, 0.9],
            'Lim Sup': [0.15, 0.5, 0.75, 0.9, 1]
        }
        
        # Create a DataFrame with default values
        df = pd.DataFrame(default_data)
        
        # Create an editable dataframe
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True
        )
    
    with col2:
        st.subheader("Datos de Costos")
        
        # Default cost values
        default_costs = {
            'Costos': ['Compra', 'Rescate', 'faltante'],
            'Monto de Costos': [45.00, 25.00, 15.00]
        }
        
        costs_df = pd.DataFrame(default_costs)
        
        # Create an editable dataframe for costs
        edited_costs_df = st.data_editor(
            costs_df,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True
        )
    
    st.header("Parámetros de Simulación")
    
    col3, col4 = st.columns(2)
    
    with col3:
        num_simulations = st.number_input("Número de Simulaciones", min_value=1000, max_value=1000000, value=100000, step=1000)
        
    with col4:
        # Input for order quantities to test
        order_quantities_text = st.text_area(
            "Cantidades de Pedido a Simular (ingrese hasta 30 valores separados por comas)", 
            "380, 390, 400, 410, 420, 430, 440"
        )
        
        # Convert text input to list of integers
        try:
            order_quantities = [int(x.strip()) for x in order_quantities_text.split(",") if x.strip()]
            if len(order_quantities) > 30:
                st.warning("Has ingresado más de 30 valores. Solo se usarán los primeros 30.")
                order_quantities = order_quantities[:30]
        except ValueError:
            st.error("Por favor ingrese valores numéricos válidos separados por comas.")
            order_quantities = [380, 400, 420, 440]

with tab2:
    # Load the Lottie animation from URL
    lottie_url = "https://lottie.host/b5ce745d-4940-400e-9471-633b1a87ac75/1ihXC84Idu.json"
    lottie_animation = load_lottieurl(lottie_url)
    
    # Function to find price based on random number and price table
    def find_price(random_value, price_table):
        for i, row in price_table.iterrows():
            if row['Lim Inf'] <= random_value < row['Lim Sup']:
                return row['Precio de Venta ($)']
        return price_table.iloc[-1]['Precio de Venta ($)']  # Default to last value if not found
    
    # Function to find mean and std dev for demand calculation based on price
    def find_demand_params(price, price_table):
        exact_match = price_table[price_table['Precio de Venta ($)'] == price]
        if not exact_match.empty:
            return exact_match.iloc[0]['Volumen Ventas (Media)'], exact_match.iloc[0]['Desv. Standard']
        
        # If no exact match, find closest price
        closest_price_idx = (price_table['Precio de Venta ($)'] - price).abs().idxmin()
        return price_table.iloc[closest_price_idx]['Volumen Ventas (Media)'], price_table.iloc[closest_price_idx]['Desv. Standard']
    
    # Function to calculate profit for a given order quantity
    def calculate_profit(order_quantity, price_table, costs_table):
        # Generate random values for price and demand
        price_random = np.random.random()  # APV
        price = find_price(price_random, price_table)  # PV
        
        mean, std_dev = find_demand_params(price, price_table)
        demand_random = np.random.random()  # AV
        demand = int(stats.norm.ppf(demand_random, loc=mean, scale=std_dev))  # VD
        demand = max(0, demand)  # Ensure demand is not negative
        
        # Calculate sales and inventory outcomes
        sales = min(demand, order_quantity)  # VR
        surplus = max(0, order_quantity - demand)  # SB
        shortage = max(0, demand - order_quantity)  # FT
        
        # Get cost values
        purchase_cost = costs_table[costs_table['Costos'] == 'Compra']['Monto de Costos'].values[0]
        salvage_value = costs_table[costs_table['Costos'] == 'Rescate']['Monto de Costos'].values[0]
        shortage_cost = costs_table[costs_table['Costos'] == 'faltante']['Monto de Costos'].values[0]
        
        # Calculate financial results
        revenue = price * sales  # Ingresos
        salvage_revenue = surplus * salvage_value  # Venta rescate
        purchase_expense = order_quantity * purchase_cost  # Compra
        shortage_expense = shortage * shortage_cost  # Costo faltante
        
        # Calculate profit
        profit = revenue + salvage_revenue - purchase_expense - shortage_expense  # BN Beneficios
        
        return {
            'APV': price_random, 
            'PV': price, 
            'AV': demand_random,
            'M': mean, 
            'DE': std_dev,
            'VD': demand,
            'CP': order_quantity,
            'VR': sales,
            'SB': surplus,
            'FT': shortage,
            'Ingresos': revenue,
            'Venta rescate': salvage_revenue,
            'Compra': purchase_expense,
            'Costo faltante': shortage_expense,
            'BN': profit
        }
    
    # Run simulations and display results when button is clicked
    if st.button("Ejecutar Simulación"):
        if len(order_quantities) == 0:
            st.error("Por favor ingrese al menos una cantidad de pedido para simular.")
        else:
            # Create container for animation and progress information
            animation_container = st.container()
            
            with animation_container:
                col_anim, col_progress = st.columns([1, 1])
                
                with col_anim:
                    # Display Lottie animation
                    if lottie_animation is not None:
                        st_lottie(lottie_animation, height=300, key="simulation_animation")
                    else:
                        st.warning("No se pudo cargar la animación. Continuando con la simulación...")
                
                with col_progress:
                    # Show a progress bar
                    progress_bar = st.progress(0)
                    
                    # Create containers for status messages
                    status_text = st.empty()
                    sim_count_text = st.empty()
                    current_qty_text = st.empty()
                    total_progress_text = st.empty()
            
            # Prepare to store results
            results = {}
            detailed_results = {}
            
            # Calculate total simulations to run
            total_simulations = len(order_quantities) * num_simulations
            completed_simulations = 0
            
            # Run simulations for each order quantity
            for i, qty in enumerate(order_quantities):
                current_qty_text.markdown(f"**Simulando para cantidad de pedido: {qty}**")
                
                # Store all profit values for this quantity for detailed analysis
                profit_values = []
                
                # Run simulations
                for sim in range(num_simulations):
                    sim_result = calculate_profit(qty, edited_df, edited_costs_df)
                    profit_values.append(sim_result['BN'])
                    
                    # Update simulation counters
                    completed_simulations += 1
                    
                    # Update progress every 1000 simulations to avoid UI slowdown
                    if sim % 1000 == 0 or sim == num_simulations - 1:
                        # Update progress bar
                        progress = completed_simulations / total_simulations
                        progress_bar.progress(progress)
                        
                        # Update status messages
                        percent_complete = progress * 100
                        sim_count_text.markdown(f"**Simulaciones completadas:** {completed_simulations:,} de {total_simulations:,}")
                        total_progress_text.markdown(f"**Progreso total:** {percent_complete:.1f}%")
                        
                        # Update status text with more details
                        status_text.markdown(f"**Estado:** Procesando cantidad {i+1} de {len(order_quantities)} ({qty} unidades)")
                
                # Store all results for this quantity
                detailed_results[qty] = profit_values
                
                # Calculate statistics
                results[qty] = {
                    'Cantidad de Pedido': qty,
                    'Beneficio Promedio': np.mean(profit_values),
                    'Desviación Estándar': np.std(profit_values),
                    'Mínimo': np.min(profit_values),
                    'Máximo': np.max(profit_values),
                    'Percentil 25%': np.percentile(profit_values, 25),
                    'Mediana': np.percentile(profit_values, 50),
                    'Percentil 75%': np.percentile(profit_values, 75)
                }
            
            # Clear animation container when done
            animation_container.empty()
            
            # Create results dataframe
            results_df = pd.DataFrame(results.values())
            
            # Display results
            st.header("Resultados de la Simulación")
            
            # Find optimal order quantity
            optimal_qty = results_df.loc[results_df['Beneficio Promedio'].idxmax()]['Cantidad de Pedido']
            
            # Display optimal quantity with highlight
            st.success(f"La cantidad óptima de pedido es: **{int(optimal_qty)}** unidades")
            st.info(f"Beneficio promedio máximo esperado: **${results_df['Beneficio Promedio'].max():.2f}**")
            
            # Display results table
            st.subheader("Tabla de Resultados")
            results_df['Cantidad de Pedido'] = results_df['Cantidad de Pedido'].astype(int)
            
            # Format the numeric columns in the dataframe
            formatted_results_df = results_df.copy()
            numeric_cols = formatted_results_df.select_dtypes(include=['float64']).columns
            for col in numeric_cols:
                formatted_results_df[col] = formatted_results_df[col].map('${:,.2f}'.format)
            
            st.dataframe(formatted_results_df, use_container_width=True, hide_index=True)
            
            # Display charts
            st.subheader("Gráficos de Resultados")
            
            col5, col6 = st.columns(2)
            
            with col5:
                # Create chart for average profit by order quantity
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(results_df['Cantidad de Pedido'], 
                         results_df['Beneficio Promedio'], 
                         marker='o', 
                         linestyle='-', 
                         color='blue')
                
                # Highlight optimal point
                ax1.plot(optimal_qty, 
                         results_df.loc[results_df['Beneficio Promedio'].idxmax()]['Beneficio Promedio'], 
                         'ro', 
                         markersize=10)
                
                ax1.set_title('Beneficio Promedio por Cantidad de Pedido')
                ax1.set_xlabel('Cantidad de Pedido')
                ax1.set_ylabel('Beneficio Promedio ($)')
                ax1.grid(True)
                st.pyplot(fig1)
            
            with col6:
                # Create boxplot of profit distribution for each order quantity
                boxplot_data = []
                labels = []
                
                # Select a subset of quantities if there are too many
                step = max(1, len(order_quantities) // 10)
                selected_quantities = order_quantities[::step]
                
                # Ensure optimal quantity is included
                if optimal_qty not in selected_quantities:
                    selected_quantities.append(int(optimal_qty))
                    selected_quantities.sort()
                
                for qty in selected_quantities:
                    boxplot_data.append(detailed_results[qty])
                    labels.append(str(qty))
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.boxplot(boxplot_data, labels=labels)
                ax2.set_title('Distribución de Beneficios por Cantidad de Pedido')
                ax2.set_xlabel('Cantidad de Pedido')
                ax2.set_ylabel('Beneficio ($)')
                ax2.grid(True)
                st.pyplot(fig2)
            
            # Show histogram for optimal quantity
            st.subheader(f"Distribución de Beneficios para la Cantidad Óptima ({int(optimal_qty)} unidades)")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.histplot(detailed_results[int(optimal_qty)], kde=True, ax=ax3)
            ax3.set_title(f'Distribución de Beneficios para Cantidad de Pedido = {int(optimal_qty)}')
            ax3.set_xlabel('Beneficio ($)')
            ax3.set_ylabel('Frecuencia')
            st.pyplot(fig3)
            
            # Show detailed statistics for optimal quantity
            st.subheader("Estadísticas Detalladas para la Cantidad Óptima")
            
            optimal_data = detailed_results[int(optimal_qty)]
            detailed_stats = {
                'Estadística': ['Media', 'Mediana', 'Desviación Estándar', 'Mínimo', 'Máximo', 
                                'Percentil 10%', 'Percentil 25%', 'Percentil 75%', 'Percentil 90%'],
                'Valor': [
                    f"${np.mean(optimal_data):.2f}",
                    f"${np.median(optimal_data):.2f}",
                    f"${np.std(optimal_data):.2f}",
                    f"${np.min(optimal_data):.2f}",
                    f"${np.max(optimal_data):.2f}",
                    f"${np.percentile(optimal_data, 10):.2f}",
                    f"${np.percentile(optimal_data, 25):.2f}",
                    f"${np.percentile(optimal_data, 75):.2f}",
                    f"${np.percentile(optimal_data, 90):.2f}"
                ]
            }
            
            stats_df = pd.DataFrame(detailed_stats)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Show a sample of simulation runs
            st.subheader("Muestra de Iteraciones de Simulación")
            
            # Run a few simulations with the optimal quantity for display
            sample_runs = []
            for _ in range(10):
                sample_runs.append(calculate_profit(int(optimal_qty), edited_df, edited_costs_df))
            
            sample_df = pd.DataFrame(sample_runs)
            
            # Format dataframe for display
            display_columns = ['CP', 'PV', 'VD', 'VR', 'SB', 'FT', 'Ingresos', 'Venta rescate', 'Compra', 'Costo faltante', 'BN']
            display_df = sample_df[display_columns].copy()
            
            # Format currency columns
            currency_cols = ['Ingresos', 'Venta rescate', 'Compra', 'Costo faltante', 'BN']
            for col in currency_cols:
                display_df[col] = display_df[col].map('${:,.2f}'.format)
            
            st.dataframe(display_df, use_container_width=True)
            
    else:
        st.info("Haga clic en 'Ejecutar Simulación' para comenzar el análisis.")