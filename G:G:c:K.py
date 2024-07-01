import numpy as np
import random
import heapq
import matplotlib.pyplot as plt

# Funciones de distribución de tiempos
def generate_interarrival_time():
    return random.uniform(6, 12)

def generate_service_time():
    return random.uniform(8, 10)

# Simulación del sistema G/G/c/K
def simulate_ggck(c, K, simulation_time=10000):
    events = []
    clock = 0
    num_in_system = 0
    num_in_queue = 0
    servers = [0] * c

    total_wait_time = 0
    total_system_time = 0
    num_customers = 0
    wait_times = []

    heapq.heappush(events, (generate_interarrival_time(), 'Llegada'))

    while clock < simulation_time:
        event_time, event_type = heapq.heappop(events)
        clock = event_time

        if event_type == 'Llegada':
            if num_in_system < K:
                num_in_system += 1
                num_customers += 1
                if 0 in servers:
                    service_time = generate_service_time()
                    total_system_time += service_time
                    servers[servers.index(0)] = service_time
                    heapq.heappush(events, (clock + service_time, 'Salida'))
                else:
                    num_in_queue += 1
            heapq.heappush(events, (clock + generate_interarrival_time(), 'Llegada'))

        elif event_type == 'Salida':
            num_in_system -= 1
            for i, server_time in enumerate(servers):
                if server_time == 0:
                    servers[i] = 0
                    break
            if num_in_queue > 0:
                num_in_queue -= 1
                service_time = generate_service_time()
                total_wait_time += clock - (clock - service_time)
                total_system_time += service_time
                wait_times.append(clock - (clock - service_time))
                heapq.heappush(events, (clock + service_time, 'departure'))

    W_q = total_wait_time / num_customers
    W = total_system_time / num_customers
    L_q = W_q * (num_customers / simulation_time)
    L = W * (num_customers / simulation_time)

    return W_q, W, L_q, L

# Valores de c y K para la simulación
c_values = [1, 2, 3, 4, 5]
K_values = [5, 10, 15, 20, 25]

# Resultados de las simulaciones
results = []

for c in c_values:
    for K in K_values:
        W_q, W, L_q, L = simulate_ggck(c, K)
        results.append((c, K, W_q, W, L_q, L))

# Graficar los resultados
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Métricas de Desempeño para Diferentes Valores de c y K')

for metric, ax, title in zip([2, 3, 4, 5], axs.flatten(), 
                             ['Tiempo promedio en la cola (Wq)', 
                              'Tiempo promedio en el sistema (W)', 
                              'Número promedio de clientes en la cola (Lq)', 
                              'Número promedio de clientes en el sistema (L)']):
    for c in c_values:
        metric_values = [result[metric] for result in results if result[0] == c]
        ax.plot(K_values, metric_values, marker='o', label=f'c={c}')
    ax.set_xlabel('Capacidad del Sistema (K)')
    ax.set_ylabel(title)
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Valores de c y K para la simulación
c_values = [1, 2, 3, 4, 5]
K_values = [5, 10, 15, 20, 25]

# Resultados de las simulaciones
results = []

for c in c_values:
    for K in K_values:
        W_q, W, L_q, L = simulate_ggck(c, K)
        results.append((c, K, W_q, W, L_q, L))

# Preparar matrices para almacenar los resultados de las métricas
W_q_matrix = np.zeros((len(c_values), len(K_values)))
W_matrix = np.zeros((len(c_values), len(K_values)))
L_q_matrix = np.zeros((len(c_values), len(K_values)))
L_matrix = np.zeros((len(c_values), len(K_values)))

# Llenar matrices con los valores de las métricas
for result in results:
    c_index = c_values.index(result[0])
    K_index = K_values.index(result[1])
    W_q_matrix[c_index, K_index] = result[2]
    W_matrix[c_index, K_index] = result[3]
    L_q_matrix[c_index, K_index] = result[4]
    L_matrix[c_index, K_index] = result[5]

# Crear gráfico de sensibilidad (heatmap)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis de Sensibilidad para el Sistema G/G/c/K')

# Heatmap para W_q
im = axs[0, 0].imshow(W_q_matrix, cmap='viridis', aspect='auto', origin='lower',
                      extent=[min(K_values), max(K_values), min(c_values), max(c_values)])
axs[0, 0].set_title('Tiempo promedio en la cola (W_q)')
axs[0, 0].set_xlabel('Capacidad del Sistema (K)')
axs[0, 0].set_ylabel('Número de Servidores (c)')
fig.colorbar(im, ax=axs[0, 0], label='Valor')

# Heatmap para W
im = axs[0, 1].imshow(W_matrix, cmap='viridis', aspect='auto', origin='lower',
                      extent=[min(K_values), max(K_values), min(c_values), max(c_values)])
axs[0, 1].set_title('Tiempo promedio en el sistema (W)')
axs[0, 1].set_xlabel('Capacidad del Sistema (K)')
axs[0, 1].set_ylabel('Número de Servidores (c)')
fig.colorbar(im, ax=axs[0, 1], label='Valor')

# Heatmap para L_q
im = axs[1, 0].imshow(L_q_matrix, cmap='viridis', aspect='auto', origin='lower',
                      extent=[min(K_values), max(K_values), min(c_values), max(c_values)])
axs[1, 0].set_title('Número promedio de clientes en la cola (L_q)')
axs[1, 0].set_xlabel('Capacidad del Sistema (K)')
axs[1, 0].set_ylabel('Número de Servidores (c)')
fig.colorbar(im, ax=axs[1, 0], label='Valor')

# Heatmap para L
im = axs[1, 1].imshow(L_matrix, cmap='viridis', aspect='auto', origin='lower',
                      extent=[min(K_values), max(K_values), min(c_values), max(c_values)])
axs[1, 1].set_title('Número promedio de clientes en el sistema (L)')
axs[1, 1].set_xlabel('Capacidad del Sistema (K)')
axs[1, 1].set_ylabel('Número de Servidores (c)')
fig.colorbar(im, ax=axs[1, 1], label='Valor')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()