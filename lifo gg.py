import numpy as np
import random
import heapq
import matplotlib.pyplot as plt

# Funciones de distribución de tiempos
def generate_interarrival_time():
    return random.uniform(6, 12)

def generate_service_time():
    return random.uniform(8, 10)

# Simulación del sistema G/G/c/K con lógica FIFO
def simulate_ggck_fifo(c, K, simulation_time=10000):
    events = []
    clock = 0
    num_in_system = 0
    num_in_queue = 0
    servers = [0] * c

    total_wait_time = 0
    total_system_time = 0
    num_customers = 0
    wait_times = []
    queue = []

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
                    queue.append(clock)  # Añadir tiempo de llegada a la cola
            heapq.heappush(events, (clock + generate_interarrival_time(), 'Llegada'))

        elif event_type == 'Salida':
            num_in_system -= 1
            servers[servers.index(min(servers))] = 0
            if num_in_queue > 0:
                num_in_queue -= 1
                arrival_time = queue.pop(0)  # FIFO: atender al primer cliente en la cola
                wait_time = clock - arrival_time
                total_wait_time += wait_time
                service_time = generate_service_time()
                total_system_time += service_time
                wait_times.append(wait_time)
                heapq.heappush(events, (clock + service_time, 'Salida'))
                servers[servers.index(0)] = service_time

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
        W_q, W, L_q, L = simulate_ggck_fifo(c, K)
        results.append((c, K, W_q, W, L_q, L))

# Graficar los resultados
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Métricas de Desempeño para Diferentes Valores de c y K (FIFO)')

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
