import numpy as np
import simpy
import matplotlib.pyplot as plt
from collections import deque

class MMcKQueueLIFO:
    def __init__(self, env, num_servers, capacidad, tasa_llegada, tasa_servicio):
        self.env = env
        self.server = simpy.Resource(env, num_servers)
        self.capacidad = capacidad
        self.tasa_llegada = tasa_llegada
        self.tasa_servicio = tasa_servicio
        self.total_clientes = 0
        self.clientes_perdidos = 0
        self.tiempo_espera = []
        self.tiempo_sistema = []
        self.queue = deque()  # Cola de espera para implementar LIFO

    def cliente(self, nombre):
        arrival_time = self.env.now
        #print(f'{nombre} llega a las {arrival_time:.2f}')

        if len(self.server.queue) + self.server.count < self.capacidad:
            with self.server.request() as request:
                yield request
                if self.queue:
                    wait_time = self.env.now - self.queue.pop()  # LIFO: atender el último cliente en la cola
                else:
                    wait_time = self.env.now - arrival_time
                self.tiempo_espera.append(wait_time)
                #print(f'{nombre} entra al sistema a las {self.env.now:.2f} despues de esperar {wait_time:.2f}')
                service_time = np.random.exponential(1 / self.tasa_servicio)
                yield self.env.timeout(service_time)
                system_time = self.env.now - arrival_time
                self.tiempo_sistema.append(system_time)
                #print(f'{nombre} se va a las {self.env.now:.2f} despues de estar en el sistema por {system_time:.2f} horas')
        else:
            #print(f'{nombre} se va de la cola a las {self.env.now:.2f}')
            self.clientes_perdidos += 1

    def run(self):
        while True:
            yield self.env.timeout(np.random.exponential(1 / self.tasa_llegada))
            self.total_clientes += 1
            self.queue.append(self.env.now)  # Añadir el tiempo de llegada a la cola
            self.env.process(self.cliente(f'cliente {self.total_clientes}'))

def simulate_mmck_lifo(tasa_llegada, tasa_servicio, num_servers, capacidad, simulation_time):
    env = simpy.Environment()  # Nuevo entorno de simulación
    queue = MMcKQueueLIFO(env, num_servers, capacidad, tasa_llegada, tasa_servicio)
    env.process(queue.run())
    env.run(until=simulation_time)
    
    # Calcular métricas
    L = np.mean(queue.tiempo_sistema) * tasa_llegada * (1 - queue.clientes_perdidos / queue.total_clientes)
    Lq = np.mean(queue.tiempo_espera) * tasa_llegada * (1 - queue.clientes_perdidos / queue.total_clientes)
    W = np.mean(queue.tiempo_sistema)
    Wq = np.mean(queue.tiempo_espera)
    
    print(f'Total clientes: {queue.total_clientes}')
    print(f'Dropped clientes: {queue.clientes_perdidos}')
    print(f'Número promedio de clientes en el sistema (L): {L:.2f}')
    print(f'Número promedio de clientes en la cola (Lq): {Lq:.2f}')
    print(f'tiempo promedio que un cliente pasa en el sistema (W): {W:.2f}')
    print(f'tiempo promedio que un cliente pasa en la cola (Wq): {Wq:.2f}')
    
    return L, Lq, W, Wq

# Parámetros de simulación
tasa_llegada = 10  # Tasa de llegada λ
tasa_servicio = 12  # Tasa de servicio μ
num_servers = 5  # Número de servidores c
capacidad = 10  # Capacidad del sistema K
simulation_time = 1000  # Tiempo de simulación

# Variar los valores de c y K
valores_c = [1, 2, 3, 4, 5]  # Número de servidores c
valores_K = [5, 10, 15, 20]  # Capacidad del sistema K

resultados_L = np.zeros((len(valores_c), len(valores_K)))
resultados_Lq = np.zeros((len(valores_c), len(valores_K)))
resultados_W = np.zeros((len(valores_c), len(valores_K)))
resultados_Wq = np.zeros((len(valores_c), len(valores_K)))

# Ejecutar simulación para cada combinación de c y K
for i, c in enumerate(valores_c):
    for j, K in enumerate(valores_K):
        L, Lq, W, Wq = simulate_mmck_lifo(tasa_llegada, tasa_servicio, c, K, simulation_time)
        resultados_L[i, j] = L
        resultados_Lq[i, j] = Lq
        resultados_W[i, j] = W
        resultados_Wq[i, j] = Wq

# Graficar los resultados comparativos
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparación de Desempeño con Lógica LIFO')

# Función para graficar
def plot_metric(ax, metric, title, resultados):
    for i, c in enumerate(valores_c):
        metric_values = resultados[i, :]
        ax.plot(valores_K, metric_values, marker='o', label=f'c={c}')
    ax.set_xlabel('Capacidad del Sistema (K)')
    ax.set_ylabel(title)
    ax.legend()
    ax.grid(True)

# Graficar cada métrica
plot_metric(axs[0, 0], resultados_L, 'Número promedio de clientes en el sistema (L)', resultados_L)
plot_metric(axs[0, 1], resultados_Lq, 'Número promedio de clientes en la cola (Lq)', resultados_Lq)
plot_metric(axs[1, 0], resultados_W, 'Tiempo promedio en el sistema (W)', resultados_W)
plot_metric(axs[1, 1], resultados_Wq, 'Tiempo promedio en la cola (Wq)', resultados_Wq)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
