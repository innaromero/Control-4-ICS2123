import numpy as np
import simpy
import matplotlib.pyplot as plt

class MMcKQueue:
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

    def cliente(self, nombre):
        arrival_time = self.env.now
        #print(f'{nombre} llega a las {arrival_time:.2f}')
        
        if len(self.server.queue) + self.server.count < self.capacidad:
            with self.server.request() as request:
                yield request
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
            self.env.process(self.cliente(f'cliente {self.total_clientes}'))

def simulate_mmck(tasa_llegada, tasa_servicio, num_servers, capacidad, simulation_time):
    env = simpy.Environment()   # Nuevo entorno de simulación
    queue = MMcKQueue(env, num_servers, capacidad, tasa_llegada, tasa_servicio)
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
    
    return  L, Lq, W, Wq

# Parámetros de simulación
tasa_llegada = 10  # Tasa de llegada λ
tasa_servicio = 12  # Tasa de servicio μ
num_servers = 5     # Número de servidores c
capacidad = 10        # Capacidad del sistema K
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
        L, Lq, W, Wq = simulate_mmck(tasa_llegada, tasa_servicio, c, K, simulation_time)
        resultados_L[i, j] = L
        resultados_Lq[i, j] = Lq
        resultados_W[i, j] = W
        resultados_Wq[i, j] = Wq


# Graficar resultados
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for i, c in enumerate(valores_c):
    for j, K in enumerate(valores_K):
        axs[0, 0].plot(K, resultados_L[i, j], marker='o', label=f'c={c}')
        axs[0, 0].set_title('Número promedio de clientes en el sistema (L)')
        axs[0, 0].set_xlabel('K (Capacidad del sistema)')
        axs[0, 0].set_ylabel('L')
        

        axs[0, 1].plot(K, resultados_Lq[i, j], marker='o', label=f'c={c}')
        axs[0, 1].set_title('Número promedio de clientes en la cola (Lq)')
        axs[0, 1].set_xlabel('K (Capacidad del sistema)')
        axs[0, 1].set_ylabel('Lq')
       

        axs[1, 0].plot(K, resultados_W[i, j], marker='o', label=f'c={c}')
        axs[1, 0].set_title('Tiempo promedio de estancia en el sistema (W)')
        axs[1, 0].set_xlabel('K (Capacidad del sistema)')
        axs[1, 0].set_ylabel('W')
        

        axs[1, 1].plot(K, resultados_Wq[i, j], marker='o', label=f'c={c}')
        axs[1, 1].set_title('Tiempo promedio de espera en la cola (Wq)')
        axs[1, 1].set_xlabel('K (Capacidad del sistema)')
        axs[1, 1].set_ylabel('Wq')


plt.tight_layout()
plt.show()

# Ejemplo de gráficos de sensibilidad
# Puedes crear gráficos para cada métrica (L, Lq, W, Wq) variando c y K
# Aquí un ejemplo para L

plt.figure(figsize=(10, 6))
for j in range(len(valores_K)):
    plt.plot(valores_c, resultados_L[:, j], marker='o', label=f'K={valores_K[j]}')
plt.xlabel('Número de servidores (c)')
plt.ylabel('Número promedio de clientes en el sistema (L)')
plt.title('Análisis de sensibilidad: Número promedio de clientes en el sistema (L)')
plt.legend(title='Capacidad del sistema (K)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for j in range(len(valores_K)):
    plt.plot(valores_c, resultados_Lq[:, j], marker='o', label=f'K={valores_K[j]}')
plt.xlabel('Número de servidores (c)')
plt.ylabel('Número promedio de clientes en el sistema (Lq)')
plt.title('Análisis de sensibilidad: Número promedio de clientes en el sistema (Lq)')
plt.legend(title='Capacidad del sistema (K)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for j in range(len(valores_K)):
    plt.plot(valores_c, resultados_W[:, j], marker='o', label=f'K={valores_K[j]}')
plt.xlabel('Número de servidores (c)')
plt.ylabel('Número promedio de clientes en el sistema (W)')
plt.title('Análisis de sensibilidad: Número promedio de clientes en el sistema (W)')
plt.legend(title='Capacidad del sistema (K)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for j in range(len(valores_K)):
    plt.plot(valores_c, resultados_Wq[:, j], marker='o', label=f'K={valores_K[j]}')
plt.xlabel('Número de servidores (c)')
plt.ylabel('Número promedio de clientes en el sistema (Wq)')
plt.title('Análisis de sensibilidad: Número promedio de clientes en el sistema (Wq)')
plt.legend(title='Capacidad del sistema (K)')
plt.grid(True)
plt.show()
