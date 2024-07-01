# COMPARACIÓN DE LOS SISTEMAS
import numpy as np

def mm1():
    L = 5
    Lq = 4.167
    W = 0.5 # horas
    Wq = 0.4167    # horas
    return L, Lq, W, Wq 

def mmc():  # con c=5
    L = 0.83367
    Lq = 0.00034561
    W = 0.08337 # horas
    Wq = 0.00003456    # horas
    return L, Lq, W, Wq 

def mm1k():     # con k=10
    L = 3.3
    Lq = 2.5
    W = 0.33 # horas
    Wq = 0.25    # horas
    return L, Lq, W, Wq 

def mmck():     # con c=5 y K=10
    L = 4.43
    Lq = 0.00034539
    W = 0.4433 # horas
    Wq = 0.00003454   # horas
    return L, Lq, W, Wq 

def comparacion(system1, system2, labels):
    results = {}
    for label, val1, val2 in zip(labels, system1, system2):
        results[label] = {
            'Sistema1': val1,
            'Sistema2': val2,
            'Diferencia entre sistema 1 y sistema 2': val1 - val2,
            'Eficiencia %': ((val1 - val2) / val2) * 100 if val2 != 0 else float('inf')
        }
    return results

# Cálculos
L_mm1, Lq_mm1, W_mm1, Wq_mm1 = mm1()
L_mmc, Lq_mmc, W_mmc, Wq_mmc = mmc()
L_mm1k, Lq_mm1k, W_mm1k, Wq_mm1k = mm1k()
L_mmck, Lq_mmck, W_mmck, Wq_mmck = mmck()

# Comparaciones
labels = ['L', 'Lq', 'W', 'Wq']
comparison_mm1_mmc = comparacion((L_mm1, Lq_mm1, W_mm1, Wq_mm1), (L_mmc, Lq_mmc, W_mmc, Wq_mmc), labels)
comparison_mm1_mm1k = comparacion((L_mm1, Lq_mm1, W_mm1, Wq_mm1), (L_mm1k, Lq_mm1k, W_mm1k, Wq_mm1k), labels)
comparison_mm1_mmck= comparacion((L_mm1, Lq_mm1, W_mm1, Wq_mm1), (L_mmck, Lq_mmck, W_mmck, Wq_mmck), labels)
comparison_mmc_mm1k= comparacion((L_mmc, Lq_mmc, W_mmc, Wq_mmc), (L_mm1k, Lq_mm1k, W_mm1k, Wq_mm1k), labels)
comparison_mmc_mmck= comparacion((L_mmc, Lq_mmc, W_mmc, Wq_mmc), (L_mmck, Lq_mmck, W_mmck, Wq_mmck), labels)
comparison_mm1k_mmck= comparacion((L_mm1k, Lq_mm1k, W_mm1k, Wq_mm1k), (L_mmck, Lq_mmck, W_mmck, Wq_mmck), labels)

# Resultados
print("Comparación M/M/1 vs M/M/c:")
for metric, values in comparison_mm1_mmc.items():
    print(f"{metric}: {values}")

print("\nComparación M/M/1 vs M/M/1/K:")
for metric, values in comparison_mm1_mm1k.items():
    print(f"{metric}: {values}")

print("\nComparación M/M/1 vs M/M/c/K:")
for metric, values in comparison_mm1_mmck.items():
    print(f"{metric}: {values}")

print("\nComparación M/M/c vs M/M/1/K:")
for metric, values in comparison_mmc_mm1k.items():
    print(f"{metric}: {values}")

print("\nComparación M/M/c vs M/M/c/K:")
for metric, values in comparison_mmc_mmck.items():
    print(f"{metric}: {values}")

print("\nComparación M/M/1/K vs M/M/c/K:")
for metric, values in comparison_mm1k_mmck.items():
    print(f"{metric}: {values}")

