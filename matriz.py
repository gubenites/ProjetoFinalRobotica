import numpy as np

matriz = np.array([1,2,2])
soma = 0
for item in matriz:
    soma+=item
matriz= matriz/soma
print(matriz)
