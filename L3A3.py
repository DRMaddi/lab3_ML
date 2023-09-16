import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

data = pd.read_excel("embeddingsdata.xlsx")

vec_a = np.array([data['embed_1'][0], data['embed_2'][0]])
vec_b = np.array([data['embed_1'][3], data['embed_2'][3]])

r_values = list(range(1, 11))

distances = []
for r in r_values:
    minkowski_distance = distance.minkowski(vec_a, vec_b, p=r)
    distances.append(minkowski_distance)

plt.plot(r_values, distances, marker='o', linestyle='-', color='b')
plt.xlabel('r (Minkowski Parameter)')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance vs. r')
plt.grid(True)
plt.show()
