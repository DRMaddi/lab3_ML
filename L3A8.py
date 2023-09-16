import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_excel("embeddingsdata.xlsx")
binary_data = data[data['Label'].isin([0, 1])]

X_data = binary_data[['embed_1', 'embed_2']]
y_data = binary_data['Label']

accuracies_k_neighbors = []
accuracies_nearest_neighbor = []

X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
k_values = range(1, 12)

for k in k_values:
    k_neighbors_classifier = KNeighborsClassifier(n_neighbors=k)
    k_neighbors_classifier.fit(X_train_data, y_train_data)
    
    y_pred_k_neighbors = k_neighbors_classifier.predict(X_test_data)
       
    accuracy_k_neighbors = accuracy_score(y_test_data, y_pred_k_neighbors)
    accuracies_k_neighbors.append(accuracy_k_neighbors)
           
    nearest_neighbor_classifier = KNeighborsClassifier(n_neighbors=1)
    nearest_neighbor_classifier.fit(X_train_data, y_train_data)
               
    y_pred_nearest_neighbor = nearest_neighbor_classifier.predict(X_test_data)
                
    accuracy_nearest_neighbor = accuracy_score(y_test_data, y_pred_nearest_neighbor)
    accuracies_nearest_neighbor.append(accuracy_nearest_neighbor)

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies_k_neighbors, marker='o', label='k-Nearest Neighbors (k=3)')
plt.plot(k_values, accuracies_nearest_neighbor, marker='o', label='Nearest Neighbor (k=1)')

plt.title('Accuracy vs. k Value')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()
