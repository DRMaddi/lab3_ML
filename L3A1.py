import numpy as np
import pandas as pd

data = pd.read_excel("embeddingsdata.xlsx")

class_0_data = data[data['Label'] == 0]  
class_1_data = data[data['Label'] == 1]  
intra_class_var_0 = np.var(class_0_data[['embed_1', 'embed_2']], ddof=1)  
intra_class_var_1 = np.var(class_1_data[['embed_1', 'embed_2']], ddof=1)  
mean_class_0 = np.mean(class_0_data[['embed_1', 'embed_2']], axis=0)  
mean_class_1 = np.mean(class_1_data[['embed_1', 'embed_2']], axis=0)  
inter_class_dist = np.linalg.norm(mean_class_0 - mean_class_1)
print(f'Intraclass spread (variance) for Class 0: {intra_class_var_0}')
print(f'Intraclass spread (variance) for Class 1: {intra_class_var_1}')
print(f'Interclass distance between Class 0 and Class 1: {inter_class_dist}')

unique_labels = data['Label'].unique()
class_centroids = {}

for label in unique_labels:
    label_data = data[data['Label'] == label]
    label_mean = np.mean(label_data[['embed_1', 'embed_2']], axis=0)
    class_centroids[label] = label_mean

for label, centroid in class_centroids.items():
    print(f'Class {label} Centroid: {centroid}')

grouped_data = data.groupby('Label')
class_std_deviations = {}

for label, group in grouped_data:
    label_std = group[['embed_1', 'embed_2']].std(axis=0)
    class_std_deviations[label] = label_std
for label, std_dev in class_std_deviations.items():
    print(f'Standard Deviation for Class {label}:')
    for col, std in zip(std_dev.index, std_dev.values):
        print(f'  {col}: {std}')

grouped_data = data.groupby('Label')
class_centroids = {}

for label, group in grouped_data:
    label_mean = group[['embed_1', 'embed_2']].mean(axis=0)
    class_centroids[label] = label_mean

class_labels = list(class_centroids.keys())
num_classes = len(class_labels)
class_distances = {}

for i in range(num_classes):
    for j in range(i + 1, num_classes):
        label1 = class_labels[i]
        label2 = class_labels[j]
        distance = np.linalg.norm(class_centroids[label1] - class_centroids[label2])
        class_distances[(label1, label2)] = distance

for (label1, label2), dist in class_distances.items():
    print(f'Distance between Class {label1} and Class {label2}: {dist}')
