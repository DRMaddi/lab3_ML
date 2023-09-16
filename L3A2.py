import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_excel("embeddingsdata.xlsx")
feature1_data = dataframe['embed_1']

num_bins = 5
hist_counts, bin_edges = np.histogram(feature1_data, bins=num_bins)
mean_feature1 = np.mean(feature1_data)
variance_feature1 = np.var(feature1_data, ddof=1)

plt.hist(feature1_data, bins=num_bins, edgecolor='black', alpha=0.7)
plt.xlabel('Feature1')
plt.ylabel('Frequency')
plt.title('Histogram of Feature1')
plt.grid(True)
plt.show()
print(f'Mean of Feature1: {mean_feature1}')
print(f'Variance of Feature1: {variance_feature1}')

# Now, the same operations for 'embed_2'
feature2_data = dataframe['embed_2']  # Using 'embed_2'

hist_counts_2, bin_edges_2 = np.histogram(feature2_data, bins=num_bins)
mean_feature2 = np.mean(feature2_data)
variance_feature2 = np.var(feature2_data, ddof=1)

plt.hist(feature2_data, bins=num_bins, edgecolor='black', alpha=0.7)
plt.xlabel('Feature2')  # Change xlabel to 'Feature2'
plt.ylabel('Frequency')  # Change ylabel to 'Frequency'
plt.title('Histogram of Feature2')  # Change title to 'Histogram of Feature2'
plt.grid(True)
plt.show()
print(f'Mean of Feature2: {mean_feature2}')
print(f'Variance of Feature2: {variance_feature2}')
