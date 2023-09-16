import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_excel("embeddingsdata.xlsx")

binary_data = data[data['Label'].isin([0, 1])]

X_data = binary_data[['embed_1', 'embed_2']]
y_data = binary_data['Label']

X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

neigh_classifier = KNeighborsClassifier(n_neighbors=3)

neigh_classifier.fit(X_train_data, y_train_data)

accuracy_score = neigh_classifier.score(X_test_data, y_test_data)
print("Accuracy:", accuracy_score)

test_data_vector = X_test_data.iloc[0]

predicted_data_class = neigh_classifier.predict([test_data_vector])

print("Predicted Class:", predicted_data_class[0])
