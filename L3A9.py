import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_excel("embeddingsdata.xlsx")
binary_data = data[data['Label'].isin([0, 1])]

X_data = binary_data[['embed_1', 'embed_2']]
y_data = binary_data['Label']
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

neigh_classifier = KNeighborsClassifier(n_neighbors=3)
neigh_classifier.fit(X_train_data, y_train_data)
y_train_pred_data = neigh_classifier.predict(X_train_data)
confusion_matrix_train_data = confusion_matrix(y_train_data, y_train_pred_data)

y_test_pred_data = neigh_classifier.predict(X_test_data)
confusion_matrix_test_data = confusion_matrix(y_test_data, y_test_pred_data)

print("Confusion Matrix (Training Data):\n", confusion_matrix_train_data)
print("\nConfusion Matrix (Test Data):\n", confusion_matrix_test_data)

classification_report_train_data = classification_report(y_train_data, y_train_pred_data)
print("\nClassification Report (Training Data):\n", classification_report_train_data)

classification_report_test_data = classification_report(y_test_data, y_test_pred_data)
print("\nClassification Report (Test Data):\n", classification_report_test_data)
