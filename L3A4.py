import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_excel("embeddingsdata.xlsx")
binary_data = data[data['Label'].isin([0, 1])]

X_new = binary_data[['embed_1', 'embed_2']]
y_new = binary_data['Label']

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.3, random_state=42)

print("X_train_new shape:", X_train_new.shape)
print("X_test_new shape:", X_test_new.shape)
print("y_train_new shape:", y_train_new.shape)
print("y_test_new shape:", y_test_new.shape)
