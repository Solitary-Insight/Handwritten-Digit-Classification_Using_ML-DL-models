import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load dataset
train_data = pd.read_csv("./dataset/mnist_train.csv")
test_data = pd.read_csv("./dataset/mnist_test.csv")
dataset = pd.concat([train_data, test_data], ignore_index=True)

X, Y = dataset.iloc[:, 1:], dataset.iloc[:, 0]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)

# Save Model
with open("models/pca_knn.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("models/pca.pkl", "wb") as f:
    pickle.dump(pca, f)

