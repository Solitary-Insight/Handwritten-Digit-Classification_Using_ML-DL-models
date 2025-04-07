import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
train_data = pd.read_csv("./dataset/mnist_train.csv")
test_data = pd.read_csv("./dataset/mnist_test.csv")
dataset = pd.concat([train_data, test_data], ignore_index=True)

X, Y = dataset.iloc[:, 1:], dataset.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train SVM Model
svm = SVC(kernel="poly", C=1.0)
svm.fit(X_train, y_train)

# Save Model
with open("models/svm.pkl", "wb") as f:
    pickle.dump(svm, f)

