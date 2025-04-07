import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
train_data = pd.read_csv("./dataset/mnist_train.csv")
test_data = pd.read_csv("./dataset/mnist_test.csv")
dataset = pd.concat([train_data, test_data], ignore_index=True)

X, Y = dataset.iloc[:, 1:], dataset.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Random Forest Model
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
print("Random Forest Accuracy:", accuracy_score(y_test, rfc.predict(X_test)))

# Save Model
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rfc, f)

