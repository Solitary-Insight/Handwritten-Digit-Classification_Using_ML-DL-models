import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import pickle

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train, X_test = X_train.reshape(-1, 28, 28, 1), X_test.reshape(-1, 28, 28, 1)
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Define CNN Model
cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
cnn.save("models/cnn_model.h5")

# Define LeNet Model
lenet = Sequential([
    Conv2D(6, (5,5), activation='relu', padding='valid', input_shape=(28,28,1)),
    AveragePooling2D(pool_size=(2,2)),
    Conv2D(16, (5,5), activation='relu', padding='valid'),
    AveragePooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
])

lenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lenet.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
lenet.save("models/lenet_model.h5")

