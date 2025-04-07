from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import io
import base64
from flask_cors import CORS

app = Flask(__name__)

# Load models
models = {
    "cnn": tf.keras.models.load_model("models/cnn_model.h5"),
    "lenet": tf.keras.models.load_model("models/lenet_model.h5"),
    "random_forest": pickle.load(open("models/random_forest.pkl", "rb")),
    "pca_knn": pickle.load(open("models/pca_knn.pkl", "rb")),
    "pca": pickle.load(open("models/pca.pkl", "rb")),
    "svm": pickle.load(open("models/svm.pkl", "rb"))
}

CORS(app)
# Route to serve HTML page
@app.route('/')
def index():
    return render_template('index.html')  # This will render the 'index.html' file

@app.route("/models", methods=["GET"])
def get_models():
    models_list = [m for m in list(models.keys()) if m != "pca"]
    return jsonify({"available_models": models_list})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    model_name = data.get("model")
    base64_image = data.get("image")

    # Validate model and image data
    if not model_name or model_name not in models:
        return jsonify({"error": "Invalid model selected."}), 400

    if not base64_image:
        return jsonify({"error": "No image provided."}), 400

    try:
        # Decode the base64 image
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert('L')
        image = image.resize((28, 28))  # Resize to 28x28 for MNIST
        image = np.array(image) / 255.0  # Normalize image
        image = image.reshape(1, 28, 28, 1)  # Reshape for CNN

    except Exception as e:
        return jsonify({"error": f"Failed to process the image: {str(e)}"}), 400

    # Make prediction using the selected model
    model = models[model_name]

    try:
        if model_name in ["cnn", "lenet"]:
            prediction = model.predict(image)
            prediction = np.argmax(prediction, axis=1)
        elif model_name in ["random_forest", "pca_knn", "svm"]:
            image_flat = image.reshape(1, -1)  # Flatten image for non-CNN models
            if model_name == "pca_knn":
                image_flat = models["pca"].transform(image_flat)  # Apply PCA
            prediction = model.predict(image_flat)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

