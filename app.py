from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model("digit_recognizer_cnn.keras")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.files["image"]

    image = Image.open(image_data).convert("L")
    image = image.resize((28, 28))

    image_array = np.array(image)
    image_array = image_array / 255.0

    image_array = image_array.reshape(1, 28, 28, 1)

    predictions = model.predict(image_array)

    predicted_digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    return jsonify({
        "digit": predicted_digit,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)