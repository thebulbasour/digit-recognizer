
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

model = tf.keras.models.load_model("digit_recognizer_cnn.keras")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.files["image"]
    image = Image.open(image_data).convert("L")
    image_array = np.array(image)

    digit_pixels = image_array > 30

    rows = np.any(digit_pixels, axis=1)
    cols = np.any(digit_pixels, axis=0)

    if not np.any(rows) or not np.any(cols):
        return jsonify({
            "error": "Please draw a digit first!"
        }), 400

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    cropped = image_array[
        row_min:row_max + 1,
        col_min:col_max + 1
    ]

    cropped_image = Image.fromarray(cropped)

    width, height = cropped_image.size
    scale = 20 / max(width, height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    cropped_image = cropped_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    final_image = Image.new("L", (28, 28), 0)

    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    final_image.paste(
        cropped_image,
        (x_offset, y_offset)
    )

    image_array = np.array(final_image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    predictions = model.predict(image_array, verbose=0)

    predicted_digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    return jsonify({
        "digit": predicted_digit,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
