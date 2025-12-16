import os
import urllib.request
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# ===============================
# MODEL DOWNLOAD CONFIG (GITHUB RELEASE)
# ===============================
MODEL_PATH = "brain_tumor_model.h5"
MODEL_URL = "https://github.com/ArnavP2305/brain-tumor/releases/download/v1.0/brain_tumor_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from GitHub Release...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)

# ===============================
# IMAGE PREPROCESSING FUNCTION
# ===============================
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = image.reshape(1, 224, 224, 3)
    return image

# ===============================
# ROUTES
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence = ""

    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file).convert("RGB")

        processed_image = preprocess_image(image)
        result = model.predict(processed_image)[0][0]

        if result > 0.5:
            prediction = "Tumor Detected"
            confidence = f"{result * 100:.2f}%"
        else:
            prediction = "No Tumor Detected"
            confidence = f"{(1 - result) * 100:.2f}%"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

# ===============================
# RUN APP (RENDER COMPATIBLE)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
