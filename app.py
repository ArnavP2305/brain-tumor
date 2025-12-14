from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model("brain_tumor_model.h5")

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence = ""

    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file)
        processed_img = preprocess_image(image)

        result = model.predict(processed_img)[0][0]

        if result > 0.5:
            prediction = "Tumor Detected"
            confidence = f"{result*100:.2f}%"
        else:
            prediction = "No Tumor Detected"
            confidence = f"{(1-result)*100:.2f}%"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
