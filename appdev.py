import os
import base64
import traceback
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# ------------------------------
# CONFIG
# ------------------------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
MODEL_PATH = "models/coffee_densenet121.h5"
model = load_model(MODEL_PATH)
CLASS_NAMES = ["Bad Coffee Bean", "Good Coffee Bean"]

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path, target_size=(224, 224)):
    """Load an image file and preprocess it for DenseNet prediction"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def encode_image(img_path):
    """Convert image to base64 string for HTML display"""
    with open(img_path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

# ------------------------------
# ROUTES
# ------------------------------
@app.route("/")
def home():
    return render_template("index.html", model_name="DenseNet121", img_size=224)

@app.route("/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist("files")
    results = []

    for f in files:
        if f.filename == "":
            results.append({"filename": f.filename, "error": "No filename provided"})
            continue

        if not allowed_file(f.filename):
            results.append({"filename": f.filename, "error": "Invalid file type"})
            continue

        filename = secure_filename(f.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

        try:
            # Save the uploaded file
            f.save(save_path)
            print(f"[INFO] Saved file to {save_path}")

            # Prepare image
            img_array = prepare_image(save_path)
            print(f"[INFO] Image shape after preprocessing: {img_array.shape}")

            # Predict
            preds = model.predict(img_array)
            print(f"[INFO] Raw model predictions: {preds}")

            class_idx = np.argmax(preds, axis=1)[0]
            predicted_class = CLASS_NAMES[class_idx]
            confidence = float(np.max(preds)) * 100
            print(f"[INFO] Predicted: {predicted_class} ({confidence:.2f}%)")

            # Encode image for frontend display
            img_encoded = encode_image(save_path)

            results.append({
                "filename": filename,
                "success": True,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 2),
                "image": img_encoded
            })

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}")
            traceback.print_exc()
            results.append({"filename": filename, "error": str(e)})

    return jsonify({
        "results": results,
        "total_files": len(files),
        "processed_files": len(results)
    })

# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
