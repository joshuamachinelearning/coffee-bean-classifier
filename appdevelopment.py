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
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# Model configuration - adjust these paths as needed
MODEL_PATH = "models/coffee_densenet121.h5"
IMG_SIZE = 224
CLASS_NAMES = ["Bad Coffee Bean", "Good Coffee Bean"]

# Load model
try:
    model = load_model(MODEL_PATH)
    print(f"[INFO] Successfully loaded model: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path, target_size=(224, 224)):
    """Load an image file and preprocess it for DenseNet prediction"""
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def encode_image(img_path):
    """Convert image to base64 string for HTML display"""
    try:
        with open(img_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        raise Exception(f"Error encoding image: {str(e)}")

def get_model_name():
    """Extract model name from path"""
    return os.path.basename(MODEL_PATH) if MODEL_PATH else "Unknown"

# ------------------------------
# ROUTES
# ------------------------------
@app.route("/")
def home():
    """Render the main page"""
    return render_template("index.html", 
                         model_name=get_model_name(), 
                         img_size=IMG_SIZE)

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image prediction requests"""
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected"}), 400

    results = []
    processed_count = 0

    for f in files:
        if f.filename == "":
            continue

        filename = secure_filename(f.filename)
        if not filename:
            results.append({
                "filename": f.filename, 
                "error": "Invalid filename"
            })
            continue

        if not allowed_file(filename):
            results.append({
                "filename": filename, 
                "error": "Invalid file type. Supported: PNG, JPG, JPEG, GIF"
            })
            continue

        # Create upload directory if it doesn't exist
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            # Save the uploaded file
            f.save(save_path)
            print(f"[INFO] Saved file to {save_path}")

            # Check file size after saving
            if os.path.getsize(save_path) > MAX_FILE_SIZE:
                os.remove(save_path)  # Clean up
                results.append({
                    "filename": filename,
                    "error": "File too large (max 16MB)"
                })
                continue

            # Prepare image for prediction
            img_array = prepare_image(save_path, target_size=(IMG_SIZE, IMG_SIZE))
            print(f"[INFO] Image shape after preprocessing: {img_array.shape}")

            # Make prediction
            preds = model.predict(img_array, verbose=0)
            print(f"[INFO] Raw predictions: {preds}")

            # Process prediction results
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
                "confidence": f"{confidence:.1f}",
                "image": img_encoded
            })
            
            processed_count += 1

            # Clean up uploaded file after processing
            try:
                os.remove(save_path)
            except:
                pass  # File cleanup is not critical

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {str(e)}")
            traceback.print_exc()
            
            # Clean up file if it exists
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
            except:
                pass

            results.append({
                "filename": filename, 
                "error": f"Processing failed: {str(e)}"
            })

    return jsonify({
        "results": results,
        "total_files": len([f for f in files if f.filename != ""]),
        "processed_files": processed_count,
        "success": processed_count > 0
    })

@app.route("/api/model-info")
def model_info():
    """Return information about the loaded model"""
    return jsonify({
        "model_name": get_model_name(),
        "model_loaded": model is not None,
        "img_size": IMG_SIZE,
        "class_names": CLASS_NAMES
    })

@app.route("/health")
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    })

# ------------------------------
# ERROR HANDLERS
# ------------------------------
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File too large (max 16MB)"}), 413

@app.errorhandler(500)
def server_error(e):
    """Handle server errors"""
    print(f"[ERROR] Server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    print(f"[INFO] Starting Coffee Bean Quality Classifier")
    print(f"[INFO] Model path: {MODEL_PATH}")
    print(f"[INFO] Model loaded: {'Yes' if model is not None else 'No'}")
    print(f"[INFO] Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"[INFO] Upload folder: {UPLOAD_FOLDER}")
    
    # Create necessary directories
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/uploads", exist_ok=True)
    
    app.run(host="0.0.0.0", port=5000, debug=True)