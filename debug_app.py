import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import base64
import time

# Create Flask app instance
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# For testing without TensorFlow first
TESTING_MODE = True  # Set to False when you want to use real model

if not TESTING_MODE:
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.densenet import preprocess_input
        from PIL import Image
        
        MODEL_PATH = 'models/your_model.h5'  # Update this path
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        TESTING_MODE = True
        print("→ Switching to testing mode")

# Class labels
CLASS_LABELS = ['Bad Coffee Bean', 'Good Coffee Bean']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def mock_prediction():
    """Mock prediction for testing"""
    import random
    predicted_class = random.randint(0, 1)
    confidence = random.uniform(0.7, 0.95)
    return predicted_class, confidence, None

def real_prediction(image_path):
    """Real prediction using loaded model"""
    try:
        from PIL import Image
        # Load and resize image
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array.astype(np.float32))
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return predicted_class, confidence, None
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/debug')
def debug_page():
    return render_template('debug.html')

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({
        'status': 'working',
        'testing_mode': TESTING_MODE,
        'upload_folder_exists': os.path.exists(UPLOAD_FOLDER)
    })

@app.route('/predict', methods=['POST'])
def predict():
    print("\n" + "="*50)
    print("🔍 PREDICT ENDPOINT CALLED")
    print(f"Request method: {request.method}")
    print(f"Request files keys: {list(request.files.keys())}")
    print(f"Request form keys: {list(request.form.keys())}")
    
    try:
        # Get files from request
        files = []
        if 'files' in request.files:
            files = request.files.getlist('files')
            print(f"✓ Found {len(files)} files under 'files' key")
        elif 'file' in request.files:
            files = [request.files['file']]
            print(f"✓ Found 1 file under 'file' key")
        else:
            print("✗ No files found in request")
            return jsonify({'error': 'No files uploaded'})
        
        # Filter valid files
        valid_files = [f for f in files if f and f.filename and f.filename.strip() != '']
        print(f"✓ Valid files after filtering: {len(valid_files)}")
        
        if not valid_files:
            print("✗ No valid files after filtering")
            return jsonify({'error': 'No valid files selected'})
        
        for i, f in enumerate(valid_files):
            print(f"  File {i+1}: {f.filename} ({f.content_type if hasattr(f, 'content_type') else 'unknown type'})")
        
        results = []
        
        for i, file in enumerate(valid_files):
            print(f"\n📁 Processing file {i+1}: {file.filename}")
            
            # Check file type
            if not allowed_file(file.filename):
                error_msg = f"Invalid file type for {file.filename}"
                print(f"✗ {error_msg}")
                results.append({
                    'filename': file.filename,
                    'error': error_msg
                })
                continue
            
            try:
                # Save file
                timestamp = str(int(time.time() * 1000))
                safe_filename = secure_filename(file.filename)
                unique_filename = f"{timestamp}_{i}_{safe_filename}"
                filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
                
                print(f"💾 Saving to: {filepath}")
                file.save(filepath)
                
                # Check if file was saved
                if not os.path.exists(filepath):
                    raise Exception("File was not saved properly")
                
                file_size = os.path.getsize(filepath)
                print(f"✓ File saved successfully ({file_size} bytes)")
                
                # Make prediction
                print("🤖 Making prediction...")
                if TESTING_MODE:
                    predicted_class, confidence, error = mock_prediction()
                    print("✓ Using mock prediction")
                else:
                    predicted_class, confidence, error = real_prediction(filepath)
                    print("✓ Using real model prediction")
                
                if error:
                    print(f"✗ Prediction error: {error}")
                    results.append({
                        'filename': file.filename,
                        'error': error
                    })
                else:
                    print(f"✓ Prediction: {CLASS_LABELS[predicted_class]} ({confidence:.3f})")
                    
                    # Convert to base64
                    print("🖼️ Converting image to base64...")
                    with open(filepath, "rb") as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode()
                    print("✓ Image converted to base64")
                    
                    results.append({
                        'filename': file.filename,
                        'success': True,
                        'predicted_class': CLASS_LABELS[predicted_class],
                        'confidence': f"{confidence * 100:.2f}%",
                        'confidence_raw': confidence,
                        'image': f"data:image/jpeg;base64,{img_base64}"
                    })
                
                # Cleanup
                try:
                    os.remove(filepath)
                    print("✓ File cleaned up")
                except Exception as cleanup_error:
                    print(f"⚠️ Cleanup warning: {cleanup_error}")
                
            except Exception as file_error:
                error_msg = f"Processing error: {str(file_error)}"
                print(f"✗ {error_msg}")
                results.append({
                    'filename': file.filename,
                    'error': error_msg
                })
        
        response_data = {
            'success': True,
            'results': results,
            'total_files': len(valid_files),
            'processed_files': len(results)
        }
        
        print(f"\n✅ RESPONSE READY:")
        print(f"   Total files: {len(valid_files)}")
        print(f"   Results: {len(results)}")
        print(f"   Successful: {len([r for r in results if r.get('success')])}")
        print(f"   Errors: {len([r for r in results if r.get('error')])}")
        print("="*50 + "\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f"💥 CRITICAL ERROR: {error_msg}")
        print("="*50 + "\n")
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    print("🚀 Starting Coffee Bean Classifier (Debug Mode)")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🧪 Testing mode: {TESTING_MODE}")
    print(f"🔗 Test endpoint: http://localhost:5000/test")
    print(f"🔧 Debug page: http://localhost:5000/debug")
    print("-" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)