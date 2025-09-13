import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import base64
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'png','jpg','jpeg','gif','bmp'}

# password can come from env var for security
app.config['MODEL_PASSWORD'] = os.environ.get('MODEL_PASSWORD','secret123')

IMG_SIZE = 224
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

loaded_models = {}  # name -> tf.keras model

def load_model(model_name):
    path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    if model_name not in loaded_models:
        loaded_models[model_name] = tf.keras.models.load_model(path)
    return loaded_models[model_name]

# default model
DEFAULT_MODEL = 'coffee_densenet.h5'
CLASS_LABELS = ['Bad Coffee Bean','Good Coffee Bean']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((IMG_SIZE,IMG_SIZE))
    arr = np.expand_dims(np.array(img),axis=0)
    arr = preprocess_input(arr.astype(np.float32))
    return arr

def predict_image(image_path, model_name):
    try:
        model = load_model(model_name)
    except Exception as e:
        return None,None,f"Error loading model: {str(e)}"
    try:
        arr = preprocess_image(image_path)
        preds = model.predict(arr)
        cls = int(np.argmax(preds[0]))
        conf = float(preds[0][cls])
        return cls,conf,None
    except Exception as e:
        return None,None,f"Prediction error: {str(e)}"

@app.route('/')
def index():
    model_files = [f for f in os.listdir(app.config['MODEL_FOLDER']) if f.endswith('.h5')]
    return render_template('index.html', models=model_files, default_model=DEFAULT_MODEL)

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form.get('model_name', DEFAULT_MODEL)
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error':'No files uploaded'})

    results=[]
    for i,file in enumerate(files):
        if not allowed_file(file.filename):
            results.append({'filename':file.filename,'error':'Invalid file type'})
            continue

        filename=f"{int(time.time()*1000)}_{i}_{secure_filename(file.filename)}"
        filepath=os.path.join(app.config['UPLOAD_FOLDER'],filename)
        file.save(filepath)

        cls,conf,error=predict_image(filepath,model_name)
        if error:
            results.append({'filename':file.filename,'error':error})
        else:
            with open(filepath,'rb') as f:
                img_b64=base64.b64encode(f.read()).decode()
            results.append({
                'filename':file.filename,
                'success':True,
                'predicted_class':CLASS_LABELS[cls],
                'confidence':f"{conf*100:.2f}%",  # exact percentage
                'confidence_raw':conf,
                'image':f"data:image/jpeg;base64,{img_b64}",
                'model_used': model_name
            })
        try: os.remove(filepath)
        except: pass

    return jsonify({'success':True,'results':results})

@app.route('/health')
def health():
    return jsonify({'status':'healthy','models':list(loaded_models.keys())})

if __name__=="__main__":
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)),debug=True)
