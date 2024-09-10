import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
import xgboost as xgb
import numpy as np
import threading
import uuid
import time

app = Flask(__name__)
CORS(app)

# Dosya yükleme klasörü
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Modeli yükle
model = xgb.XGBClassifier()
model.load_model('my_xgb_model.json')

# İşlerin durumu
jobs = {}

# Ana sayfa
@app.route('/')
def home():
    return "Flask API çalışıyor!"

# Uzun süren tahmin işlemini arka planda başlatır
def long_running_prediction(job_id, img_path):
    try:
        # Resmi işleme
        pic = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(pic)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Özellikleri ResNet152'den çıkar
        resnet_model = ResNet152(weights='imagenet', include_top=False, pooling='avg')
        features = resnet_model.predict(img_array)

        # Tahmin yap
        prediction = model.predict(features)

        # İşlem tamamlandı
        jobs[job_id] = {'status': 'completed', 'prediction': int(prediction[0])}
    except Exception as e:
        jobs[job_id] = {'status': 'failed', 'error': str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    if 'img' not in request.files:
        return jsonify({'error': 'No file part found'}), 400
    
    img_file = request.files['img']
    
    if img_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Dosya yolunu güvenli bir şekilde oluştur ve yükle
    filename = secure_filename(img_file.filename)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])  # Klasör yoksa oluştur
    
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_file.save(img_path)

    # İş kimliği oluştur ve arka planda tahmin işlemi başlat
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'pending'}

    # Arka planda işlemi başlat
    threading.Thread(target=long_running_prediction, args=(job_id, img_path)).start()

    # İş kimliğini döndür
    return jsonify({'jobId': job_id})

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = jobs.get(job_id)
    if job:
        return jsonify(job)
    else:
        return jsonify({'error': 'Job not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
