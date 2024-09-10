import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Dosya yükleme klasörü
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Modeli yükle
model = xgb.XGBClassifier()
model.load_model('my_xgb_model.json')

# Ana sayfa
@app.route('/')
def home():
    prediction = request.args.get("prediction")
    return render_template("home.html", prediction=prediction)

# Tahmin yapma
@app.route('/predict', methods=['POST'])
def predict():
    # Dosya yükleme işlemi
    if 'img' not in request.files:
        return "No file part"
    img_file = request.files['img']
    
    if img_file.filename == '':
        return "No selected file"
    
    # Dosya yolunu güvenli bir şekilde oluştur ve yükle
    filename = secure_filename(img_file.filename)
    
    # Dosya yükleme klasörünün var olup olmadığını kontrol et
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])  # Klasör yoksa oluştur
    
    # Dosyayı kaydet
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_file.save(img_path)
    
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
    
    # Sonucu göster
    return redirect(url_for("home", prediction=int(prediction[0])))

if __name__ == '__main__':
    app.run(debug=True)
