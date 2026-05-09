from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from plant_info import plant_categories

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model('model/plant_model.h5')

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def predict_plant(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    predicted_class = labels[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    return predicted_class, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        file.save(filepath)

        plant, confidence = predict_plant(filepath)
        category = plant_categories.get(plant, "unknown")

    return render_template(
    'index.html',
    prediction=plant,
    confidence=round(confidence, 2),
    category=category,
    image_path=filepath
)

if __name__ == '__main__':
    app.run(debug=True)