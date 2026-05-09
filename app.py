from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from plant_info import plant_details

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model('model/plant_model.h5')

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]


def predict_plant(img_path):

    img = image.load_img(img_path, target_size=(224, 224))

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

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            file.filename
        )

        file.save(filepath)

        plant, confidence = predict_plant(filepath)
        print("Predicted Plant:", plant)

        details = plant_details.get(plant.lower(), {

            "scientific_name": "Unknown",
            "plant_type": "Unknown",
            "benefits": "No data available",
            "nutrients": "No data available",
            "primary_consumers": "Unknown",
            "medicinal_uses": "No data available",
            "toxicity": "Unknown",
            "habitat": "Unknown",
            "water_requirement": "Unknown",
            "sunlight_requirement": "Unknown",
            "uses": "Unknown",
            "growth_rate": "Unknown",
            "lifespan": "Unknown"
        })

        return render_template(

            'index.html',

            prediction=plant,

            confidence=round(confidence, 2),

            scientific_name=details["scientific_name"],

            plant_type=details["plant_type"],

            benefits=details["benefits"],

            nutrients=details["nutrients"],

            primary_consumers=details["primary_consumers"],

            medicinal_uses=details["medicinal_uses"],

            toxicity=details["toxicity"],

            habitat=details["habitat"],

            water_requirement=details["water_requirement"],

            sunlight_requirement=details["sunlight_requirement"],

            uses=details["uses"],

            growth_rate=details["growth_rate"],

            lifespan=details["lifespan"],

            image_path=filepath
        )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)