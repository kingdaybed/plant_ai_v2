from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from plant_info import plant_details
import uuid

app = Flask(__name__)

# =========================
# CONFIG
# =========================
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# LOAD LABELS
# =========================
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# =========================
# LOAD TFLITE MODEL
# =========================
interpreter = tf.lite.Interpreter(model_path="model/plant_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# PREDICTION FUNCTION
# =========================
def predict_plant(img_path):

    img = image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Run TFLite model
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    return predicted_class, confidence

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    # ✅ ADD THIS (IMPORTANT SAFETY CHECK)
    if file.filename == "":
        return render_template("index.html")

    if file:

        filename = str(uuid.uuid4()) + ".jpg"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        plant, confidence = predict_plant(filepath)

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
            description=details["description"],
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

            scientific_name_info=details["scientific_name_info"],
            plant_type_info=details["plant_type_info"],
            benefits_info=details["benefits_info"],
            nutrients_info=details["nutrients_info"],
            primary_consumers_info=details["primary_consumers_info"],
            medicinal_uses_info=details["medicinal_uses_info"],
            toxicity_info=details["toxicity_info"],
            habitat_info=details["habitat_info"],
            water_requirement_info=details["water_requirement_info"],
            sunlight_requirement_info=details["sunlight_requirement_info"],
            uses_info=details["uses_info"],
            growth_rate_info=details["growth_rate_info"],
            lifespan_info=details["lifespan_info"],

            scientific_name_url=details.get("scientific_name_url"),
            plant_type_url=details.get("plant_type_url"),
            benefits_url=details.get("benefits_url"),
            nutrients_url=details.get("nutrients_url"),
            primary_consumers_url=details.get("primary_consumers_url"),
            medicinal_uses_url=details.get("medicinal_uses_url"),
            toxicity_url=details.get("toxicity_url"),
            habitat_url=details.get("habitat_url"),
            water_requirement_url=details.get("water_requirement_url"),
            sunlight_requirement_url=details.get("sunlight_requirement_url"),
            uses_url=details.get("uses_url"),
            growth_rate_url=details.get("growth_rate_url"),
            lifespan_url=details.get("lifespan_url"),

            image_path="/static/uploads/" + filename


        )

    return render_template('index.html')


# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)