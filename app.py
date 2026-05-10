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

    file = request.files.get('image')

    if not file or file.filename == "":
        return render_template("index.html")

    # save image
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # predict
    plant, confidence = predict_plant(filepath)

    # safe fallback (IMPORTANT FIX)
    details = plant_details.get(plant.lower(), {})

    return render_template(
        'index.html',

        prediction=plant,
        confidence=round(confidence, 2),

        image_path="/static/uploads/" + filename,

        # main values
        description=details.get("description", "No data available"),
        scientific_name=details.get("scientific_name", "Unknown"),
        plant_type=details.get("plant_type", "Unknown"),
        benefits=details.get("benefits", "No data available"),
        nutrients=details.get("nutrients", "No data available"),
        primary_consumers=details.get("primary_consumers", "Unknown"),
        medicinal_uses=details.get("medicinal_uses", "No data available"),
        toxicity=details.get("toxicity", "Unknown"),
        habitat=details.get("habitat", "Unknown"),
        water_requirement=details.get("water_requirement", "Unknown"),
        sunlight_requirement=details.get("sunlight_requirement", "Unknown"),
        uses=details.get("uses", "Unknown"),
        growth_rate=details.get("growth_rate", "Unknown"),
        lifespan=details.get("lifespan", "Unknown"),

        # info text
        scientific_name_info=details.get("scientific_name_info", ""),
        plant_type_info=details.get("plant_type_info", ""),
        benefits_info=details.get("benefits_info", ""),
        nutrients_info=details.get("nutrients_info", ""),
        primary_consumers_info=details.get("primary_consumers_info", ""),
        medicinal_uses_info=details.get("medicinal_uses_info", ""),
        toxicity_info=details.get("toxicity_info", ""),
        habitat_info=details.get("habitat_info", ""),
        water_requirement_info=details.get("water_requirement_info", ""),
        sunlight_requirement_info=details.get("sunlight_requirement_info", ""),
        uses_info=details.get("uses_info", ""),
        growth_rate_info=details.get("growth_rate_info", ""),
        lifespan_info=details.get("lifespan_info", ""),

        # links (optional safety fallback)
        scientific_name_url=details.get("scientific_name_link"),
        plant_type_url=details.get("plant_type_link"),
        benefits_url=details.get("benefits_link"),
        nutrients_url=details.get("nutrients_link"),
        primary_consumers_url=details.get("primary_consumers_link"),
        medicinal_uses_url=details.get("medicinal_uses_link"),
        toxicity_url=details.get("toxicity_link"),
        habitat_url=details.get("habitat_link"),
        water_requirement_url=details.get("water_requirement_link"),
        sunlight_requirement_url=details.get("sunlight_requirement_link"),
        uses_url=details.get("uses_link"),
        growth_rate_url=details.get("growth_rate_link"),
        lifespan_url=details.get("lifespan_link"),
        description_url=details.get("description_link"),
    )


# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)