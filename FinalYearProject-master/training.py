from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Load your X-ray image and blood test data
def prepare_inputs(xray_image_path, blood_test_data):
    # Load and preprocess the X-ray image
    img = load_img(xray_image_path, target_size=(500, 500), color_mode='grayscale')
    xray_array = img_to_array(img) / 255.0
    xray_array = np.expand_dims(xray_array, axis=0)  # shape: (1, 500, 500, 1)

    # Prepare the blood test data
    blood_values = np.array([blood_test_data])  # shape: (1, 7)

    return [xray_array, blood_values]

# Load the model if exists, or create a new one
model_save_path = r'C:\Users\HP\Downloads\FinalYearProject-master\FinalYearProject-master\models\fusion_model.h5'

try:
    model = load_model(model_save_path)
    print("Loaded existing model.")
except:
    print("Model not found, creating a new model.")
    model, checkpoint = create_fusion_model(model_save_path=model_save_path)

# Sample inputs
xray_image_path = "your_image_path.jpg"  # Path to your X-ray image
blood_test_data = [5.2, 250, 60, 13.5, 12000, 15, 0.5]  # Example blood test data

inputs = prepare_inputs(xray_image_path, blood_test_data)

# Train the model
# Note: Replace this with your training data and epochs
model.fit(inputs, epochs=10, batch_size=32, validation_data=inputs, callbacks=[checkpoint])

# After training, the best model will be saved at `model_save_path`.
