from flask import Flask, request, render_template, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from weasyprint import HTML
import tensorflow as tf
import numpy as np
import joblib
import os
from datetime import datetime
import io
import logging
from logging import handlers
import cv2
import psutil
import gc
from fusion_model import focal_loss

app = Flask(__name__)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = handlers.RotatingFileHandler('app.log', maxBytes=1000000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"OpenCV version: {cv2.__version__}")
logger.info(f"Server running at http://localhost:5000")

# Blood test fields with reference ranges
FIELDS = [
    ('wbc', {'label': 'White Blood Cell (WBC) Count', 'range': '4.0-11.0', 'unit': 'x10^9/L'}),
    ('neutrophils', {'label': 'Neutrophils', 'range': '2.0-7.5', 'unit': 'x10^9/L'}),
    ('lymphocytes', {'label': 'Lymphocytes', 'range': '1.0-3.5', 'unit': 'x10^9/L'}),
    ('crp', {'label': 'C-Reactive Protein (CRP)', 'range': '0.0-10.0', 'unit': 'mg/L'}),
    ('esr', {'label': 'Erythrocyte Sedimentation Rate (ESR)', 'range': '0.0-20.0', 'unit': 'mm/hr'}),
    ('platelets', {'label': 'Platelets', 'range': '150.0-450.0', 'unit': 'x10^9/L'}),
    ('hemoglobin', {'label': 'Hemoglobin', 'range': '12.0-16.0', 'unit': 'g/dL'}),
]

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'output_heatmaps')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
XRAY_MODEL_PATH = os.path.join(MODEL_DIR, 'pneu_cnn_model.h5')
FUSION_MODEL_PATH = os.path.join(MODEL_DIR, 'fusion_model2.h5')
BLOOD_MODEL_PATH = os.path.join(MODEL_DIR, 'pneumonia_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'pneumonia_scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'pneumonia_label_encoder.pkl')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def setup_directories():
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info(f"Upload directory: {UPLOAD_FOLDER}")
        logger.info(f"Output directory: {OUTPUT_FOLDER}")
        logger.info(f"Model directory: {MODEL_DIR}")
    except Exception as e:
        logger.error(f"Error setting up directories: {str(e)}")
        raise RuntimeError(f"Failed to set up directories: {str(e)}")

def load_model_safely(model_path, model_name, custom_objects=None):
    if not os.path.exists(model_path):
        logger.error(f"{model_name} model file not found at {model_path}")
        raise FileNotFoundError(f"{model_name} model file not found at {model_path}")
    try:
        tf.keras.backend.clear_session()
        model = load_model(model_path, custom_objects=custom_objects)
        logger.info(f"{model_name} model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading {model_name} model: {str(e)}")
        raise RuntimeError(f"Failed to load {model_name} model: {str(e)}")

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU memory growth enabled")
    else:
        logger.info("No GPU found, using CPU")
except Exception as e:
    logger.warning(f"Error configuring GPU: {str(e)}")

try:
    setup_directories()
except Exception as e:
    logger.error(f"Server startup failed: {str(e)}")
    raise

try:
    xray_model = load_model_safely(XRAY_MODEL_PATH, "X-ray CNN")
except Exception as e:
    logger.error(f"Server startup failed: {str(e)}")
    raise

try:
    fusion_model = load_model_safely(
        FUSION_MODEL_PATH,
        "Fusion",
        custom_objects={'focal_loss_fn': focal_loss(gamma=1.5, alpha=0.25)}
    )
except Exception as e:
    logger.warning(f"Error loading fusion model: {str(e)}. Using biomarker-based fallback.")
    fusion_model = None

try:
    if not os.path.exists(BLOOD_MODEL_PATH):
        raise FileNotFoundError(f"Blood test model file not found at {BLOOD_MODEL_PATH}")
    blood_model = joblib.load(BLOOD_MODEL_PATH)
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder file not found at {LABEL_ENCODER_PATH}")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    logger.info("Blood test model, scaler, and label encoder loaded successfully")
except Exception as e:
    logger.error(f"Server startup failed: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_gradcam(model, img_array, layer_name='conv2d_1'):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(
            tf.multiply(conv_outputs, pooled_grads[..., tf.newaxis]),
            axis=-1
        )
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img = img_array[0, :, :, 0] * 255.0
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        superimposed_img = heatmap * 0.4 + img
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        heatmap_path = os.path.join(OUTPUT_FOLDER, f'heatmap_{timestamp}.png')
        cv2.imwrite(heatmap_path, superimposed_img)
        return heatmap_path
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {str(e)}")
        return None

def predict_blood_test(blood_values):
    try:
        logger.debug(f"Raw blood_values: {blood_values}, shape: {np.array(blood_values).shape}")
        # Ensure blood_values is a NumPy array with shape (1, 7)
        if isinstance(blood_values, np.ndarray):
            if blood_values.ndim == 1:
                blood_values = blood_values.reshape(1, -1)
            elif blood_values.ndim > 2:
                blood_values = blood_values.reshape(1, -1)
        else:
            blood_values = np.array(blood_values, dtype=float).reshape(1, -1)
        
        if blood_values.shape != (1, 7):
            logger.error(f"Invalid blood values shape: {blood_values.shape}, expected (1, 7)")
            raise ValueError(f"Expected blood_values shape (1, 7), got {blood_values.shape}")

        # Scale features and predict
        features_scaled = scaler.transform(blood_values)
        logger.debug(f"Scaled features: {features_scaled}")
        
        prediction = blood_model.predict(features_scaled)
        probabilities = blood_model.predict_proba(features_scaled)[0]
        label = label_encoder.inverse_transform(prediction)[0]
        
        if label == 'Pneumonia':
            wbc, neutrophils, lymphocytes, crp, esr, platelets, hemoglobin = blood_values[0]
            if neutrophils > 7.5 and crp > 10.0:
                label = 'Bacterial'
                confidence = probabilities.max()
            else:
                label = 'Viral'
                confidence = probabilities.max()
        else:
            label = 'Healthy'
            confidence = probabilities.max()
            
        importances = blood_model.feature_importances_
        feature_names = [f[0] for f in FIELDS]
        
        logger.info(f"Blood test prediction: {label} (Confidence: {confidence:.2%})")
        logger.info("Blood test feature importance:")
        for name, importance in zip(feature_names, importances):
            logger.info(f"{FIELDS[feature_names.index(name)][1]['label']}: {importance:.4f}")
            
        del blood_values, features_scaled, prediction, probabilities
        gc.collect()
        
        return {
            'label': label,
            'confidence': confidence,
            'importances': dict(zip([f[1]['label'] for f in FIELDS], importances))
        }
    except Exception as e:
        logger.error(f"Error in blood test prediction: {str(e)}")
        return None, f"Blood test prediction failed: {str(e)}"

def predict_xray(image_path):
    try:
        logger.info(f"Attempting to load X-ray image from: {image_path}")
        if not os.path.exists(image_path):
            logger.error(f"X-ray image not found at: {image_path}")
            raise FileNotFoundError(f"X-ray image not found at: {image_path}")
        
        tf.keras.backend.clear_session()
        
        img = load_img(image_path, target_size=(500, 500), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = xray_model.predict(img_array, verbose=0)[0]
        label = 'Pneumonia' if prediction[0] > 0.5 else 'Normal'
        confidence = prediction[0] if label == 'Pneumonia' else 1.0 - prediction[0]
        
        if label == 'Pneumonia':
            label = 'Bacterial' if confidence > 0.7 else 'Viral'
        else:
            label = 'Healthy'
            
        heatmap_path = generate_gradcam(xray_model, img_array)
        
        del img_array
        tf.keras.backend.clear_session()
        gc.collect()
        
        logger.info(f"X-ray prediction: {label} (Confidence: {confidence:.2%})")
        return {'label': label, 'confidence': confidence, 'heatmap_path': heatmap_path}
    except Exception as e:
        logger.error(f"Error in X-ray prediction: {str(e)}")
        return None, f"X-ray prediction failed: {str(e)}"

def predict_fusion(xray_image_path, blood_values):
    try:
        logger.debug(f"Fusion input - X-ray path: {xray_image_path}, Blood values: {blood_values}")
        if not os.path.exists(xray_image_path):
            logger.error(f"X-ray image not found at: {xray_image_path}")
            raise FileNotFoundError(f"X-ray image not found at: {xray_image_path}")
        xray_result = predict_xray(xray_image_path)
        if not xray_result or isinstance(xray_result, tuple):
            error_msg = xray_result[1] if isinstance(xray_result, tuple) else "X-ray prediction failed"
            raise ValueError(error_msg)
        blood_result = predict_blood_test(blood_values)
        if not blood_result or isinstance(blood_result, tuple):
            logger.warning(f"Blood test prediction failed: {blood_result[1] if isinstance(blood_result, tuple) else 'Unknown error'}. Falling back to X-ray result.")
            return {
                'label': xray_result['label'],
                'confidence': xray_result['confidence'],
                'heatmap_path': xray_result.get('heatmap_path')
            }
        if fusion_model:
            img = load_img(xray_image_path, target_size=(128, 128), color_mode='grayscale')
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            blood_input = scaler.transform(blood_values)
            fusion_pred = fusion_model.predict([img_array, blood_input], verbose=0)
            label_idx = np.argmax(fusion_pred, axis=1)[0]
            label = ['Healthy', 'Viral', 'Bacterial'][label_idx]
            confidence = fusion_pred[0][label_idx]
            logger.info(f"Fusion model prediction: {label} (Confidence: {confidence:.2%})")
        else:
            logger.warning("Fusion model unavailable. Using biomarker-based fallback.")
            neutrophils, crp, lymphocytes = blood_values[0][1], blood_values[0][3], blood_values[0][2]
            if xray_result['label'] == blood_result['label']:
                label = xray_result['label']
                confidence = 0.5 * xray_result['confidence'] + 0.5 * blood_result['confidence']
            elif neutrophils > 7.5 and crp > 10.0:
                label = 'Bacterial'
                confidence = blood_result['confidence'] * 0.7 + xray_result['confidence'] * 0.3
            elif lymphocytes > 3.5:
                label = 'Viral'
                confidence = blood_result['confidence'] * 0.7 + xray_result['confidence'] * 0.3
            else:
                label = blood_result['label'] if blood_result['confidence'] > xray_result['confidence'] else xray_result['label']
                confidence = max(blood_result['confidence'], xray_result['confidence'])
            logger.info(f"Biomarker-based prediction: {label} (Confidence: {confidence:.2%})")
        return {'label': label, 'confidence': min(confidence, 1.0), 'heatmap_path': xray_result.get('heatmap_path')}
    except Exception as e:
        logger.error(f"Error in fusion prediction: {str(e)}. X-ray path: {xray_image_path}, Blood values: {blood_values}")
        if xray_result and not isinstance(xray_result, tuple):
            logger.info("Returning X-ray result as fallback")
            return {
                'label': xray_result['label'],
                'confidence': xray_result['confidence'],
                'heatmap_path': xray_result.get('heatmap_path')
            }
        return {'label': 'N/A', 'confidence': 0.0, 'heatmap_path': None}

def create_report(submit_type, xray_result=None, blood_result=None, fusion_result=None, blood_values=None, validated_blood_values=None):
    try:
        xray_result = xray_result if isinstance(xray_result, dict) else {'label': 'N/A', 'confidence': 0.0, 'heatmap_path': 'N/A'}
        blood_result = blood_result if isinstance(blood_result, dict) else {'label': 'N/A', 'confidence': 0.0, 'importances': {}}
        fusion_result = fusion_result if isinstance(fusion_result, dict) else {'label': 'N/A', 'confidence': 0.0}
        blood_values = blood_values if blood_values else ['N/A'] * len(FIELDS)
        heatmap_path = xray_result.get('heatmap_path', 'N/A').replace('\\', '/') if xray_result.get('heatmap_path') else 'N/A'
        if heatmap_path != 'N/A':
            heatmap_path = os.path.join('output_heatmaps', os.path.basename(heatmap_path)).replace('\\', '/')
        format_values = []
        if validated_blood_values is not None and isinstance(validated_blood_values, np.ndarray):
            format_values = validated_blood_values.flatten().tolist()
        else:
            try:
                format_values = [float(v) if v and isinstance(v, str) and v.strip() else 0.0 for v in blood_values]
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert blood_values to floats: {e}. Using zeros.")
                format_values = [0.0] * len(blood_values)

        explanation = "<h2>Interpretation</h2>"
        if submit_type == 'xray_upload':
            if xray_result['label'] == 'Healthy':
                explanation += f"""
                <p><strong>Healthy Diagnosis:</strong> The chest X-ray shows no signs of pneumonia (confidence: {xray_result['confidence']:.2%}).
                The lungs appear clear with no abnormal patterns such as consolidation or interstitial markings.</p>
                """
            elif xray_result['label'] == 'Bacterial':
                explanation += f"""
                <p><strong>Bacterial Pneumonia:</strong> The chest X-ray indicates bacterial pneumonia (confidence: {xray_result['confidence']:.2%}).
                Patterns such as consolidation or lobar infiltrates were detected, suggestive of bacterial infection.</p>
                """
            elif xray_result['label'] == 'Viral':
                explanation += f"""
                <p><strong>Viral Pneumonia:</strong> The chest X-ray suggests viral pneumonia (confidence: {xray_result['confidence']:.2%}).
                Interstitial patterns or diffuse haziness were observed, consistent with viral infection.</p>
                """
            else:
                explanation += "<p><strong>No Diagnosis:</strong> The X-ray analysis could not produce a valid result.</p>"

        elif submit_type == 'blood_test':
            if blood_result['label'] == 'Healthy':
                explanation += f"""
                <p><strong>Healthy Diagnosis:</strong> The blood test shows no signs of pneumonia (confidence: {blood_result['confidence']:.2%}).
                All biomarkers (e.g., neutrophils: {format_values[1]:.2f}, CRP: {format_values[3]:.2f}) are within normal ranges.</p>
                """
            elif blood_result['label'] == 'Bacterial':
                explanation += f"""
                <p><strong>Bacterial Pneumonia:</strong> The blood test indicates bacterial pneumonia (confidence: {blood_result['confidence']:.2%}).
                Elevated neutrophils ({format_values[1]:.2f} x10^9/L) and CRP ({format_values[3]:.2f} mg/L) suggest a bacterial infection.</p>
                """
            elif blood_result['label'] == 'Viral':
                explanation += f"""
                <p><strong>Viral Pneumonia:</strong> The blood test suggests viral pneumonia (confidence: {blood_result['confidence']:.2%}).
                Elevated lymphocytes ({format_values[2]:.2f} x10^9/L) indicate a viral infection.</p>
                """
            else:
                explanation += "<p><strong>No Diagnosis:</strong> The blood test analysis could not produce a valid result.</p>"

        elif submit_type == 'fusion_predict':
            if fusion_result['label'] == 'Healthy':
                explanation += f"""
                <p><strong>Healthy Diagnosis:</strong> Both the chest X-ray (confidence: {xray_result['confidence']:.2%}) and blood test
                (confidence: {blood_result['confidence']:.2%}) show no signs of pneumonia. The fusion analysis confirms a healthy status
                (confidence: {fusion_result['confidence']:.2%}). The X-ray shows clear lungs, and blood markers are normal.</p>
                """
            elif fusion_result['label'] == 'Bacterial':
                explanation += f"""
                <p><strong>Bacterial Pneumonia:</strong> The fusion analysis indicates bacterial pneumonia (confidence: {fusion_result['confidence']:.2%}).
                The X-ray shows consolidation (confidence: {xray_result['confidence']:.2%}), and the blood test reveals high neutrophils
                ({format_values[1]:.2f} x10^9/L) and CRP ({format_values[3]:.2f} mg/L, confidence: {blood_result['confidence']:.2%}).</p>
                """
            elif fusion_result['label'] == 'Viral':
                explanation += f"""
                <p><strong>Viral Pneumonia:</strong> The fusion analysis suggests viral pneumonia (confidence: {fusion_result['confidence']:.2%}).
                The X-ray shows interstitial patterns (confidence: {xray_result['confidence']:.2%}), and the blood test indicates elevated
                lymphocytes ({format_values[2]:.2f} x10^9/L, confidence: {blood_result['confidence']:.2%}).</p>
                """
            if xray_result['label'] != 'N/A' and blood_result['label'] != 'N/A' and xray_result['label'] != blood_result['label']:
                explanation += f"""
                <p><strong>Conflicting Results:</strong> The X-ray ({xray_result['label']}: {xray_result['confidence']:.2%}) and blood test
                ({blood_result['label']}: {blood_result['confidence']:.2%}) differ. The fusion analysis prioritizes the most confident result.</p>
                """

        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pneumonia Clinical Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                p, li {{ line-height: 1.6; }}
                .section {{ margin-bottom: 20px; }}
                img {{ max-width: 300px; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Pneumonia Clinical Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """

        report_html += """
            <div class="section">
                <h2>Input Summary</h2>
        """
        if submit_type in ['blood_test', 'fusion_predict']:
            report_html += "<p><strong>Blood Test Values:</strong></p><ul>"
            for key, value in zip([f[0] for f in FIELDS], blood_values):
                field = FIELDS[[f[0] for f in FIELDS].index(key)][1]
                report_html += f"""
                    <li>{field['label']}: {value or 'N/A'} {field['unit']} (Reference: {field['range']})</li>
                """
            report_html += "</ul>"
        if submit_type in ['xray_upload', 'fusion_predict']:
            report_html += f"""
                <p><strong>X-Ray Image:</strong> {os.path.basename(xray_result.get('heatmap_path', 'N/A')) if xray_result.get('heatmap_path') else 'N/A'}</p>
            """
        report_html += "</div>"

        if submit_type in ['xray_upload', 'fusion_predict']:
            report_html += f"""
                <div class="section">
                    <h2>Chest X-Ray Analysis</h2>
                    <p><strong>Prediction:</strong> {xray_result['label']} (Confidence: {xray_result['confidence'] * 100:.2f}%)</p>
                    <p><strong>Grad-CAM Heatmap:</strong> <img src="{heatmap_path}" alt="Grad-CAM Heatmap"></p>
                    <p><strong>Explanation:</strong> The X-ray was analyzed using a convolutional neural network. The heatmap highlights
                    areas of interest, such as consolidation (bacterial) or interstitial patterns (viral).</p>
                </div>
            """

        if submit_type in ['blood_test', 'fusion_predict']:
            report_html += """
                <div class="section">
                    <h2>Blood Test Analysis</h2>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                            <th>Reference Range</th>
                            <th>Status</th>
                        </tr>
            """
            for key, value in zip([f[0] for f in FIELDS], format_values):
                field = FIELDS[[f[0] for f in FIELDS].index(key)][1]
                status = 'Normal'
                try:
                    value_float = float(value) if value else 0
                    range_min, range_max = map(float, field['range'].split('-'))
                    if value_float < range_min:
                        status = 'Low'
                    elif value_float > range_max:
                        status = 'High'
                except:
                    status = 'N/A'
                report_html += f"""
                        <tr>
                            <td>{field['label']}</td>
                            <td>{value_float:.2f}</td>
                            <td>{field['range']}</td>
                            <td>{status}</td>
                        </tr>
                """
            report_html += f"""
                    </table>
                    <p><strong>Prediction:</strong> {blood_result['label']} (Confidence: {blood_result['confidence'] * 100:.2f}%)</p>
                    <p><strong>Explanation:</strong> A random forest model analyzed blood biomarkers to detect pneumonia.</p>
                    <p><strong>Feature Importance:</strong></p>
                    <ul>
            """
            for name, importance in blood_result.get('importances', {}).items():
                report_html += f"<li>{name}: {importance:.4f}</li>"
            report_html += "</ul></div>"

        if submit_type == 'fusion_predict':
            report_html += f"""
                <div class="section">
                    <h2>Fusion Analysis</h2>
                    <p><strong>Prediction:</strong> {fusion_result['label']} (Confidence: {fusion_result['confidence'] * 100:.2f}%)</p>
                    <p><strong>Explanation:</strong> The fusion model combines X-ray and blood test data to provide a comprehensive diagnosis.</p>
                </div>
            """

        report_html += f"""
            <div class="section">
                {explanation}
            </div>
        """

        report_html += """
            <div class="section">
                <h2>Recommendations</h2>
        """
        final_label = fusion_result['label'] if submit_type == 'fusion_predict' else \
                      blood_result['label'] if submit_type == 'blood_test' else xray_result['label']
        
        if final_label in ['Viral', 'Bacterial']:
            if final_label == 'Bacterial':
                report_html += """
                <p><strong>Bacterial Pneumonia Recommendations:</strong></p>
                <ul>
                    <li>Consult a pulmonologist within 24-48 hours.</li>
                    <li>Order sputum and blood cultures to identify the bacteria.</li>
                    <li>Start antibiotics (e.g., azithromycin) as prescribed.</li>
                    <li>Monitor oxygen levels and fever daily.</li>
                    <li>Seek hospitalization if symptoms worsen (e.g., SpO2 < 92%).</li>
                </ul>
                """
            else:
                report_html += """
                <p><strong>Viral Pneumonia Recommendations:</strong></p>
                <ul>
                    <li>Consult a pulmonologist within 48-72 hours.</li>
                    <li>Test for viral infections (e.g., influenza, RSV).</li>
                    <li>Consider antivirals (e.g., oseltamivir) if prescribed.</li>
                    <li>Stay hydrated and rest; use supportive care.</li>
                    <li>Watch for secondary bacterial infections.</li>
                </ul>
                """
        else:
            report_html += """
            <p><strong>Healthy Recommendations:</strong></p>
            <ul>
                <li>Schedule a check-up in 6 months.</li>
                <li>Monitor for cough, fever, or breathing issues.</li>
                <li>Maintain a healthy diet and exercise routine.</li>
            </ul>
            """
        report_html += """
            </div>
        </body>
        </html>
        """
        return report_html, explanation
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return None, f"Report generation failed: {str(e)}"

def validate_blood_values(blood_values, field_names=None):
    if field_names is None:
        field_names = [f[0] for f in FIELDS]
    validated_values = []
    error = None
    warnings = []
    
    try:
        if not isinstance(blood_values, (list, np.ndarray)) or len(blood_values) != 7:
            error = f"Expected 7 blood values, got {len(blood_values)}"
            logger.error(error)
            return None, error, []
        
        for value, field_name in zip(blood_values, field_names):
            field = next((f[1] for f in FIELDS if f[0] == field_name), None)
            if not field:
                error = f"Invalid field name: {field_name}"
                logger.error(error)
                return None, error, []
                
            try:
                value_float = float(value) if value and str(value).strip() else 0.0
                if value_float < 0:
                    error = f"Negative value for {field['label']}: {value_float}. Value cannot be negative."
                    logger.error(error)
                    return None, error, []
                    
                range_min, range_max = map(float, field['range'].split('-'))
                tolerance = 0.1
                min_tolerance = range_min * (1 - tolerance)
                max_tolerance = range_max * (1 + tolerance)
                
                if not (min_tolerance <= value_float <= max_tolerance):
                    value_float = np.clip(value_float, min_tolerance, max_tolerance)
                    warning = f"Value {value_float} for {field['label']} clipped to range [{min_tolerance:.2f}, {max_tolerance:.2f}]."
                    logger.warning(warning)
                    warnings.append(warning)
                    
                validated_values.append(value_float)
            except (ValueError, TypeError) as e:
                error = f"Invalid value for {field['label']}: {value}. Must be numeric."
                logger.error(error)
                return None, error, []
                
        validated_values = np.array(validated_values).reshape(1, -1)
        logger.info(f"Validated blood values: {validated_values}, shape: {validated_values.shape}")
        return validated_values, error, warnings
        
    except Exception as e:
        error = f"Validation failed: {str(e)}"
        logger.error(error)
        return None, error, []

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.debug(f"Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")

@app.route('/', methods=['GET', 'POST'])
def index():
    blood_test_prediction = None
    xray_result = None
    fusion_result = None
    image_path = None
    blood_test_results = {}  # Always initialize to avoid undefined error
    error = None
    chart_data = []
    report = None
    explanation = None
    warnings = []
    submit_type = None

    try:
        if request.method == 'POST':
            submit_type = request.form.get('submit_type')
            # Collect blood values
            blood_values = []
            for field in FIELDS:
                value = request.form.get(field[0], '0.0').strip()
                blood_values.append(value)
            logger.info(f"Received blood test values: {dict(zip([f[0] for f in FIELDS], blood_values))}")
            
            # Validate blood values
            blood_values_valid, validation_error, validation_warnings = validate_blood_values(blood_values)
            warnings.extend(validation_warnings)
            
            if validation_error:
                error = validation_error
                logger.error(f"Validation error: {error}")
            else:
                # Populate blood_test_results even if prediction fails
                blood_test_results = {f[1]['label']: v for f, v in zip(FIELDS, blood_values)}
                logger.debug(f"Validated blood_values: {blood_values_valid}, shape: {blood_values_valid.shape}")
            
            if submit_type == 'blood_test' and not validation_error:
                try:
                    blood_result = predict_blood_test(blood_values_valid)
                    if isinstance(blood_result, tuple):
                        error = blood_result[1]
                    else:
                        blood_test_prediction = blood_result['label']
                        chart_data.append({'label': 'Blood Test', 'confidence': blood_result['confidence'] * 100})
                except Exception as e:
                    logger.error(f"Error processing blood test: {str(e)}")
                    error = f"Blood test processing failed: {str(e)}"

            elif submit_type == 'xray_upload' and 'xray_image' in request.files:
                xray_file = request.files['xray_image']
                if xray_file and xray_file.filename and allowed_file(xray_file.filename):
                    try:
                        filename = f"{int(datetime.now().timestamp())}_{xray_file.filename}"
                        full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        xray_file.save(full_image_path)
                        logger.info(f"X-ray file saved to: {full_image_path}")
                        xray_result = predict_xray(full_image_path)
                        if isinstance(xray_result, tuple):
                            error = xray_result[1]
                        else:
                            image_path = os.path.join('uploads', filename).replace('\\', '/')
                            chart_data.append({'label': 'X-Ray', 'confidence': xray_result['confidence'] * 100})
                    except Exception as e:
                        logger.error(f"Error processing X-ray: {str(e)}")
                        error = f"X-ray processing failed: {str(e)}"
                else:
                    error = "Invalid or missing X-ray image. Please upload a JPG, JPEG, or PNG file."

            elif submit_type == 'fusion_predict' and 'xray_image' in request.files and not validation_error:
                xray_file = request.files['xray_image']
                if xray_file and xray_file.filename and allowed_file(xray_file.filename):
                    try:
                        filename = f"{int(datetime.now().timestamp())}_{xray_file.filename}"
                        full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        xray_file.save(full_image_path)
                        logger.info(f"X-ray file saved to: {full_image_path}")
                        image_path = os.path.join('uploads', filename).replace('\\', '/')
                        xray_result = predict_xray(full_image_path)
                        if isinstance(xray_result, tuple):
                            error = xray_result[1]
                        else:
                            blood_result = predict_blood_test(blood_values_valid)
                            if isinstance(blood_result, tuple):
                                logger.warning(f"Blood test prediction failed: {blood_result[1]}. Using X-ray result.")
                                blood_result = {'label': 'N/A', 'confidence': 0.0, 'importances': {}}
                            fusion_result = predict_fusion(full_image_path, blood_values_valid)
                            if isinstance(fusion_result, tuple):
                                error = fusion_result[1]
                            else:
                                chart_data = [
                                    {'label': 'X-Ray', 'confidence': xray_result['confidence'] * 100},
                                    {'label': 'Blood Test', 'confidence': blood_result['confidence'] * 100},
                                    {'label': 'Fusion', 'confidence': fusion_result['confidence'] * 100}
                                ]
                    except Exception as e:
                        logger.error(f"Error during fusion prediction: {str(e)}")
                        error = f"Fusion prediction failed: {str(e)}"
                else:
                    error = "Invalid or missing X-ray image. Please upload a JPG, JPEG, or PNG file."

            if not error and (
                (submit_type == 'xray_upload' and xray_result and isinstance(xray_result, dict)) or
                (submit_type == 'blood_test' and blood_test_prediction) or
                (submit_type == 'fusion_predict' and fusion_result and isinstance(fusion_result, dict))
            ):
                try:
                    report, explanation = create_report(
                        submit_type=submit_type,
                        xray_result=xray_result,
                        blood_result=blood_result if submit_type in ['blood_test', 'fusion_predict'] else None,
                        fusion_result=fusion_result,
                        blood_values=blood_values,
                        validated_blood_values=blood_values_valid
                    )
                    if report is None:
                        error = explanation
                except Exception as e:
                    logger.error(f"Error generating report: {str(e)}")
                    error = f"Report generation failed: {str(e)}"

            gc.collect()
            log_memory_usage()

        heatmap_path = None
        if xray_result and isinstance(xray_result, dict) and 'heatmap_path' in xray_result and xray_result['heatmap_path'] != 'N/A':
            heatmap_path = os.path.join('output_heatmaps', os.path.basename(xray_result['heatmap_path'])).replace('\\', '/')

        logger.debug(f"Rendering template with blood_test_results: {blood_test_results}")
        return render_template('index.html',
                               blood_test_prediction=blood_test_prediction,
                               xray_result=xray_result if isinstance(xray_result, dict) else None,
                               fusion_result=fusion_result if isinstance(fusion_result, dict) else None,
                               image_path=image_path,
                               heatmap_path=heatmap_path,
                               fields=FIELDS,
                               blood_test_results=blood_test_results,
                               error=error,
                               chart_data=chart_data,
                               report=report,
                               explanation=explanation,
                               warnings=warnings)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return render_template('index.html',
                               error=f"Server error: {str(e)}",
                               fields=FIELDS,
                               blood_test_results=blood_test_results)

@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        report_html = request.form.get('report_html')
        if not report_html:
            logger.error("No report HTML provided for download")
            return "No report available", 400
        
        pdf_io = io.BytesIO()
        HTML(string=report_html).write_pdf(pdf_io)
        pdf_io.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return send_file(
            pdf_io,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'pneumonia_report_{timestamp}.pdf'
        )
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return f"Error generating report: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)