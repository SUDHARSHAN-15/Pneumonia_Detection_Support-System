# 🫁 Pneumo Detection Support Sytem
## AI-Powered Pneumonia Detection & Clinical Decision Support System

PneumoVision AI is a deep learning-powered medical imaging system designed to assist in the early detection of pneumonia using chest X-ray images combined with clinical metadata.

The system integrates a Convolutional Neural Network (CNN) for image feature extraction and an XGBoost classifier for enhanced prediction performance. It also incorporates Grad-CAM visualizations to provide interpretability, enabling clinicians to understand which lung regions influenced the diagnosis.

Built using Python, TensorFlow, Flask, and XGBoost, this project demonstrates applied AI in healthcare diagnostics with explainable AI integration.

---

# 🏥 Problem Statement

Pneumonia is a life-threatening respiratory infection that:

- Affects millions globally each year
- Requires early diagnosis for effective treatment
- Often relies on expert radiologists for interpretation

Challenges in traditional diagnosis:

- Manual X-ray interpretation is time-consuming
- Human error and fatigue may affect accuracy
- Limited access to specialists in rural areas

PneumoVision AI aims to assist clinicians by providing:

- Fast automated screening
- High-accuracy classification
- Visual explanations for model decisions

---

# 🎯 Objectives

- Develop an AI-based system for pneumonia detection from chest X-rays
- Improve diagnostic accuracy using hybrid modeling (CNN + XGBoost)
- Provide explainable predictions using Grad-CAM
- Deploy the model through a web-based Flask application
- Enable real-time diagnostic assistance

---

# 🏗️ System Architecture

Chest X-Ray Image
↓
Image Preprocessing & Resizing
↓
CNN Feature Extraction
↓
Feature Vector Output
↓
XGBoost Classifier
↓
Prediction (Pneumonia / Normal)
↓
Grad-CAM Visualization
↓
Flask Web Application Interface


---

# 📂 Dataset

The model is trained on publicly available chest X-ray datasets containing:

- Pneumonia-positive cases
- Normal lung X-ray images

Dataset includes:

- Pediatric and adult cases
- Thousands of labeled images
- Balanced training and validation splits

Preprocessing steps:

- Image resizing
- Pixel normalization
- Data augmentation
- Noise reduction

---

# 🧠 Deep Learning Model

## Convolutional Neural Network (CNN)

The CNN model is responsible for:

- Extracting spatial features from X-ray images
- Learning texture and pattern representations
- Identifying lung opacity regions

### CNN Architecture

- Convolutional layers
- ReLU activation
- MaxPooling layers
- Flatten layer
- Feature embedding output

---

## Hybrid Classification Model (CNN + XGBoost)

Instead of using a standard dense layer for classification, this project uses:

- CNN for feature extraction
- XGBoost for final classification

### Why Hybrid Approach?

- Better structured feature handling
- Improved generalization
- Enhanced classification performance
- Reduced overfitting risk

---

# 📊 Model Performance

Performance Metrics:

- Accuracy: ~94%
- Precision
- Recall
- F1-Score
- Confusion Matrix analysis

Optimization Techniques:

- Dropout layers to reduce overfitting
- Adam optimizer
- Learning rate tuning
- Early stopping

---

# 🔍 Explainable AI (Grad-CAM Integration)

Medical AI must be interpretable.

This project integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to:

- Highlight infected lung regions
- Visualize heatmaps over X-ray images
- Improve clinical trust
- Provide diagnostic transparency

Grad-CAM helps answer:

"Why did the model predict pneumonia?"

---

# 🌐 Web Application (Flask)

The system is deployed as a Flask web application with:

- Image upload interface
- Real-time prediction display
- Probability score output
- Grad-CAM visualization overlay
- Clean and responsive UI

---

# 🧪 Workflow

1️⃣ Upload Chest X-ray  
2️⃣ Image Preprocessing  
3️⃣ CNN Feature Extraction  
4️⃣ XGBoost Classification  
5️⃣ Grad-CAM Heatmap Generation  
6️⃣ Display Final Diagnosis  

---

# 🧰 Tech Stack

## Programming
- Python 3.x

## Deep Learning
- TensorFlow
- Keras (CNN)

## Machine Learning
- XGBoost
- Scikit-learn

## Explainability
- Grad-CAM

## Backend
- Flask

## Visualization
- Matplotlib
- OpenCV

## Tools
- Git
- Jupyter Notebook

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

git clone https://github.com/SUDHARSHAN-15/Pneumonia_Detection_Support-System.git

cd PneumoVision-AI


## 2️⃣ Install Dependencies

pip install -r requirements.txt


## 3️⃣ Train Model

python train_model.py


## 4️⃣ Run Web Application

python app.py


---

# 🚀 Future Enhancements

- Multi-disease detection (COVID-19, Tuberculosis)
- Integration with hospital management systems
- Cloud deployment (AWS / Azure)
- Real-time hospital dashboard
- Model retraining pipeline
- Integration with electronic medical records (EMR)

---

# ⚠️ Disclaimer

This system is designed for educational and research purposes. It is not intended to replace professional medical diagnosis. Clinical decisions should always involve qualified healthcare professionals.

---

# 🌍 Real-World Impact

PneumoVision AI can support:

- Rural healthcare centers
- Emergency screening
- Radiology assistance
- Faster clinical workflow
- AI-assisted diagnostic decision-making

---

# 👤 Author

**Sudharshan M**  
B.Tech – Artificial Intelligence & Data Science  
Email: sudharshan1504@gmail.com  

---

# ⭐ If you found this project valuable, consider giving it a star.