import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, Dropout, Concatenate,
    BatchNormalization, LeakyReLU, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from logging import handlers
import psutil
import gc
from datetime import datetime
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = handlers.RotatingFileHandler('training.log', maxBytes=1000000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

logger.info(f"TensorFlow version: {tf.__version__}")

# Force CPU usage due to limited GPU memory (MX350 2GB)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
logger.info("Forcing CPU usage for training due to limited GPU memory")

# Use float32 to avoid mixed precision issues
tf.keras.mixed_precision.set_global_policy('float32')

def focal_loss(gamma=1.5, alpha=0.25):
    """Focal loss for imbalanced datasets."""
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)
    return focal_loss_fn

class DataGenerator(Sequence):
    """Custom data generator for paired X-ray and blood test data."""
    def __init__(self, image_paths, blood_data, labels, batch_size=2, image_size=(128, 128), shuffle=True):
        self.image_paths = image_paths
        self.blood_data = blood_data
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        self.datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indices = self.indices[start_idx:end_idx]
        
        batch_xray = []
        batch_blood = self.blood_data[batch_indices]
        batch_labels = tf.keras.utils.to_categorical(self.labels[batch_indices], num_classes=3)
        
        for idx in batch_indices:
            try:
                img = load_img(self.image_paths[idx], target_size=self.image_size, color_mode='grayscale')
                img_array = img_to_array(img) / 255.0
                if self.shuffle:
                    img_array = self.datagen.random_transform(img_array)
                batch_xray.append(img_array)
            except Exception as e:
                logger.error(f"Error loading image {self.image_paths[idx]}: {str(e)}")
                batch_xray.append(np.zeros((self.image_size[0], self.image_size[1], 1)))
        
        return [np.array(batch_xray), batch_blood], batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __del__(self):
        """Ensure resources are released."""
        self.image_paths = None
        self.blood_data = None
        self.labels = None
        self.indices = None
        gc.collect()

def validate_blood_values(blood_values, field_names=None):
    """Validate blood test values against reference ranges."""
    FIELDS = [
        ('wbc', {'label': 'White Blood Cell (WBC) Count', 'range': '4.0-11.0', 'unit': 'x10^9/L'}),
        ('neutrophils', {'label': 'Neutrophils', 'range': '2.0-7.5', 'unit': 'x10^9/L'}),
        ('lymphocytes', {'label': 'Lymphocytes', 'range': '1.0-3.5', 'unit': 'x10^9/L'}),
        ('crp', {'label': 'C-Reactive Protein (CRP)', 'range': '0.0-10.0', 'unit': 'mg/L'}),
        ('esr', {'label': 'Erythrocyte Sedimentation Rate (ESR)', 'range': '0.0-20.0', 'unit': 'mm/hr'}),
        ('platelets', {'label': 'Platelets', 'range': '150.0-450.0', 'unit': 'x10^9/L'}),
        ('hemoglobin', {'label': 'Hemoglobin', 'range': '12.0-16.0', 'unit': 'g/dL'}),
    ]
    
    validated_values = []
    error = None
    warnings = []
    
    if field_names is None:
        field_names = [f[0] for f in FIELDS]
    
    blood_values = np.array(blood_values)
    if blood_values.ndim == 1:
        blood_values = blood_values.reshape(1, -1)
    
    for row in blood_values:
        row_validated = []
        for value, field_name in zip(row, field_names):
            field = next((f[1] for f in FIELDS if f[0] == field_name), None)
            try:
                value_float = float(value)
                if value_float < 0:
                    error = f"Negative value for {field['label']}: {value_float}. Value cannot be negative."
                    logger.error(error)
                    return None, error, []
                range_min, range_max = map(float, field['range'].split('-'))
                tolerance = 0.1
                min_tolerance = range_min * (1 - tolerance)
                max_tolerance = range_max * (1 + tolerance)
                value_float = np.clip(value_float, min_tolerance, max_tolerance)
                if not (min_tolerance <= value_float <= max_tolerance):
                    warning = f"Value {value_float} for {field['label']} clipped to range [{min_tolerance}, {max_tolerance}]."
                    logger.warning(warning)
                    warnings.append(warning)
                row_validated.append(value_float)
            except (ValueError, TypeError):
                error = f"Invalid value for {field['label']}: {value}. Must be numeric."
                logger.error(error)
                return None, error, []
        validated_values.append(row_validated)
    
    validated_values = np.array(validated_values)
    if validated_values.ndim == 1:
        validated_values = validated_values.reshape(1, -1)
    
    logger.info(f"Validated blood values shape: {validated_values.shape}")
    return validated_values, error, warnings

def load_and_preprocess_data(
    image_dir,
    blood_data_path,
    scaler_save_path=r'C:\Users\HP\Downloads\FinalYearProject-master\FinalYearProject-master\models\pneumonia_scaler.pkl'
):
    """Load and preprocess paired X-ray images and blood test data."""
    try:
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(blood_data_path):
            raise FileNotFoundError(f"Blood data file not found: {blood_data_path}")
        
        blood_df = pd.read_csv(blood_data_path)
        logger.info(f"Blood test data loaded. Shape: {blood_df.shape}")
        logger.info(f"Sample blood test data (first row): \n{blood_df.head(1)}")
        
        column_mapping = {
            'wbc': ['wbc', 'WBC', 'white_blood_cell_count', 'white_blood_cells'],
            'neutrophils': ['neutrophils', 'neutrophil_count', 'neut'],
            'lymphocytes': ['lymphocytes', 'lymphocyte', 'lymph'],
            'crp': ['crp', 'c_reactive_protein', 'CRP'],
            'esr': ['esr', 'ESR', 'erythrocyte_sedimentation_rate'],
            'platelets': ['platelets', 'platelet_count', 'plt'],
            'hemoglobin': ['hemoglobin', 'hgb', 'hb'],
            'target': ['target', 'label', 'class', 'diagnosis']
        }
        
        actual_columns = blood_df.columns.tolist()
        logger.info(f"Actual CSV columns: {actual_columns}")
        mapped_columns = {}
        for expected_col, aliases in column_mapping.items():
            for alias in aliases:
                if alias in actual_columns:
                    mapped_columns[alias] = expected_col
                    break
            else:
                raise ValueError(f"Could not find column for '{expected_col}' in CSV. Available columns: {actual_columns}")
        
        blood_df = blood_df.rename(columns=mapped_columns)
        
        required_columns = ['wbc', 'neutrophils', 'lymphocytes', 'crp', 'esr', 'platelets', 'hemoglobin', 'target']
        missing_cols = [col for col in required_columns if col not in blood_df.columns]
        if missing_cols:
            raise ValueError(f"Blood data missing columns: {missing_cols}")
        
        blood_df = blood_df.dropna(subset=required_columns)
        logger.info(f"After dropping NA, blood data shape: {blood_df.shape}")
        
        X_blood = blood_df[required_columns[:-1]].values
        X_blood, error, warnings = validate_blood_values(X_blood, required_columns[:-1])
        if error:
            raise ValueError(error)
        for warning in warnings:
            logger.warning(warning)
        
        scaler = StandardScaler()
        X_blood = scaler.fit_transform(X_blood)
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        logger.info(f"Blood test scaler saved at: {scaler_save_path}")
        
        label_map = {'Healthy': 0, 'Viral': 1, 'Bacterial': 2, 'Pneumonia': 1}
        y_blood = blood_df['target'].str.capitalize().map(label_map)
        invalid_labels = y_blood.isna()
        if invalid_labels.any():
            logger.error(f"Found {invalid_labels.sum()} invalid labels: {blood_df[invalid_labels]['target'].unique()}")
            raise ValueError(f"Invalid or missing labels in blood test data.")
        y_blood = y_blood.values
        
        image_paths = []
        labels = []
        valid_extensions = ('.jpg', '.jpeg', '.png')
        
        label_dirs = {
            'NORMAL': 0,
            'PNEUMONIA_VIRAL': 1,
            'PNEUMONIA_BACTERIAL': 2
        }
        found_dirs = []
        
        for label, class_id in label_dirs.items():
            label_dir = os.path.join(image_dir, label)
            if os.path.exists(label_dir):
                found_dirs.append(label)
                for filename in os.listdir(label_dir):
                    if filename.lower().endswith(valid_extensions):
                        image_paths.append(os.path.join(label_dir, filename))
                        labels.append(class_id)
            else:
                logger.warning(f"Directory not found: {label_dir}")
        
        if not found_dirs:
            label_dir = os.path.join(image_dir, 'PNEUMONIA')
            if os.path.exists(label_dir):
                logger.info("Falling back to binary classification (NORMAL vs PNEUMONIA)")
                for filename in os.listdir(label_dir):
                    if filename.lower().endswith(valid_extensions):
                        image_paths.append(os.path.join(label_dir, filename))
                        labels.append(1)
                normal_dir = os.path.join(image_dir, 'NORMAL')
                if os.path.exists(normal_dir):
                    for filename in os.listdir(normal_dir):
                        if filename.lower().endswith(valid_extensions):
                            image_paths.append(os.path.join(normal_dir, filename))
                            labels.append(0)
                else:
                    raise FileNotFoundError(f"NORMAL directory not found in {image_dir}")
            else:
                raise FileNotFoundError(f"No valid X-ray directories found in {image_dir}")
        
        image_paths = np.array(image_paths)
        y_xray = np.array(labels)
        logger.info(f"X-ray images found: {len(image_paths)}, Labels: {len(labels)}, Distribution: {np.bincount(y_xray)}")
        
        min_samples = min(len(image_paths), len(X_blood))
        if min_samples == 0:
            raise ValueError("No valid data samples found.")
        
        indices = np.random.permutation(min_samples)
        image_paths = image_paths[indices[:min_samples]]
        X_blood = X_blood[indices[:min_samples]]
        y_xray = y_xray[indices[:min_samples]]
        y_blood = y_blood[indices[:min_samples]]
        
        if not np.array_equal(y_xray, y_blood):
            logger.warning(f"Label mismatch detected. Using blood test labels.")
            y = y_blood
        else:
            y = y_blood
        
        logger.info(f"Paired data. Final samples: {min_samples}, Label distribution: {np.bincount(y)}")
        
        class_weights = {}
        for cls in range(3):
            count = np.bincount(y)[cls] if cls < len(np.bincount(y)) else 1
            class_weights[cls] = (1 / count) * (len(y) / 3.0)
        logger.info(f"Class weights: {class_weights}")
        
        train_idx, val_idx = train_test_split(
            np.arange(min_samples),
            test_size=0.3,  # Increased to 30% for better validation
            random_state=42,
            stratify=y
        )
        
        train_generator = DataGenerator(
            image_paths[train_idx], X_blood[train_idx], y[train_idx],
            batch_size=2, image_size=(128, 128), shuffle=True
        )
        val_generator = DataGenerator(
            image_paths[val_idx], X_blood[val_idx], y[val_idx],
            batch_size=2, image_size=(128, 128), shuffle=False
        )
        
        logger.info(f"Training samples: {len(train_generator)} batches, Validation samples: {len(val_generator)} batches")
        
        return train_generator, val_generator, class_weights
    
    except Exception as e:
        logger.error(f"Error loading/preprocessing data: {str(e)}")
        raise

def create_fusion_model(image_shape=(128, 128, 1), blood_input_shape=(7,), num_classes=3):
    """Create a lightweight fusion model for low-memory systems."""
    try:
        # Image branch
        image_input = Input(shape=image_shape, name='xray_input')
        x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(image_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2))(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.3)(x)
        xray_features = x
        
        # Blood test branch
        blood_input = Input(shape=blood_input_shape, name='blood_input')
        b = Dense(32, kernel_initializer='he_normal')(blood_input)
        b = BatchNormalization()(b)
        b = LeakyReLU(alpha=0.1)(b)
        b = Dense(16, kernel_initializer='he_normal')(b)
        b = LeakyReLU(alpha=0.1)(b)
        b = Dropout(0.3)(b)
        blood_features = b
        
        # Fusion
        fusion = Concatenate()([xray_features, blood_features])
        f = Dense(32, kernel_initializer='he_normal')(fusion)
        f = BatchNormalization()(f)
        f = LeakyReLU(alpha=0.1)(f)
        f = Dropout(0.3)(f)
        output = Dense(num_classes, activation='softmax', name='output')(f)
        
        model = Model(inputs=[image_input, blood_input], outputs=output)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=focal_loss(gamma=1.5, alpha=0.25),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        model.summary(print_fn=lambda x: logger.info(x))
        
        model_save_path = r'C:\Users\HP\Downloads\FinalYearProject-master\FinalYearProject-master\models\fusion_model2.h5'
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
        
        return model, [checkpoint, reduce_lr, early_stopping]
    
    except Exception as e:
        logger.error(f"Error creating fusion model: {str(e)}")
        raise

def log_memory_usage():
    """Log system memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory Usage: {mem_info.rss / 1024**2:.2f} MB")

def main():
    """Main training function."""
    dataset_dir = r"C:\Users\HP\Downloads\pneumonia dataset\chest_xray\train"
    blood_data_path = r"C:\Users\HP\Downloads\pneumonia_blood_data_large.csv"
    scaler_save_path = r"C:\Users\HP\Downloads\FinalYearProject-master\FinalYearProject-master\models\pneumonia_scaler.pkl"
    
    try:
        log_memory_usage()
        train_generator, val_generator, class_weights = load_and_preprocess_data(
            dataset_dir, blood_data_path, scaler_save_path
        )
        
        model, callbacks = create_fusion_model()
        
        history = model.fit(
            train_generator,
            epochs=20,  # Increased for better convergence
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
            workers=1,  # Reduced to avoid iterator issues
            use_multiprocessing=False
        )
        
        # Log validation predictions
        logger.info("Generating validation predictions...")
        val_predictions = []
        val_labels = []
        for i in range(len(val_generator)):
            (xray_batch, blood_batch), labels = val_generator[i]
            batch_pred = model.predict([xray_batch, blood_batch], verbose=0)
            val_predictions.append(batch_pred)
            val_labels.append(labels)
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        pred_classes = np.argmax(val_predictions, axis=1)
        true_classes = np.argmax(val_labels, axis=1)
        logger.info(f"Validation prediction accuracy: {np.mean(pred_classes == true_classes):.4f}")
        for cls in range(3):
            cls_mask = true_classes == cls
            if cls_mask.sum() > 0:
                logger.info(f"Class {cls} (Healthy:0, Viral:1, Bacterial:2) accuracy: {np.mean(pred_classes[cls_mask] == true_classes[cls_mask]):.4f}")
        
        model.save(r'C:\Users\HP\Downloads\FinalYearProject-master\FinalYearProject-master\models\fusion_model2.h5')
        logger.info("Training completed successfully")
        
        # Clean up generators
        del train_generator
        del val_generator
        gc.collect()
        log_memory_usage()
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()