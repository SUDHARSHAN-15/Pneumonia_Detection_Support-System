import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import stat
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', r'C:\Users\HP\Downloads\pneumonia_blood_data_large.csv')
MODEL_DIR = os.path.join(BASE_DIR, r'C:\Users\HP\Downloads\FinalYearProject-master\FinalYearProject-master\models')
OUTPUT_DIR = os.path.join(BASE_DIR, r'C:\Users\HP\Downloads\FinalYearProject-master\FinalYearProject-master\outputs')
RANDOM_STATE = 42

# Feature order (must match FIELDS in app.py)
FEATURES = ['wbc', 'neutrophils', 'lymphocytes', 'crp', 'esr', 'platelets', 'hemoglobin']

# Reference ranges for augmentation (aligned with app.py FIELDS)
REFERENCE_RANGES = {
    'wbc': (4.0, 11.0),
    'neutrophils': (2.0, 7.5),
    'lymphocytes': (1.0, 3.5),
    'crp': (0.0, 10.0),
    'esr': (0.0, 20.0),
    'platelets': (150.0, 450.0),
    'hemoglobin': (12.0, 16.0)
}

def ensure_directory_writable(directory):
    """Ensure the directory is writable by setting permissions."""
    try:
        os.makedirs(directory, exist_ok=True)
        # Set read/write/execute permissions for user, group, others
        os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        logger.info(f"Directory {directory} is writable")
    except PermissionError as e:
        logger.error(f"Permission denied for {directory}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to set permissions for {directory}: {str(e)}")
        raise

def setup_directories():
    """Create model and output directories if they don't exist."""
    try:
        ensure_directory_writable(MODEL_DIR)
        ensure_directory_writable(OUTPUT_DIR)
        logger.info(f"Model directory: {MODEL_DIR}")
        logger.info(f"Output directory: {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Error setting up directories: {str(e)}")
        raise RuntimeError(f"Failed to create directories: {str(e)}")

def load_data(file_path):
    """Load and validate the dataset."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}. Please ensure the file exists.")
        df = pd.read_csv(file_path, on_bad_lines='warn')
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        if 'target' not in df.columns:
            raise ValueError("Dataset must contain a 'target' column")
        if not all(feat in df.columns for feat in FEATURES):
            raise ValueError(f"Dataset must contain features: {FEATURES}")
        
        # Check for non-numeric values
        for feat in FEATURES:
            if not pd.api.types.is_numeric_dtype(df[feat]):
                raise ValueError(f"Feature {feat} contains non-numeric values")
            min_val, max_val = df[feat].min(), df[feat].max()
            logger.info(f"{feat} range: {min_val:.2f} - {max_val:.2f}")
        
        # Log unique target classes
        unique_targets = df['target'].unique()
        logger.info(f"Unique target classes: {unique_targets}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise RuntimeError(f"Failed to load data: {str(e)}")

def augment_data(df):
    """Augment dataset with synthetic out-of-range samples, balancing target distribution."""
    try:
        n_samples = int(0.1 * len(df))  # Add 10% synthetic samples
        synthetic_data = []
        np.random.seed(RANDOM_STATE)
        
        # Get target class distribution
        target_counts = df['target'].value_counts(normalize=True)
        logger.info(f"Target distribution: {target_counts.to_dict()}")
        
        for _ in range(n_samples):
            sample = {}
            # Sample target based on original distribution
            target = np.random.choice(target_counts.index, p=target_counts.values)
            for feat, (min_val, max_val) in REFERENCE_RANGES.items():
                # 50% chance for out-of-range value
                if np.random.random() < 0.5:
                    # Extend range by 50% below min and 100% above max
                    if np.random.random() < 0.5:
                        # Below min
                        value = np.random.uniform(min_val * 0.5, min_val)
                    else:
                        # Above max
                        value = np.random.uniform(max_val, max_val * 2.0)
                else:
                    # Within range
                    value = np.random.uniform(min_val, max_val)
                sample[feat] = value
            sample['target'] = target
            synthetic_data.append(sample)
        
        synthetic_df = pd.DataFrame(synthetic_data)
        augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
        logger.info(f"Augmented data with {n_samples} synthetic samples. New shape: {augmented_df.shape}")
        
        # Log synthetic data ranges
        for feat in REFERENCE_RANGES:
            min_val, max_val = synthetic_df[feat].min(), synthetic_df[feat].max()
            logger.info(f"Synthetic {feat} range: {min_val:.2f} - {max_val:.2f}")
        
        # Log new target distribution
        new_target_counts = augmented_df['target'].value_counts(normalize=True)
        logger.info(f"Augmented target distribution: {new_target_counts.to_dict()}")
        
        return augmented_df
    except Exception as e:
        logger.error(f"Error augmenting data: {str(e)}")
        raise RuntimeError(f"Failed to augment data: {str(e)}")

def preprocess_data(df):
    """Preprocess the dataset."""
    try:
        initial_rows = df.shape[0]
        df = df.dropna(subset=FEATURES + ['target'])
        logger.info(f"Dropped {initial_rows - df.shape[0]} rows with missing values. New shape: {df.shape}")

        # Clip extreme values to prevent scaling issues
        for feat, (min_val, max_val) in REFERENCE_RANGES.items():
            df[feat] = df[feat].clip(lower=min_val * 0.5, upper=max_val * 2.0)
        
        label_encoder = LabelEncoder()
        df['target'] = label_encoder.fit_transform(df['target'])
        logger.info(f"Target classes: {list(label_encoder.classes_)}")

        X = df[FEATURES]
        y = df['target']
        logger.info(f"Features (in order): {list(X.columns)}")
        return X, y, label_encoder
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise RuntimeError(f"Failed to preprocess data: {str(e)}")

def split_and_scale_data(X, y):
    """Split and scale the data."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Features scaled successfully")
        return X_train, X_test, y_train, y_test, scaler
    except Exception as e:
        logger.error(f"Error splitting/scaling data: {str(e)}")
        raise RuntimeError(f"Failed to split/scale data: {str(e)}")

def train_model(X_train, y_train, n_jobs=1):
    """Train the RandomForestClassifier with hyperparameter tuning."""
    try:
        model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=n_jobs, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return best_model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise RuntimeError(f"Failed to train model: {str(e)}")

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the model and generate plots."""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        n_classes = len(label_encoder.classes_)
        logger.info(f"Number of classes: {n_classes}")
        
        # Handle binary vs. multiclass ROC-AUC
        if n_classes == 2:
            y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = y_pred_proba  # Use all probabilities for multiclass
        
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Accuracy Score: {accuracy:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # ROC-AUC (binary or multiclass)
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {cm_path}")
        
        if n_classes == 2:
            # ROC curve (binary only)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            roc_path = os.path.join(OUTPUT_DIR, f'roc_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(roc_path)
            plt.close()
            logger.info(f"ROC curve saved to {roc_path}")
            
            # Precision-recall curve (binary only)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label='Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')
            pr_path = os.path.join(OUTPUT_DIR, f'precision_recall_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(pr_path)
            plt.close()
            logger.info(f"Precision-Recall curve saved to {pr_path}")
        else:
            logger.warning("ROC and Precision-Recall curves are not generated for multiclass classification")
        
        return accuracy, roc_auc
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise RuntimeError(f"Failed to evaluate model: {str(e)}")

def evaluate_out_of_range(model, scaler, label_encoder):
    """Evaluate model on synthetic out-of-range data."""
    try:
        n_samples = 100
        synthetic_data = []
        np.random.seed(RANDOM_STATE)
        
        # Use same target distribution as training data
        target_counts = pd.Series(label_encoder.classes_).value_counts(normalize=True)
        
        for _ in range(n_samples):
            sample = {}
            target = np.random.choice(label_encoder.classes_, p=target_counts.values)
            for feat, (min_val, max_val) in REFERENCE_RANGES.items():
                if np.random.random() < 0.7:  # 70% chance for out-of-range
                    if np.random.random() < 0.5:
                        value = np.random.uniform(min_val * 0.5, min_val)
                    else:
                        value = np.random.uniform(max_val, max_val * 2.0)
                else:
                    value = np.random.uniform(min_val, max_val)
                sample[feat] = value
            sample['target'] = target
            synthetic_data.append(sample)
        
        synthetic_df = pd.DataFrame(synthetic_data)
        X_synthetic = synthetic_df[FEATURES]
        y_synthetic = label_encoder.transform(synthetic_df['target'])
        
        X_synthetic_scaled = scaler.transform(X_synthetic)
        y_pred = model.predict(X_synthetic_scaled)
        
        accuracy = accuracy_score(y_synthetic, y_pred)
        logger.info(f"Out-of-range Accuracy: {accuracy:.4f}")
        
        if len(label_encoder.classes_) == 2:
            y_pred_proba = model.predict_proba(X_synthetic_scaled)[:, 1]
            roc_auc = roc_auc_score(y_synthetic, y_pred_proba)
        else:
            y_pred_proba = model.predict_proba(X_synthetic_scaled)
            roc_auc = roc_auc_score(y_synthetic, y_pred_proba, multi_class='ovr')
        logger.info(f"Out-of-range ROC-AUC: {roc_auc:.4f}")
        
        logger.info("Out-of-range Classification Report:")
        logger.info("\n" + classification_report(y_synthetic, y_pred, target_names=label_encoder.classes_))
        
        # Log sample predictions
        for i in range(min(5, n_samples)):
            logger.info(f"Sample {i+1}: Features: {X_synthetic.iloc[i].to_dict()}, "
                        f"True: {label_encoder.classes_[y_synthetic[i]]}, Predicted: {label_encoder.classes_[y_pred[i]]}")
    except Exception as e:
        logger.error(f"Error evaluating out-of-range data: {str(e)}")
        raise RuntimeError(f"Failed to evaluate out-of-range data: {str(e)}")

def plot_feature_importance(model, feature_names):
    """Plot and log feature importance."""
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        fi_path = os.path.join(OUTPUT_DIR, f'feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.tight_layout()
        plt.savefig(fi_path)
        plt.close()
        logger.info(f"Feature importance plot saved to {fi_path}")
        
        logger.info("Feature Importance:")
        for i in indices:
            logger.info(f"{feature_names[i]}: {importances[i]:.4f}")
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise RuntimeError(f"Failed to plot feature importance: {str(e)}")

def save_artifacts(model, scaler, label_encoder):
    """Save model, scaler, and label encoder."""
    try:
        model_path = os.path.join(MODEL_DIR, 'pneumonia_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'pneumonia_scaler.pkl')
        encoder_path = os.path.join(MODEL_DIR, 'pneumonia_label_encoder.pkl')
        
        # Check write permissions
        for path in [model_path, scaler_path, encoder_path]:
            dir_path = os.path.dirname(path)
            if not os.access(dir_path, os.W_OK):
                raise PermissionError(f"No write permission for directory {dir_path}")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(label_encoder, encoder_path)
        
        # Verify file permissions
        for path in [model_path, scaler_path, encoder_path]:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Label encoder saved to {encoder_path}")
    except Exception as e:
        logger.error(f"Error saving artifacts: {str(e)}")
        raise RuntimeError(f"Failed to save artifacts: {str(e)}")

def main():
    """Main function to orchestrate the pipeline."""
    try:
        setup_directories()
        
        df = load_data(DATA_PATH)
        
        # Augment data with out-of-range samples
        df = augment_data(df)
        
        X, y, label_encoder = preprocess_data(df)
        
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
        
        model = train_model(X_train, y_train, n_jobs=-1)
        
        accuracy, roc_auc = evaluate_model(model, X_test, y_test, label_encoder)
        
        # Evaluate on out-of-range data
        evaluate_out_of_range(model, scaler, label_encoder)
        
        plot_feature_importance(model, X.columns)
        
        save_artifacts(model, scaler, label_encoder)
        
        logger.info("Training pipeline completed successfully")
        logger.info(f"Final Accuracy: {accuracy:.4f}")
        logger.info(f"Final ROC-AUC: {roc_auc:.4f}")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()