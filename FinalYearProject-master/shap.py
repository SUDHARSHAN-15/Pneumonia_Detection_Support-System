import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set GPU memory allocation strategy
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def build_cnn_model(input_shape=(500, 500, 1)):
    """Build a simplified CNN model for binary classification."""
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_last_conv_layer_name(model):
    """Get the name of the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

def compute_gradcam(model, image_array, last_conv_layer_name):
    """Compute the Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        class_idx = tf.cast(predictions[0] > 0.5, tf.int32)
        class_output = predictions[:, 0]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10
    return heatmap

def overlay_heatmap_on_image(image_path, heatmap, output_dir, alpha=0.5):
    """Overlay the heatmap on the original image and save it with a unique filename."""
    try:
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)

        # Create output directory if it doesn't exist
        logger.debug(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename using timestamp
        base_filename = os.path.basename(image_path)
        unique_filename = f"gradcam_{int(time.time())}_{base_filename}"
        output_file = os.path.join(output_dir, unique_filename)
        logger.debug(f"Saving heatmap to: {output_file}")

        # Save the heatmap
        success = cv2.imwrite(output_file, overlay)
        if not success:
            raise Exception(f"Failed to save heatmap image: {output_file}")

        logger.debug(f"✅ Heatmap saved to: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error in overlay_heatmap_on_image: {str(e)}")
        raise

def process_single_image(image_path, model_path, output_dir, target_size=(500, 500), color_mode='grayscale'):
    """Process a single image for prediction and Grad-CAM visualization."""
    try:
        logger.debug(f"Loading model from: {model_path}")
        model = load_model(model_path)
        last_conv = get_last_conv_layer_name(model)

        logger.debug(f"Loading image: {image_path}")
        img = load_img(image_path, target_size=target_size, color_mode=color_mode)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        logger.debug("Predicting...")
        prediction = model.predict(img_array, verbose=0)[0][0]
        label = 'Pneumonia' if prediction >= 0.5 else 'Normal'
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        logger.debug("Computing Grad-CAM...")
        heatmap = compute_gradcam(model, img_array, last_conv)

        logger.debug("Overlaying heatmap...")
        heatmap_path = overlay_heatmap_on_image(image_path, heatmap, output_dir)

        return {
            'label': label,
            'confidence': float(confidence),
            'heatmap_path': heatmap_path
        }

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        raise

def process_images_in_directory(model, input_dir, output_dir, target_size=(500, 500), color_mode='grayscale'):
    """Process all images in the input directory and subdirectories for Grad-CAM."""
    valid_extensions = ('.jpg', '.jpeg', '.png')
    try:
        logger.debug("Getting last convolutional layer...")
        last_conv = get_last_conv_layer_name(model)
        logger.debug(f"Last conv layer: {last_conv}")

        for root, _, files in os.walk(input_dir):
            for filename in files:
                if filename.lower().endswith(valid_extensions):
                    image_path = os.path.join(root, filename)
                    try:
                        logger.debug(f"\nProcessing image: {image_path}")
                        img = load_img(image_path, target_size=target_size, color_mode=color_mode)
                        img_array = img_to_array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)

                        logger.debug("Computing Grad-CAM heatmap...")
                        heatmap = compute_gradcam(model, img_array, last_conv)

                        logger.debug("Creating visualization...")
                        overlay_heatmap_on_image(image_path, heatmap, output_dir)

                    except Exception as e:
                        logger.error(f"❌ Error processing {image_path}: {str(e)}")
                        continue
    except Exception as e:
        logger.error(f"Error in process_images_in_directory: {str(e)}")
        raise

def train_model(train_dir, val_dir, model_path, input_shape=(500, 500, 1), epochs=10, batch_size=4):
    """Train the CNN model on the dataset."""
    try:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=input_shape[:2],
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='binary'
        )

        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=input_shape[:2],
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='binary'
        )

        logger.debug(f"Class indices: {train_generator.class_indices}")
        logger.debug(f"Training images: {train_generator.samples}")
        logger.debug(f"Validation images: {val_generator.samples}")

        model = build_cnn_model(input_shape)
        checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max')

        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[checkpoint]
        )

        return model

    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise

if __name__ == "__main__":
    # Paths
    dataset_dir = r"C:\Users\HP\Downloads\pneumonia dataset\chest_xray"
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')
    model_path = r"C:\Users\HP\Downloads\FinalYearProject-master\FinalYearProject-master\models\pneu_cnn_model.h5"
    output_dir = r"C:\Users\HP\Downloads\FinalYearProject-master\FinalYearProject-master\static\output_heatmaps"

    try:
        # Verify dataset directories
        if not os.path.exists(train_dir) or not os.path.exists(val_dir) or not os.path.exists(test_dir):
            raise FileNotFoundError("One or more dataset directories not found. Check dataset path.")

        # Debug: Count images in train subdirectories
        logger.debug("Checking dataset...")
        for subdir in ['NORMAL', 'PNEUMONIA']:
            train_subdir = os.path.join(train_dir, subdir)
            if os.path.exists(train_subdir):
                logger.debug(f"Train {subdir}: {len(os.listdir(train_subdir))} images")
            else:
                logger.debug(f"Train {subdir} directory not found")

        # Train the model
        logger.debug("Training model...")
        model = train_model(train_dir, val_dir, model_path, epochs=10, batch_size=4)

        # Load the best saved model
        logger.debug("Loading trained model...")
        model = load_model(model_path)

        # Process test images with Grad-CAM
        logger.debug("Processing test images with Grad-CAM...")
        process_images_in_directory(model, test_dir, output_dir)

        logger.debug("✅ Training and Grad-CAM processing completed successfully!")

    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise