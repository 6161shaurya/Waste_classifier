# classifier_model.py

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2 # For transfer learning
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Import data_loader utilities
from utils.data_loader import download_and_extract_trashnet, prepare_dataset_for_training

# --- Global/Config Parameters ---
IMG_HEIGHT, IMG_WIDTH = 128, 128 # Image dimensions for the model
BATCH_SIZE = 32
EPOCHS = 20 # Number of training epochs (can adjust)
MODEL_SAVE_PATH = 'models/waste_classifier_model.h5' # Path to save the trained model

def build_transfer_learning_model(num_classes, img_height, img_width):
    """
    Builds a CNN model using Transfer Learning with MobileNetV2.
    MobileNetV2 is pre-trained on ImageNet and is efficient.

    Args:
        num_classes (int): Number of output classes for classification.
        img_height (int): Height of input images.
        img_width (int): Width of input images.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
    print("Building transfer learning model (MobileNetV2)...")
    # Load the pre-trained MobileNetV2 model without the top (classification) layer
    base_model = MobileNetV2(
        input_shape=(img_height, img_width, 3), # Expects 3 color channels (RGB)
        include_top=False, # Exclude the top classification layer
        weights='imagenet' # Use weights pre-trained on ImageNet
    )

    # Freeze the base model layers to prevent their weights from being updated during training
    # This keeps the learned features from ImageNet intact.
    base_model.trainable = False

    # Create a new Sequential model and add the base model
    model = Sequential([
        base_model, # The pre-trained MobileNetV2
        GlobalAveragePooling2D(), # Global average pooling layer to reduce dimensions
        Dense(128, activation='relu'), # A dense layer for processing features
        Dropout(0.3), # Dropout for regularization (prevents overfitting)
        Dense(num_classes, activation='softmax') # Output layer with 'softmax' for multi-class probability
    ])

    # Compile the model
    # Adam optimizer is common for deep learning
    # 'categorical_crossentropy' for multi-class classification with one-hot encoded labels
    # 'accuracy' as a metric to monitor
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary() # Print a summary of the model architecture
    print("Model built successfully.")
    return model

def train_model(model, train_ds, val_ds, epochs, model_save_path):
    """
    Trains the provided Keras model using the given datasets.

    Args:
        model (tf.keras.Model): The compiled Keras model.
        train_ds (tf.keras.preprocessing.image.DirectoryIterator): Training data generator.
        val_ds (tf.keras.preprocessing.image.DirectoryIterator): Validation data generator.
        epochs (int): Number of training epochs.
        model_save_path (str): Path to save the best model weights.

    Returns:
        tf.keras.callbacks.History: History object containing training metrics per epoch.
    """
    print(f"Starting model training for {epochs} epochs...")
    
    # Callbacks for better training management
    # ModelCheckpoint: Saves the best model weights based on validation accuracy
    checkpoint_callback = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy', # Monitor validation accuracy
        save_best_only=True, # Save only the best model
        mode='max', # Maximize validation accuracy
        verbose=1
    )
    # EarlyStopping: Stops training if validation accuracy doesn't improve for a few epochs
    early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=5, # Number of epochs with no improvement after which training will be stopped
        mode='max',
        verbose=1,
        restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping_callback] # Add callbacks here
    )
    print("Model training complete.")
    return history

def evaluate_model(model, test_ds, class_names):
    """
    Evaluates the trained model on the test dataset and prints classification metrics.

    Args:
        model (tf.keras.Model): The trained Keras model.
        test_ds (tf.keras.preprocessing.image.DirectoryIterator): Test data generator.
        class_names (list): List of class names corresponding to model output.
    """
    print("\nEvaluating model on test set...")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate predictions for classification report and confusion matrix
    print("\nGenerating predictions for classification report and confusion matrix...")
    y_pred_probs = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Get true labels from the test generator
    y_true_labels = test_ds.classes
    
    # Ensure y_true_labels length matches y_pred_classes length (important for test_ds that might loop)
    # y_true_labels should already be the full set of labels if test_ds.classes is used
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true_labels, y_pred_classes, target_names=class_names))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true_labels, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Plots training and validation accuracy and loss over epochs.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- Main execution block for classifier_model.py ---
if __name__ == '__main__':
    print("Starting waste classifier model training process...")

    # 1. Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # 2. Download and prepare dataset
    base_data_path = 'data'
    trashnet_url = 'https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip'
    extracted_dataset_path = download_and_extract_trashnet(data_dir=base_data_path, url=trashnet_url)

    if not extracted_dataset_path:
        print("Failed to download or extract dataset. Exiting.")
        exit()

    train_ds, val_ds, test_ds, class_names = prepare_dataset_for_training(
        raw_dataset_path=extracted_dataset_path,
        img_height=IMG_HEIGHT, img_width=IMG_WIDTH, batch_size=BATCH_SIZE
    )

    if not all([train_ds, val_ds, test_ds, class_names]):
        print("Failed to prepare dataset. Exiting.")
        exit()
        
    num_classes = len(class_names)
    if num_classes == 0:
        print("No classes detected. Exiting.")
        exit()

    # 3. Build the model
    model = build_transfer_learning_model(num_classes, IMG_HEIGHT, IMG_WIDTH)

    # 4. Train the model
    history = train_model(model, train_ds, val_ds, EPOCHS, MODEL_SAVE_PATH)

    # 5. Plot training history
    plot_training_history(history)

    # 6. Evaluate the model
    # Load the best model saved by ModelCheckpoint for final evaluation
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    evaluate_model(best_model, test_ds, class_names)

    print("\nWaste classifier model training and evaluation complete.")