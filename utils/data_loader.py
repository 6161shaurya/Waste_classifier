# utils/data_loader.py

import os
import requests
import zipfile
import shutil
from tqdm import tqdm # For showing download progress
import tensorflow as tf
import pandas as pd # For creating DataFrame to split paths/labels
from sklearn.model_selection import train_test_split # For splitting data

def download_and_extract_trashnet(data_dir='data', url='https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip'):
    """
    Downloads the TrashNet dataset (or a similar small zip file) and extracts it.
    If the data is already extracted, it skips the download.

    Args:
        data_dir (str): The base directory where the dataset will be stored.
        url (str): The URL to the zip file containing the dataset.
    
    Returns:
        str: The path to the extracted dataset directory (e.g., 'data/dataset-resized').
             Returns None if download/extraction fails.
    """
    zip_file_path = os.path.join(data_dir, os.path.basename(url))
    extracted_dir_name = os.path.basename(url).replace('.zip', '')
    extracted_path = os.path.join(data_dir, extracted_dir_name)

    # Create the base data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Check if data is already extracted and looks complete
    if os.path.exists(extracted_path) and any(os.path.isdir(os.path.join(extracted_path, d)) for d in os.listdir(extracted_path) if not d.startswith('.')):
        print(f"Dataset already extracted to {extracted_path}. Skipping download.")
        return extracted_path

    # --- Download the zip file ---
    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 KB
        tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(zip_file_path))

        with open(zip_file_path, 'wb') as file:
            for data_chunk in response.iter_content(block_size):
                tqdm_bar.update(len(data_chunk))
                file.write(data_chunk)
        tqdm_bar.close()
        print(f"Downloaded {zip_file_path}")

        # --- Extract the zip file ---
        print(f"Extracting {zip_file_path} to {data_dir}...") # Extract to data_dir
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir) # Extracts directly into data_dir
        print("Extraction complete.")
        
        # --- Clean up the zip file after extraction ---
        os.remove(zip_file_path)
        print(f"Removed temporary zip file: {zip_file_path}")

        # --- Handle common nested directory issue (e.g., zip contains 'dataset-resized/dataset-resized/...') ---
        if os.path.exists(os.path.join(extracted_path, extracted_dir_name)) and \
           os.path.isdir(os.path.join(extracted_path, extracted_dir_name)) and \
           any(os.path.isdir(os.path.join(extracted_path, extracted_dir_name, d)) for d in os.listdir(os.path.join(extracted_path, extracted_dir_name)) if not d.startswith('.')):
            
            print(f"Adjusting directory structure: moving contents from {os.path.join(extracted_path, extracted_dir_name)} to {extracted_path}")
            temp_move_dir = os.path.join(data_dir, "temp_extracted_contents")
            os.rename(os.path.join(extracted_path, extracted_dir_name), temp_move_dir)
            
            shutil.rmtree(extracted_path)
            
            os.rename(temp_move_dir, extracted_path)
            print("Directory structure adjusted.")
        
        return extracted_path

    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path) # Clean up partial download
        return None
    except zipfile.BadZipFile as e:
        print(f"Error extracting zip file (possibly corrupted): {e}")
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def prepare_dataset_for_training(raw_dataset_path, img_height=128, img_width=128, batch_size=32, val_split=0.1, test_split=0.1):
    """
    Prepares the dataset for TensorFlow model training, including data splitting
    into training, validation, and test sets and creating ImageDataGenerators.
    This function handles datasets where all images are initially in class subdirectories
    within a single main folder (like TrashNet's 'dataset-resized').

    Args:
        raw_dataset_path (str): Path to the extracted dataset (e.g., 'data/dataset-resized').
                                 This folder should directly contain class-named subdirectories.
        img_height (int): Height of images after resizing.
        img_width (int): Width of images after resizing.
        batch_size (int): Number of images per batch.
        val_split (float): Proportion of data to use for the validation set.
        test_split (float): Proportion of data to use for the test set.
    
    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names)
               TensorFlow ImageDataGenerators for train, validation, test sets
               and a list of class names.
               Returns (None, None, None, None) if dataset path is invalid or empty.
    """
    if not os.path.exists(raw_dataset_path):
        print(f"Error: Raw dataset path '{raw_dataset_path}' does not exist.")
        return None, None, None, None

    print(f"Preparing dataset from {raw_dataset_path}...")

    # Define the new directories for split data
    base_split_dir = os.path.join(os.path.dirname(raw_dataset_path), "split_dataset") # e.g., data/split_dataset
    train_dir = os.path.join(base_split_dir, 'train')
    val_dir = os.path.join(base_split_dir, 'validation')
    test_dir = os.path.join(base_split_dir, 'test')

    # Get class names and all image paths
    all_image_paths = []
    all_labels = []
    
    class_names = sorted([d for d in os.listdir(raw_dataset_path) if os.path.isdir(os.path.join(raw_dataset_path, d)) and not d.startswith('.')])
    
    if not class_names:
        print(f"Error: No class subdirectories found in '{raw_dataset_path}'. Please check dataset structure.")
        return None, None, None, None

    print(f"Found classes: {class_names}")

    # Check if dataset is already split into the target train/val/test directories
    if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir) and \
       all(os.path.exists(os.path.join(train_dir, c)) for c in class_names): # Basic check for content
        print("Dataset already split into train/validation/test folders. Skipping splitting.")
    else:
        print("Splitting dataset into train/validation/test sets...")
        # Clean up old split if it exists, to ensure a fresh split
        if os.path.exists(base_split_dir):
            print(f"Cleaning up old split directory: {base_split_dir}")
            shutil.rmtree(base_split_dir)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for class_name in class_names:
            class_path = os.path.join(raw_dataset_path, class_name)
            images_in_class = [os.path.join(class_path, img) for img in os.listdir(class_path) if not img.startswith('.')]
            
            # Split current class images into train, val, test
            train_files, test_val_files = train_test_split(images_in_class, test_size=(val_split + test_split), random_state=42)
            val_files, test_files = train_test_split(test_val_files, test_size=(test_split / (val_split + test_split)), random_state=42)
            
            # Create destination directories for each class within the splits
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # Copy files to their respective new directories
            for f in tqdm(train_files, desc=f"Copying {class_name} to train"): shutil.copy(f, os.path.join(train_dir, class_name))
            for f in tqdm(val_files, desc=f"Copying {class_name} to val"): shutil.copy(f, os.path.join(val_dir, class_name))
            for f in tqdm(test_files, desc=f"Copying {class_name} to test"): shutil.copy(f, os.path.join(test_dir, class_name))
        
        print(f"Dataset splitting complete. Data moved to {base_split_dir}")

    # --- ImageDataGenerators for Loading and Augmentation ---
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, # Normalize pixel values
        rotation_range=20, # Randomly rotate
        width_shift_range=0.2, # Randomly shift horizontally
        height_shift_range=0.2, # Randomly shift vertically
        shear_range=0.2, # Shear transformations
        zoom_range=0.2, # Randomly zoom
        horizontal_flip=True, # Randomly flip horizontally
        fill_mode='nearest' # Fill empty pixels
    )

    # Validation and test generators only rescale (no augmentation)
    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    print("Creating data generators...")
    train_ds = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical', # For multi-class classification
        seed=42
    )

    val_ds = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        seed=42
    )

    test_ds = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False, # Do not shuffle test set for consistent evaluation
        seed=42
    )

    print("Dataset preparation complete.")
    return train_ds, val_ds, test_ds, class_names


# --- Example usage (for testing data_loader.py independently) ---
if __name__ == '__main__':
    print("Running data_loader.py for testing purposes...")
    base_data_path = 'data' # This creates 'data' folder at project root
    
    # URL for the TrashNet dataset (dataset-resized.zip)
    trashnet_url = 'https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip'
    
    extracted_dataset_path = download_and_extract_trashnet(data_dir=base_data_path, url=trashnet_url)

    if extracted_dataset_path:
        print(f"Extracted dataset base path: {extracted_dataset_path}")
        # Verify that the extracted path contains subdirectories named after classes
        
        train_ds, val_ds, test_ds, class_names = prepare_dataset_for_training(
            raw_dataset_path=extracted_dataset_path, # Should be 'data/dataset-resized'
            img_height=128, img_width=128, batch_size=32
        )

        if train_ds:
            print("\nDataset preparation successful!")
            print(f"Class names: {class_names}")
            print(f"Number of classes: {len(class_names)}")
            print(f"Training batches: {len(train_ds)}")
            print(f"Validation batches: {len(val_ds)}")
            print(f"Test batches: {len(test_ds)}")
            
            # Display a sample batch
            print("\nDisplaying a sample batch from training set:")
            for images, labels in train_ds.take(1):
                plt.figure(figsize=(10, 10))
                for i in range(9): # Display first 9 images
                    ax = plt.subplot(3, 3, i + 1)
                    plt.imshow(images[i].numpy())
                    # Get class name from one-hot encoded label
                    predicted_class_label = class_names[tf.argmax(labels[i])]
                    plt.title(predicted_class_label)
                    plt.axis("off")
                plt.tight_layout()
                plt.show()
                break
        else:
            print("Dataset preparation failed.")
    else:
        print("Dataset download/extraction failed or skipped.")