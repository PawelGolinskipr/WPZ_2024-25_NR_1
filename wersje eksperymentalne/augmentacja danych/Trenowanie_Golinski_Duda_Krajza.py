import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy.io as sio
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

rozmiar = (32, 32)

def load_svhn_data(mat_file, rozmiar):
    data = sio.loadmat(mat_file)
    images = data['X']
    labels = data['y']
    
    
    images = np.moveaxis(images, -1, 0)

   
    resized_images = np.array([cv2.resize(img, rozmiar) for img in images])
    
    # Normalizacja obrazów
    resized_images = resized_images / 255.0

   
    labels = np.where(labels == 10, 0, labels)
    
    return resized_images, labels.squeeze()

def load_images_from_folder(folder, rozmiar):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                
                img_resized = cv2.resize(img, rozmiar)
                img_resized = img_resized / 255.0  
                images.append(img_resized)
                filenames.append(filename)
    return np.array(images), filenames

def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

X, y = load_svhn_data('extra_32x32.mat', rozmiar)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


datagen = ImageDataGenerator(
    rotation_range=10,      
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    zoom_range=0.1,         
    horizontal_flip=False,   
    fill_mode='nearest'      
)


datagen.fit(X_train)


if os.path.exists("model_Svhn.keras"):
    print("Wczytywanie istniejącego modelu...")
    model = tf.keras.models.load_model("model_Svhn.keras")
else:
    print("Trenowanie nowego modelu...")
    model = create_model(input_shape=(rozmiar[0], rozmiar[1], 3)) 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
 
    model.fit(datagen.flow(X_train, y_train, batch_size=32), 
              validation_data=(X_val, y_val), 
              epochs=10)
    

    model.save("model_Svhn.keras")
    print("Model zapisany.")






