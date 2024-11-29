import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
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


model = create_model(input_shape=(rozmiar[0], rozmiar[1], 3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



if os.path.exists("model_Svhn.keras"):
    
    print("Wczytywanie istniejącego modelu...")
    model = tf.keras.models.load_model("model_Svhn.keras")
else:
    
    print("Trenowanie nowego modelu...")
    model = create_model(input_shape=(rozmiar[0], rozmiar[1], 3)) 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    
    
    model.save("model_Svhn.keras")
    print("Model zapisany.")

#history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))


X_img, img_filenames = load_images_from_folder('zdjecia', rozmiar)


predictions = model.predict(X_img)
predicted_labels = np.argmax(predictions, axis=1)

# Wyświetlenie wyników przewidywania
for i, filename in enumerate(img_filenames):
    print(f"Plik: {filename}, Przewidziana cyfra: {predicted_labels[i]}")

    
    plt.imshow(X_img[i])
    plt.title(f"Przewidziana cyfra: {predicted_labels[i]}")
    plt.show()