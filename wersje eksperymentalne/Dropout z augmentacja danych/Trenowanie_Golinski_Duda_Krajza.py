import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import scipy.io as sio
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


rozmiar = (32, 32)  


def load_svhn_data(mat_file):
    data = sio.loadmat(mat_file)
    images = data['X']
    labels = data['y']
    
    images = np.moveaxis(images, -1, 0)  # Przestawienie osi (TensorFlow oczekuje kanałów jako ostatni wymiar)
    images = images / 255.0  # Normalizacja do zakresu [0, 1]
    
    labels = np.where(labels == 10, 0, labels)  # Zamiana etykiety "10" na "0" (cyfra 0)
    
    return images, labels.squeeze()


def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),  # Dropout po pierwszej warstwie konwolucyjnej
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Dropout po drugiej warstwie konwolucyjnej
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Dropout w warstwie w pełni połączonej
        
        layers.Dense(10, activation='softmax')
    ])
    return model


X, y = load_svhn_data('extra_32x32.mat')

# Podział na zbiory treningowy i walidacyjny
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Augmentacja danych
datagen = ImageDataGenerator(
    rotation_range=15,          # Losowe obroty do 15 stopni
    width_shift_range=0.1,      # Przesunięcia w poziomie
    height_shift_range=0.1,     # Przesunięcia w pionie
    zoom_range=0.1,             # Skalowanie (zoom-in, zoom-out)
    brightness_range=[0.8, 1.2] # Zmiany jasności
)

datagen.fit(X_train)


model = create_model(input_shape=(rozmiar[0], rozmiar[1], 3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu z augmentacją danych
model.fit(datagen.flow(X_train, y_train, batch_size=32), 
          validation_data=(X_val, y_val), epochs=50)

# Zapis modelu
model.save("model_Svhn.keras")
print("Model zapisany.")
