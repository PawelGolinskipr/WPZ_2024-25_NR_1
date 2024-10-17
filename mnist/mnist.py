import tensorflow as tf
from keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
# do wyszukiwaniu plików png
import os
import fnmatch

print(tf.config.list_physical_devices('GPU'))


# rozmiar plików -> znacznie zwiękrza czas trenowania z 6ms/step(28x28)->44ms/step(68x684) -> 235ms/step(128x128)  /testowania/rozpoznywanie
rozmiar = [28,48 ,68,128][0]
#print(rozmiar)
early_stopping = EarlyStopping(monitor='accuracy', patience=2, verbose=1, mode='max', restore_best_weights=True) # jeżeli jest już dokładność max (to te z "mode")

def szukamPlikow(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

model=None
mainloop=True



def load_data():
    # Załaduj dane MNIST
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Rozszerzamy wymiary do (num_samples, height, width, channels)
    train_images = np.expand_dims(train_images, axis=-1)  # Dodajemy kanał
    test_images = np.expand_dims(test_images, axis=-1)    # Dodajemy kanał
    
    # Zmiana rozmiaru obrazów na 64x64
    train_images = tf.image.resize(train_images, [rozmiar, rozmiar])
    test_images = tf.image.resize(test_images, [rozmiar, rozmiar])

    # Przekształcamy do typu float32 i normalizujemy
    train_images = train_images.numpy().astype('float32') / 255.0
    test_images = test_images.numpy().astype('float32') / 255.0
    
    return (train_images, train_labels), (test_images, test_labels)

# Wczytaj dane
(train_images, train_labels), (test_images, test_labels) = load_data()





def przygodowanieZdjecia(image_path, target_size=(rozmiar, rozmiar)):# żevy działało dla innych zdjeć niż 28x28
    image = load_img(image_path, color_mode="grayscale", target_size=target_size)  # Upewnij się, że obrazek jest odpowiednio przeskalowany
    image_array = img_to_array(image)
    
    # Przeskalowanie wartości pikseli do zakresu [0, 1]
    image_array = image_array.astype('float32') / 255.0

    # Dodaj wymiar batcha
    image_array = np.expand_dims(image_array, axis=0)
    plt.imshow(image_array.squeeze(), cmap='gray')  # Używamy cmap='gray' dla obrazów w skali szarości
    plt.axis('off')  # Ukrycie osi
    plt.show()
    return image_array

while mainloop:
    if model is None:
        print("NIE MASZ MODELU")
    else:
        print("Model jest aktywny")
    wybor=input("Wybierz: \n1. wczytanie modelu\n2.zapisanie modelu\n3. trenowanie modelu\n4.testowanie modelu\n5.nowy model\n6. rozpoznawanie cyfry\n7. wyjscie\n")
    match wybor:
        case "1":
            try:
                model = tf.keras.models.load_model("mnist_model.keras")
                #49 pierwsze dokładności 1.0
                
                # bez tego wywala błąd o wagi - trzeba ponownie dać compile
                optimizer = Adam() 
                model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                
                
                print("Model wczytany")
            except Exception as e:
                print(e) # tu jest wyrzucenie błedu
        case "2":
            model.save('mnist_model.keras')
            print("Model zapisany")
        case "3":
            try:

                ileRazyprzejsc=int(input("Ile razy chcesz trenowac model? (automatycznie zakończy trenowanie jeżeli dokładność będzie gorsze niż poprzedniej teracji)"))
            
                model.fit(train_images, train_labels, epochs=ileRazyprzejsc, batch_size=100, callbacks=[early_stopping],validation_data=(test_images, test_labels))
            except Exception as e:
                print(e) # tu jest wyrzucenie błedu
        case "4":
            try:
                test_loss, test_acc = model.evaluate(test_images, test_labels)
                print(f"Dokładność na zestawie testowym: {test_acc}")
            except Exception as e:
                print(e) # tu jest wyrzucenie błedu/brak modelu
        case "5":
            model = None
            model = models.Sequential()

# Warstwa konwolucyjna
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(rozmiar, rozmiar, 1)))
            model.add(layers.MaxPooling2D((2, 2)))

# Warstwa Dropout
            model.add(layers.Dropout(0.25))  # 25% neuronów zostanie wyłączonych

# Kolejne warstwy konwolucyjne
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))

# Warstwa Dropout
            model.add(layers.Dropout(0.25))  # 25% neuronów zostanie wyłączonych

# Spłaszczenie
            model.add(layers.Flatten())

# W pełni połączona warstwa
            model.add(layers.Dense(64, activation='relu'))

# Warstwa Dropout
            model.add(layers.Dropout(0.5))  # 50% neuronów zostanie wyłączonych

# Wyjściowa warstwa
            model.add(layers.Dense(10, activation='softmax'))

# Kompilacja modelu
            model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        
        case "6":
            # rozpoznanie cyfry użykownika pod folder "zdjecia" wszystykie zdjecia w folderze "zdjecia", only png 
            folder="zdjecia/"
            pliki=szukamPlikow("*.png", folder)
            print(pliki)

            for plik in pliki:
                
                wynik=model.predict(przygodowanieZdjecia(plik))
                # wynik zawiera prawdopodobieństwo index to jest cyfra
                print(wynik[0][np.argmax(wynik)])
                print("Najwyższa przewidywana klasa: ", np.argmax(wynik)) 
                print("Wynik: ", wynik.flatten(), " dla pliku: ", plik)
        case "7":
            mainloop=False
        case _:
            print("Nie ma takiej opcji")



"""

# Budowa modelu CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Dodanie warstwy spłaszczającej i w pełni połączonej
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# Trenowanie modelu
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Ocena modelu na zestawie testowym
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Dokładność na zestawie testowym: {test_acc}")
"""
