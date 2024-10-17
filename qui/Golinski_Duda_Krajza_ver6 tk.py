import tkinter as tk
from PIL import Image, ImageGrab
import numpy as np
from tkinter import *

# Ustawienia płótna i pędzla
canvas_width = 320
canvas_height = 320
brush_size = 30  # Grubość pędzla

root = tk.Tk()
root.title("Rysowanie cyfry")

# Tworzenie płótna
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

drawing = False

def start_drawing(event):
    """Rozpocznij rysowanie przy kliknięciu myszką."""
    global drawing
    drawing = True
    draw(event.x, event.y)

def stop_drawing(event):
    """Zatrzymaj rysowanie przy zwolnieniu myszki."""
    global drawing
    drawing = False

def draw(x, y):
    """Rysuj na płótnie, gdy myszka jest wciśnięta."""
    if drawing:
        canvas.create_oval(x, y, x + brush_size, y + brush_size, fill='black', outline='black')

def clear_canvas():
    """Czyść płótno."""
    canvas.delete("all")

def process_drawing():
    """Przetwarzaj rysunek na dane wejściowe dla modelu."""
    
    # Pobieranie współrzędnych płótna
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Zrzut ekranu dokładnie tego obszaru
    img = ImageGrab.grab(bbox=(x, y, x1, y1))

    # Zmiana rozmiaru obrazu na 32x32 i konwersja do odcieni szarości
    img = img.resize((32, 32))
    img = img.convert('L')  # Konwersja do skali szarości
    img.save('image.png')  # Zapis obrazu

    # img_array = np.array(img) / 255.0  # Normalizacja do przedziału [0, 1]
    # img_array = img_array.reshape((1, 32, 32, 1))  # Przygotowanie dla modelu
    # return img_array

# Przypisanie funkcji do zdarzeń
canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", lambda event: draw(event.x, event.y))
canvas.bind("<ButtonRelease-1>", stop_drawing)

# Przyciski
recognize_button = tk.Button(root, text="Rozpoznaj", command=process_drawing)
recognize_button.pack()

clear_button = tk.Button(root, text="Wyczyść", command=clear_canvas)
clear_button.pack()

root.mainloop()
