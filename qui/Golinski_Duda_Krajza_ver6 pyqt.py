import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen, QImage, QColor, QPixmap
from PIL import Image

class DrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        
        self.setWindowTitle("rozposnywanie cyfry")
        self.setGeometry(100, 100, 320, 400)

        
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        
        layout = QVBoxLayout()

        
        self.recognize_button = QPushButton("Rozpoznaj")
        self.recognize_button.clicked.connect(self.process_drawing)
        layout.addWidget(self.recognize_button)

        


        self.clear_button = QPushButton("Wyczyść")
        self.clear_button.clicked.connect(self.clear_canvas)
        layout.addWidget(self.clear_button)

        


        self.canvas_label = QLabel(self)
        self.canvas_label.setFixedSize(320, 320)
        self.canvas_image = QImage(self.canvas_label.size(), QImage.Format_RGB32)
        self.canvas_image.fill(Qt.white)
        layout.addWidget(self.canvas_label)

        
        self.central_widget.setLayout(layout)

        
        self.drawing = False
        self.last_point = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.canvas_image)
            pen = QPen(Qt.black, 30, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)

            if self.last_point:
                painter.drawLine(self.last_point - self.canvas_label.pos(), event.pos() - self.canvas_label.pos())
            self.last_point = event.pos()

            self.update_canvas()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_point = None

    def update_canvas(self):
        """Odświeżenie płótna."""
        pixmap = QPixmap.fromImage(self.canvas_image) 
        self.canvas_label.setPixmap(pixmap)

    def clear_canvas(self):
        """Czyszczenie płótna."""
        self.canvas_image.fill(Qt.white)
        self.update_canvas()

    def process_drawing(self):
        """Przetwarzanie rysunku na 32x32 oraz zapis do pliku PNG."""
        
        buffer = self.canvas_image.bits().asstring(self.canvas_image.width() * self.canvas_image.height() * 4)
        pil_image = Image.frombytes('RGBA', (self.canvas_image.width(), self.canvas_image.height()), buffer)

        
        
        pil_image = pil_image.convert('L').resize((32, 32))
        pil_image.save('image.png')

        


        img_array = np.array(pil_image) / 255.0
        img_array = img_array.reshape((1, 32, 32, 1))
        print(img_array)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrawingApp()
    window.show()
    sys.exit(app.exec_())