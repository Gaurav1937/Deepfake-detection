import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
)
from PyQt6.QtGui import QPixmap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class DeepfakeDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.model = None

    def init_ui(self):
        self.setWindowTitle("Deepfake Detection")
        self.setGeometry(100, 100, 600, 400)

        # Layout
        layout = QVBoxLayout()

        # Label to display selected image
        self.image_label = QLabel("No image selected.")
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setScaledContents(True)
        layout.addWidget(self.image_label)

        # Button to select an image
        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)

        # Button to predict
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict_image)
        layout.addWidget(self.predict_button)

        # Label to display prediction result
        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        # Set layout
        self.setLayout(layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_label.setPixmap(QPixmap(file_path))
            self.selected_image_path = file_path

    def predict_image(self):
        if not hasattr(self, 'selected_image_path'):
            self.result_label.setText("Please select an image first.")
            return

        # Load the image and preprocess it
        img = load_img(self.selected_image_path, target_size=(224, 224))  # Match model input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize

        # Load the model if not already loaded
        if self.model is None:
            self.model = load_model("df_model.keras")

        # Predict
        prediction = self.model.predict(img_array)[0][0]

        # Display result
        if prediction < 0.5:
            self.result_label.setText(f"Prediction: Fake ({prediction:.2f})")
        else:
            self.result_label.setText(f"Prediction: Real ({prediction:.2f})")

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepfakeDetectionApp()
    window.show()
    sys.exit(app.exec())
