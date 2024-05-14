import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout, QMessageBox
from PyQt5.QtCore import Qt

from modules.image_processing_modules import (
    preprocess_image,
    dynamic_color_segmentation,
    calculate_road_length,
    create_mask,
    find_contours,
    apply_morphology,
    calculate_metrics,
    calculate_accuracy,
    calculate_iou,
    hybrid_edge_detection,
)

class ImageProcessorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Road Feature Detection App")
        self.setGeometry(100, 100, 1000, 800)

        # Main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        # Buttons
        self.btn_load = QPushButton("Load Image", self)
        self.btn_load.clicked.connect(self.loadImage)
        layout.addWidget(self.btn_load)

        self.btn_process = QPushButton("Process Image", self)
        self.btn_process.clicked.connect(self.processImage)
        layout.addWidget(self.btn_process)

        # Image display
        self.label_original = QLabel(self)
        self.label_original.setFixedSize(300, 300)
        self.label_processed = QLabel(self)
        self.label_processed.setFixedSize(300, 300)
        self.label_edges = QLabel(self)
        self.label_edges.setFixedSize(300, 300)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_processed)
        image_layout.addWidget(self.label_edges)
        layout.addLayout(image_layout)

        # Metrics display
        self.label_metrics = QLabel("Metrics: F1, Accuracy, Length", self)
        layout.addWidget(self.label_metrics)

    def loadImage(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open file", "/home", "Image files (*.jpg *.jpeg *.png)"
        )
        if fname:
            self.original_image = cv2.imread(fname)
            self.displayImage(self.original_image, self.label_original)

    def processImage(self):
        if hasattr(self, "original_image"):
            processed, edges, metrics = self.runProcessing(self.original_image)
            self.displayImage(processed, self.label_processed)
            self.displayImage(edges, self.label_edges)
            self.label_metrics.setText(
                f"Metrics: F1 Score: {metrics['f1']:.2f}, Accuracy: {metrics['accuracy']:.2f}, Total Length: {metrics['road_length']:.2f} meters"
            )
        else:
            QMessageBox.warning(self, "Error", "Load an image first!")

    def displayImage(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def runProcessing(self, img):
        preprocessed = preprocess_image(img)
        edges = hybrid_edge_detection(preprocessed)
        highlighted_roads, road_mask, metrics = self.analyzeImage(preprocessed)
        return highlighted_roads, edges, metrics

    def analyzeImage(self, img):
        # Apply dynamic color segmentation to get the initial mask
        mask = dynamic_color_segmentation(img)

        # Apply morphological closing to fill small holes and connect nearby objects
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Optionally, you could also apply opening to remove small noise
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours from the morphologically enhanced mask
        contours = find_contours(opened_mask)
        road_mask = create_mask(contours, img.shape[:2])

        # Calculate the road length and other metrics
        road_length, road_contours = calculate_road_length(contours, pixel_to_real=0.05)
        iou_score = calculate_iou(road_mask, mask)
        precision, recall, f1 = calculate_metrics(road_mask, mask)
        accuracy = calculate_accuracy(road_mask, mask)

        # Draw detected roads on the original image
        highlighted_roads = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(highlighted_roads, road_contours, -1, (0, 255, 0), 2)

        return highlighted_roads, road_mask, {"f1": f1, "accuracy": accuracy, "road_length": road_length}

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ex = ImageProcessorApp()
    ex.show()
    sys.exit(app.exec_())
