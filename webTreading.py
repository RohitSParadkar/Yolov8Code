import cv2
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from io import BytesIO
from ultralytics import YOLO
from threading import Thread
import time

class YOLOThreaded:
    def __init__(self, video_source, model_path, label):
        self.video_source = video_source
        self.model = YOLO(model_path)
        self.label = label
        self.cap = cv2.VideoCapture(video_source)
        self.stop_flag = False
        self.thread = Thread(target=self.process_frames)

    def process_frames(self):
        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Resize frame to 600x600
            frame = cv2.resize(frame, (600, 600))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform prediction on the frame
            results = self.model.predict(frame_rgb, conf=0.1)
            result = results[0]

            # Convert the image from numpy array to a PIL Image
            pil_image = Image.fromarray(result.plot()[:, :, ::-1])

            # Convert PIL Image to BytesIO
            image_bytes = BytesIO()
            pil_image.save(image_bytes, format='PNG')
            image_bytes.seek(0)

            # Display the image using PyQt
            pixmap = QPixmap()
            pixmap.loadFromData(image_bytes.getvalue())

            # Update label with new pixmap (GUI update)
            self.label.setPixmap(pixmap)
            self.label.repaint()  # Force label repaint

            # Calculate frame rate (FPS) and ensure it's valid
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                time.sleep(1 / fps)  # Slow down processing to match video frame rate

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        self.thread.join()
        self.cap.release()

class MultiCamYOLOThreaded:
    def __init__(self, video_sources, model_path):
        self.app = QApplication([])
        self.windows = []
        self.labels = []
        self.threads = []

        for idx, video_source in enumerate(video_sources):
            window = QWidget()
            window.setWindowTitle(f"Camera {idx + 1} - YOLO Prediction Result")
            layout = QVBoxLayout()
            label = QLabel()
            layout.addWidget(label)
            window.setLayout(layout)
            window.show()
            self.windows.append(window)
            self.labels.append(label)

            yolo_thread = YOLOThreaded(video_source, model_path, label)
            yolo_thread.start()
            self.threads.append(yolo_thread)

    def start(self):
        self.app.exec_()  # Start the PyQt application event loop

    def stop(self):
        for thread in self.threads:
            thread.stop()  # Stop all YOLOThreaded instances
        self.app.quit()  # Quit the PyQt application event loop

if __name__ == "__main__":
    # Specify video sources (webcam indices)
    video_sources = [0, 1]  # Adjust based on the number of webcams connected

    # Specify path to the YOLO model
    model_path = "../../models/best_10_04_2023.pt"

    # Create MultiCamYOLOThreaded instance
    multi_cam_yolo_threaded = MultiCamYOLOThreaded(video_sources, model_path)

    # Start the PyQt application event loop
    multi_cam_yolo_threaded.start()

    # Clean up resources when the application exits
    multi_cam_yolo_threaded.stop()
