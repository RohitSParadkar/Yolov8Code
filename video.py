import os
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from io import BytesIO
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("../../models/bestyM_6_5_2024.pt")
names = model.names
print(names)

# Open the video file
video_path = "../../output/testVideo/testData2.mp4"
cap = cv2.VideoCapture(video_path)

# Create PyQt application and window
app = QApplication([])

widget = QWidget()
layout = QVBoxLayout()
label = QLabel()
layout.addWidget(label)
widget.setLayout(layout)
widget.setWindowTitle("YOLO Prediction Result")

# Read the first frame from the video
ret, frame = cap.read()

# Resize frame to 600x600
frame = cv2.resize(frame, (600, 600))

# Convert frame to RGB (PIL format)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

while ret:
    # Perform prediction on the frame
    results = model.predict(frame_rgb,conf=0.1)
    result = results[0]
    

    # Convert the image from numpy array to a PIL Image
    pil_image = Image.fromarray(result.plot()[:,:,::-1])

    # Convert PIL Image to BytesIO
    image_bytes = BytesIO()
    pil_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Display the image using PyQt
    pixmap = QPixmap()
    pixmap.loadFromData(image_bytes.getvalue())
    label.setPixmap(pixmap)
    widget.show()  # Show the widget
    app.processEvents()  # Allow the application to process events (update GUI)
    
    # Read the next frame
    ret, frame = cap.read()
    
    
    if ret:
        # Resize frame to 600x600
        frame = cv2.resize(frame, (600, 600))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Release the video capture object and close the window when done
cap.release()
cv2.destroyAllWindows()

# Start the application event loop
app.exec_()
