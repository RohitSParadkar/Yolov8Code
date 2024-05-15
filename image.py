import os
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
import time
from PyQt5.QtGui import QPixmap
from io import BytesIO
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("../../models/best_10_04_2023.pt")

# Get the class names predicted by the model

start = time.time()
# Perform prediction on a test image
results = model.predict("../../output/testImage/test1.jpg", conf=0.1)
result = results[0]
names = model.names


# result.boxes cls:classes_id , confidance score ,bounding boxes
# for r in results:
#     print("\nresults:",r.boxes)

for r in results:
    classList = []
    for c in r.boxes.cls:
        classList.append(names[int(c)])
        #print(names[int(c)])
    # print("flangeWithGasket" in classList)
    if(len(classList)==1 and "flangeWithGasket"in classList):
        print("Perfect")
    else:
        print("Damage detected")
    

# Convert the image from numpy array to a PIL Image
pil_image = Image.fromarray(result.plot()[:, :, ::-1])

# Resize the image
desired_width = 600  # Set your desired width
desired_height = 600  # Set your desired height
pil_image = pil_image.resize((desired_width, desired_height), Image.LANCZOS)

# Convert PIL Image to BytesIO
image_bytes = BytesIO()
pil_image.save(image_bytes, format='PNG')
image_bytes.seek(0)

# Display the image using PyQt
app = QApplication([])
widget = QWidget()
layout = QVBoxLayout()
label = QLabel()
pixmap = QPixmap()
pixmap.loadFromData(image_bytes.getvalue())
label.setPixmap(pixmap)
layout.addWidget(label)
widget.setLayout(layout)

# Set window size
window_width = desired_width + 10
window_height = desired_height + 10
widget.setGeometry(10, 10, window_width, window_height)
widget.setWindowTitle("YOLO Prediction Result V8")
widget.show()
end = time.time()
print("The total time is:",(end-start)*1000,"ms")
app.exec_()
