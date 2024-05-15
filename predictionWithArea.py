import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Constants
THRESHOLD_VALUE = 200
MAX_PIXEL_VALUE = 255
MIN_AREA_THRESHOLD = 58000
MAX_AREA_THRESHOLD = 40000

color1 = (128, 128, 128)
color2 = (128, 128, 128)
color3 = (128, 128, 128)

model = YOLO("../../models/bestyM_6_5_2024.pt")
names = model.names

# Mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at pixel coordinates (x={x}, y={y})")

# Open video capture
cap = cv2.VideoCapture(0)

# Create a window for displaying the frame
cv2.namedWindow("Frame")

# Set mouse callback function
cv2.setMouseCallback('Frame', click_event)

# Initialize your_frame outside the loop
your_frame = np.zeros((800, 200, 3), dtype=np.uint8)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (800, 800))
    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Extract flange region
    flange = frame[217:523, 125:533]
    
    # Convert to grayscale
    greyFlange = cv2.cvtColor(flange, cv2.COLOR_BGR2GRAY)
    
    # Thresholding
    _, threshold = cv2.threshold(greyFlange, THRESHOLD_VALUE, MAX_PIXEL_VALUE, cv2.THRESH_BINARY_INV)
    
    # Display thresholded image and processed flange
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Grey Flange", greyFlange)
    cv2.imshow("Flange", flange)
    
    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # (x, y, w, h) = cv2.boundingRect(cnt)
        
        # Draw rectangle if area is above a threshold
        if (area > MIN_AREA_THRESHOLD):
            print(area)
            color1 = (128, 128, 128)
            color2 = (128, 128, 128)
            color3 = (128, 128, 128)
            # cv2.rectangle(flange, (x, y), (x+w, y+h), (0, 255, 0), 3)
            results = model.predict(frame, conf=0.1, iou=0.4)
            
            for r in results:
                classList = []
                for c in r.boxes.cls:
                    classList.append(names[int(c)])
                if len(classList) == 0:
                    color1 = (0, 255, 255)  # Yellow
                elif len(classList) == 1 and "flangeWithGasket" in classList:
                    color2 = (0, 255, 0)  # Red
                elif len(classList) == 2 and "flangeWithGasket" in classList and "flangeBase" in classList:
                    color2 = (0, 255, 0)  # Green
                else:
                    color3 = (0, 0, 255)  # Grey (default)
                cv2.circle(your_frame, (100, 100), 50, color3, -1)
                cv2.circle(your_frame, (100, 100 + 50*2 + 20), 50, color1, -1)
                cv2.circle(your_frame, (100, 100 + (50*2 + 20)*2), 50, color2, -1)
            pil_image = Image.fromarray(results[0].plot()[:, :, ::-1])
            frame_with_overlay = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Concatenate your frame with the webcam frame
            final_frame = np.hstack((your_frame, frame_with_overlay))

            # Display the image using OpenCV
            cv2.imshow("Webcam - YOLO Prediction Result", final_frame)
        else:
            # Update the yellow circle in your_frame
            color1 = (0, 255, 255)  # Yellow
            color2 = (128, 128, 128)
            color3 = (128, 128, 128)
            cv2.circle(your_frame, (100, 100), 50, color3, -1)
            cv2.circle(your_frame, (100, 100 + 50*2 + 20), 50, color1, -1)
            cv2.circle(your_frame, (100, 100 + (50*2 + 20)*2), 50, color2, -1)
            final_frame = np.hstack((your_frame, frame))
            cv2.imshow("Webcam - YOLO Prediction Result", final_frame)
    
    # Exit loop if 'q' is pressed
    key = cv2.waitKey(1)
    if key == 113:  # 'q' key to exit
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
