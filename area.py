import cv2
import numpy as np

# Constants
THRESHOLD_VALUE = 200
MAX_PIXEL_VALUE = 255
MIN_AREA_THRESHOLD = 35000
MAX_AREA_THRESHOLD = 40000

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

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Extract flange region
    flange = frame[41:381, 91:592]
    
    # Convert to grayscale
    greyFlange = cv2.cvtColor(flange, cv2.COLOR_BGR2GRAY)
    
    # Thresholding
    _, threshold = cv2.threshold(greyFlange, THRESHOLD_VALUE, MAX_PIXEL_VALUE, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        # Draw rectangle if area is above a threshold
        if (area > MIN_AREA_THRESHOLD)and (area < MAX_AREA_THRESHOLD):
            cv2.rectangle(flange, (x, y), (x+w, y+h), (0, 255, 0), 3)
            print(area)
    
    # Display thresholded image and processed flange
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Grey Flange", greyFlange)
    cv2.imshow("Flange", flange)
    
    # Exit loop if 'q' is pressed
    key = cv2.waitKey(1)
    if key == 113:  # 'q' key to exit
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
