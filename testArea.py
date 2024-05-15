import cv2
import numpy as np

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at pixel coordinates (x={x}, y={y})")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Frame", frame)
    
    # Set mouse callback function
    cv2.setMouseCallback('Frame', click_event)
    #x=91, y=68
    #x=592, y=353
    flange = frame[68:353,91:592]
    greyFlange = cv2.cvtColor(flange,cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(greyFlange, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        total_area += area

    # Print the total area
    print("Total area of object:", total_area)

    # Optionally, visualize the contours
    cv2.drawContours(flange, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', flange)
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     print(area)
    #     # (x,y,w,h) = cv2.boundingRect(cnt)
    #     # cv2.rectangle(flange,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("threshold",threshold)
    cv2.imshow("greyFlange",greyFlange)
    cv2.imshow("Flange",flange)
    key = cv2.waitKey(1)
    if key == 113:  # 'q' key to exit
        break

cap.release()
cv2.destroyAllWindows()
