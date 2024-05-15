import streamlit as st
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("../../models/best_v8_19_03.pt")

# Title and description
st.title("Real-time Object Detection App")
st.write("Detecting objects in images and video stream...")

# Function to perform object detection on each frame
def detect_objects(frame):
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Perform object detection
    results = model.predict(frame_rgb, conf=0.1)
    result = results[0]
    # Convert the image from numpy array to a PIL Image
    pil_image = Image.fromarray(result.plot()[:, :, ::-1])
    return pil_image, result

# Function to read video stream from a file and display frames
def main(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video codec and create a VideoWriter object
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    # Read the video stream frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        detected_image, _ = detect_objects(frame)

        # Display the original frame and the frame with detections side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, caption="Original Frame", use_column_width=True)
        with col2:
            st.image(detected_image, caption="Detected Objects", use_column_width=True)
        
        # Convert PIL Image to numpy array for saving to video
        detected_frame_np = np.array(detected_image)[:, :, ::-1]

        # Write the frame to the output video
        out.write(detected_frame_np)

    # Release the video capture object and video writer
    cap.release()
    out.release()

# File upload section
uploaded_file = st.file_uploader("Upload image or video file", type=["mp4", "jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension in ["jpg", "jpeg", "png"]:
        # If uploaded file is an image
        image = Image.open(uploaded_file)
        detected_image, _ = detect_objects(np.array(image))
        
        # Display the original image and the image with detected objects side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(detected_image, caption="Detected objects in the uploaded image", use_column_width=True)
    elif file_extension == "mp4":
        # If uploaded file is a video
        video_bytes = uploaded_file.read()
        video_path = "temp.mp4"
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # Run the main function with the uploaded video file
        main(video_path)
