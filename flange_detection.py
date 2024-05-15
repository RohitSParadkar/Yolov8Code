#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:24:18 2024

@author: Ajay Suryawanshi
"""

import cv2
import numpy as np
import json
import os 
from os import system, listdir
import logging
from logging import handlers

def read_configuration():
    """read configuration file to get configuration data"""

    try:
        global config
        with open("/home/admin1/technosoft_dev/Flange_object_detction/config.json") as f:
            config = json.load(f)
            print(config)
        return config
    except Exception as e:
        print(f"Not able to raead configuration file please check | ERROR while reading the file : {e}")
        
        
def get_logger():
    global logger
    try:
        if not ("Service_flange_detection" in listdir("/var/log/")):
            os.chmod("/var/log/Service_flange_detection", 0o777)

            system("mkdir /var/log/Service_flange_detection")
    except:
        print("getting error while making directory Object_etection in /var/log/")
    
   
    logging_level = config["logging_level"]
    logger=logging.getLogger(__name__)
    if not logger.hasHandlers():        
        log_file_name = "/var/log/Service_flange_detection/object_detection.log"
        formatter = logging.Formatter('[%(asctime)s]  [%(levelname)s]  [%(funcName)s]  [ %(message)s ]')
        handler = handlers.TimedRotatingFileHandler(log_file_name, when="midnight", interval=1, backupCount=3)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging_level)
    logger.info("*********************************************************")
    return logger
            

def filter_silver_gray_color(image_path, tolerance_saturation=225, tolerance_value=30):
    """
    Filters silver-gray color in an image using HSV color space and adds contours.

    Args:
        image_path (str): Path to the image file.
        tolerance_saturation (int, optional): Tolerance around the target Saturation. Defaults to 175.
        tolerance_value (int, optional): Tolerance around the target Value. Defaults to 30.

    Returns:
        tuple: (original_image, mask, contours_image, roi_image)
        
    """
    try:
        
        image = image_path
        image_height, image_width, _ = image.shape
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_limit = np.array([0, 0, 30 - tolerance_value], dtype="uint8")
        upper_limit = np.array([180, tolerance_saturation, 135], dtype="uint8")
        mask = cv2.inRange(hsv, lower_limit, upper_limit)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_image = np.zeros_like(image)
        # cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
        roi_image = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            area = w*h
            print(x,y,x + w, y + h,w,h)
            if x == 0 or x + w == image_width or y == 0 or y + h == image_height:
                infer = False
                print(f"Bounding box touches image boundary. and area of the object is: - {w*h}")
            else:
                infer = True
                
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
            roi_image = image[y:y + h, x:x + w]
    
        return image, mask, contours_image, roi_image, area, infer
    
    except Exception as e:
        print(f"Got Error while proccess the frames error is: - {e}")



def infer_on_video(video_path, full_flange_folder, half_flange_folder):
    os.makedirs(full_flange_folder, exist_ok=True)
    os.makedirs(half_flange_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            try:
                frame_number += 1
                _, _, _, _, area, infer = filter_silver_gray_color(frame)
                
                if infer:
                    # # if 210000<= area <= 250000:

                    if area > 110000:
                        cv2.imwrite(os.path.join(full_flange_folder, f"full_object_{frame_number}.jpg"), frame)
                else:
                    cv2.imwrite(os.path.join(half_flange_folder, f"half_object_{frame_number}.jpg"), frame)
    
            except Exception as e:
                print(f"Got exception in main process. Error is: {e}")
                
    cap.release()
    cv2.destroyAllWindows()

            
if __name__ == "__main__":
    # config = read_configuration()
    # logger = get_logger() 
    infer_on_video("/home/admin1/Downloads/web 1.mp4", '/home/admin1/technosoft_dev/Flange_object_detction/full_objects','/home/admin1/technosoft_dev/Flange_object_detction/half_objects')
