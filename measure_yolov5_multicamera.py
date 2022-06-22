import faulthandler

faulthandler.enable()
import cv2
import torch
from realsense_camera import *
import time
from fps import FPS
import math
import pyrealsense2 as rs
import numpy as np
import cv2
import logging
from draw_object_info import draw_object_info
from scipy.ndimage import rotate

# Configure depth and color streams...
# ...from Camera 1: FRONT
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('919122072891')
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ...from Camera 2: LEFT
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('919122073270')
config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ...from Camera 3: RIGHT
pipeline_3 = rs.pipeline()
config_3 = rs.config()
config_3.enable_device('112422070486')
config_3.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config_3.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming from both cameras
pipeline_1.start(config_1)
print("Started Pipeline 1 ... ...")
pipeline_2.start(config_2)
print("Started Pipeline 2 ... ...")
pipeline_3.start(config_3)
print("Started Pipeline 3 ... ...")

print("... ... ... ...")
model = torch.hub.load('ultralytics/yolov5','custom', path='checkpoints/yolov5s.pt')
print("Loaded YOLO...")
total_fps = FPS()

try:
    while True:

        # Camera 1: FRONT
        # Wait for a coherent pair of frames: depth and color
        front_frames = pipeline_1.wait_for_frames()
        front_depth_frame = front_frames.get_depth_frame()
        front_color_frame = front_frames.get_color_frame()
        if not front_depth_frame or not front_color_frame:
            continue
        # Convert images to numpy arrays
        front_depth_image = np.asanyarray(front_depth_frame.get_data())
        front_color_image = np.asanyarray(front_color_frame.get_data())
        
        # Camera 2: LEFT
        # Wait for a coherent pair of frames: depth and color
        left_frames = pipeline_2.wait_for_frames()
        left_depth_frame = left_frames.get_depth_frame()
        left_color_frame = left_frames.get_color_frame()
        if not left_depth_frame or not left_color_frame:
            continue
        # Convert images to numpy arrays
        left_depth_image = np.asanyarray(left_depth_frame.get_data())
        left_color_image = np.asanyarray(left_color_frame.get_data())
        left_depth_image = rotate(left_depth_image, 90, reshape=False)
        left_color_image = rotate(left_color_image, 90, reshape=False)

        # Camera 3: RIGHT
        # Wait for a coherent pair of frames: depth and color
        right_frames = pipeline_3.wait_for_frames()
        right_depth_frame = right_frames.get_depth_frame()
        right_color_frame = right_frames.get_color_frame()
        if not right_depth_frame or not right_color_frame:
            continue
        # Convert images to numpy arrays
        right_depth_image = np.asanyarray(right_depth_frame.get_data())
        right_color_image = np.asanyarray(right_color_frame.get_data())
        right_depth_image = rotate(right_depth_image, 90, reshape=False)
        right_color_image = rotate(right_color_image, 90, reshape=False)

        # Concat all the camera frames together
        concat_depth_image = np.concatenate((right_depth_image, front_depth_image, left_depth_image), axis = 1)
        concat_color_image = np.concatenate((right_color_image, front_color_image, left_color_image), axis = 1)

        total_fps.start()
        result = model(concat_color_image)
        classes =  list(result.pandas().xyxy[0]["name"])
        confidence = list(result.pandas().xyxy[0]["confidence"])
        coordinates = result.xyxy[0].detach().cpu().numpy()
        centre_pts = []
        obj_coordinates = []

        for coord, cls in zip(coordinates, classes, confidence):
            xCenter = int(coord[0]) + int((int(coord[2]) - int(coord[0]))/2)
            yCenter = int(coord[1]) + int((int(coord[3]) - int(coord[1]))/2)
        
            centre_pts.append((xCenter, yCenter))
            obj_coordinates.append([int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])])
        
        # Show depth info of the objects
        bgr = draw_object_info(concat_color_image, concat_depth_image, obj_coordinates, classes, centre_pts, confidence)
        total_fps.stop()

        cv2.putText(bgr, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

        # Show Images with Depth from all cameras
        cv2.imshow('With Depth', concat_color_image)

        # Show Actual images from all cameras
        front_color_rgb = cv2.cvtColor(front_color_image, cv2.COLOR_BGR2RGB)
        front_color = cv2.putText(front_color_rgb, 'FRONT CAM', (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) )

        left_color_image = cv2.cvtColor(left_color_image, cv2.COLOR_BGR2RGB)
        left_color = cv2.putText(left_color_image, 'LEFT CAM', (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) )

        right_color_image = cv2.cvtColor(right_color_image, cv2.COLOR_BGR2RGB)
        right_color = cv2.putText(right_color_image, 'RIGHT CAM', (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) )

        cv2.imshow('Actual', np.hstack((right_color, front_color, left_color)))
        cv2.waitKey(1)

        key = cv2.waitKey(1)
        if key == 27:
            break


finally:
    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()
    pipeline_3.stop()