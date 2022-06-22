import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import torch
from realsense_camera import *
from fps import FPS
import math


def draw_object_info(bgr_frame, depth_frame, obj_boxes, obj_classes , obj_centers):
        # loop through the detection
        for box, class_id, obj_center in zip(obj_boxes, obj_classes, obj_centers):
            x, y, x2, y2 = box
            

            color = (255, 0, 0)
            color = (int(color[0]), int(color[1]), int(color[2]))

            cx, cy = obj_center
            depth_mm = depth_frame[cy, cx] 
            
            cv2.line(bgr_frame, (cx, y), (cx, y2), color, 1)
            cv2.line(bgr_frame, (x, cy), (x2, cy), color, 1)

            class_name = class_id
            depth = (depth_mm / 10) * math.cos(math.radians(33.4)) 
            
            cv2.rectangle(bgr_frame, (x, y), (x + 250, y + 70), color, -1)
            cv2.putText(bgr_frame, class_name.capitalize(), (x + 5, y + 25), 0, 0.8, (255, 255, 255), 2)
            cv2.putText(bgr_frame, "{} cm".format(depth), (x + 5, y + 60), 0, 1.0, (255, 255, 255), 2)
            cv2.rectangle(bgr_frame, (x, y), (x2, y2), color, 1)
        return bgr_frame

# Configure depth and color streams...
# ...from Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('919122072891')
config_1.enable_stream(rs.stream.depth)
config_1.enable_stream(rs.stream.color)

# ...from Camera 2
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('020522070950')
config_2.enable_stream(rs.stream.depth)
config_2.enable_stream(rs.stream.color)


# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)

print("Loading ...")
model = torch.hub.load('ultralytics/yolov5','custom', path='checkpoints/yolov5s.pt')
print("Loaded YOLO...")
total_fps = FPS()
total_fps2 = FPS()


while True:
    # Camera 1
    # Wait for a coherent pair of frames: depth and color
    frames_1 = pipeline_1.wait_for_frames()
    depth_frame_1 = frames_1.get_depth_frame()
    color_frame_1 = frames_1.get_color_frame()
    if not depth_frame_1 or not color_frame_1:
        continue
    # Convert images to numpy arrays
    depth_image_1 = np.asanyarray(depth_frame_1.get_data())
    color_image_1 = np.asanyarray(color_frame_1.get_data())
    
    # Camera 2
    # Wait for a coherent pair of frames: depth and color
    frames_2 = pipeline_2.wait_for_frames()
    depth_frame_2 = frames_2.get_depth_frame()
    color_frame_2 = frames_2.get_color_frame()
    if not depth_frame_2 or not color_frame_2:
        continue
    # Convert images to numpy arrays
    depth_image_2 = np.asanyarray(depth_frame_2.get_data())
    color_image_2 = np.asanyarray(color_frame_2.get_data())
   
    color_image_concat = np.concatenate((color_image_1, color_image_2), axis=1)
    depth_image_concat = np.concatenate((depth_image_1, depth_image_2), axis=1)
    
    total_fps.start()
    result = model(color_image_concat)
    classes =  list(result.pandas().xyxy[0]["name"])
    coordinates = result.xyxy[0].detach().cpu().numpy()
    centre_pts = []
    obj_coordinates = []

    for coord, cls in zip(coordinates, classes):
        xCenter = int(coord[0]) + int((int(coord[2]) - int(coord[0]))/2)
        yCenter = int(coord[1]) + int((int(coord[3]) - int(coord[1]))/2)
    
        centre_pts.append((xCenter, yCenter))
        obj_coordinates.append([int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])])
    
	# Show depth info of the objects
    bgr = draw_object_info(color_image_concat, depth_image_concat, obj_coordinates, classes, centre_pts)
    total_fps.stop()
    
    cv2.putText(bgr, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

    # Show images from both cameras
    color_image_concat = cv2.cvtColor(color_image_concat, cv2.COLOR_BGR2RGB)
    color_image_2 = cv2.cvtColor(color_image_2, cv2.COLOR_BGR2RGB)
    cv2.imshow('top and front', color_image_concat)
    cv2.imshow('actual front', color_image_2)
    cv2.waitKey(1)

    key = cv2.waitKey(1)
    if key == 27:
        break



# Stop streaming
cv2.destroyAllWindows()
pipeline_1.stop()
pipeline_2.stop()
