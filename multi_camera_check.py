import pyrealsense2 as rs
import numpy as np
import cv2
import logging


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

try:
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
        
        # Stack all images horizontally
        # images = np.hstack((color_image_1, depth_colormap_1,color_image_2, depth_colormap_2))

        # Show images from both cameras
        cv2.cvtColor(color_image_1, cv2.COLOR_BGR2RGB)
        cv2.imshow('test', color_image_1)
        cv2.imshow('front', color_image_2)
        cv2.waitKey(1)

        key = cv2.waitKey(1)
        if key == 27:
            break


finally:
    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()