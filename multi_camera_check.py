import pyrealsense2 as rs
import numpy as np
import cv2
import logging


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

try:
    while True:

        # Camera 1: FRONT
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        front_depth_frame_1 = frames_1.get_depth_frame()
        front_color_frame_1 = frames_1.get_color_frame()
        if not front_depth_frame_1 or not front_color_frame_1:
            continue
        # Convert images to numpy arrays
        front_depth_image_1 = np.asanyarray(front_depth_frame_1.get_data())
        front_color_image_1 = np.asanyarray(front_color_frame_1.get_data())
        
        # Camera 2: Left
        # Wait for a coherent pair of frames: depth and color
        left_frames_2 = pipeline_2.wait_for_frames()
        left_depth_frame_2 = left_frames_2.get_depth_frame()
        left_color_frame_2 = left_frames_2.get_color_frame()
        if not left_depth_frame_2 or not left_color_frame_2:
            continue
        # Convert images to numpy arrays
        left_depth_image_2 = np.asanyarray(left_depth_frame_2.get_data())
        left_color_image_2 = np.asanyarray(left_color_frame_2.get_data())

        # Camera 3: RIGHT
        # Wait for a coherent pair of frames: depth and color
        right_frames_1 = pipeline_3.wait_for_frames()
        right_depth_frame_1 = right_frames_1.get_depth_frame()
        right_color_frame_1 = right_frames_1.get_color_frame()
        if not right_depth_frame_1 or not right_color_frame_1:
            continue
        # Convert images to numpy arrays
        right_depth_image_1 = np.asanyarray(right_depth_frame_1.get_data())
        right_color_image_1 = np.asanyarray(right_color_frame_1.get_data())


        # Show images from all cameras
        front_color_image_1 = cv2.cvtColor(front_color_image_1, cv2.COLOR_BGR2RGB)
        front_color = cv2.putText(front_color_image_1, 'FRONT CAM', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0) )

        left_color_image_2 = cv2.cvtColor(left_color_image_2, cv2.COLOR_BGR2RGB)
        left_color = cv2.putText(left_color_image_2, 'LEFT CAM', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0) )

        right_color_image_1 = cv2.cvtColor(right_color_image_1, cv2.COLOR_BGR2RGB)
        right_color = cv2.putText(right_color_image_1, 'RIGHT CAM', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0) )

        cv2.imshow('multi-camera', np.hstack((right_color, front_color, left_color)))
        cv2.waitKey(1)

        key = cv2.waitKey(1)
        if key == 27:
            break


finally:
    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()
    pipeline_3.stop()