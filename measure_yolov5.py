import faulthandler

faulthandler.enable()
#https://pysource.com
import cv2
import torch
from realsense_camera import *
import time
from fps import FPS

# Load Realsense camera
rs = RealsenseCamera()

def draw_object_info(bgr_frame, depth_frame, obj_boxes, obj_classes , obj_centers):
        # loop through the detection
        # print("object boxes: ", obj_boxes)
        # print("object classes: ", obj_classes)
        # print("object centers: ", obj_centers)
        # print("bgr frame: ", bgr_frame.shape)
        # print("depth frame: ", depth_frame.shape)
        for box, class_id, obj_center in zip(obj_boxes, obj_classes, obj_centers):
            x, y, x2, y2 = box
            

            color = (255, 0, 0)
            color = (int(color[0]), int(color[1]), int(color[2]))

            cx, cy = obj_center
            # print(cx, cy)
            # print(depth_frame.size)
            depth_mm = depth_frame[cy, cx] # SOMETHING WRONG HERE :((((
            
            cv2.line(bgr_frame, (cx, y), (cx, y2), color, 1)
            cv2.line(bgr_frame, (x, cy), (x2, cy), color, 1)

            class_name = class_id
            cv2.rectangle(bgr_frame, (x, y), (x + 250, y + 70), color, -1)
            cv2.putText(bgr_frame, class_name.capitalize(), (x + 5, y + 25), 0, 0.8, (255, 255, 255), 2)
            cv2.putText(bgr_frame, "{} cm".format(depth_mm / 10), (x + 5, y + 60), 0, 1.0, (255, 255, 255), 2)
            cv2.rectangle(bgr_frame, (x, y), (x2, y2), color, 1)
        return bgr_frame


print("Loading ...")
model = torch.hub.load('ultralytics/yolov5','custom', path='checkpoints/yolov5s.pt')
# yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5/yolov5s.pt')
print("Loaded YOLO...")
total_fps = FPS()
while True:
    # Get frame in real time from Realsense camera
    
    ret, bgr_frame, depth_frame = rs.get_frame_stream()
    bgr_src = bgr_frame.copy()
    # depth_src = depth_frame.copy()
    total_fps.start()
    result = model(bgr_src)
    classes =  list(result.pandas().xyxy[0]["name"])
    coordinates = result.xyxy[0].detach().cpu().numpy()
    centre_pts = []
    obj_coordinates = []

    for coord, cls in zip(coordinates, classes):
        # cv2.putText(bgr_src, cls, (int(coord[0]),int(coord[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        # cv2.rectangle(bgr_src, (int(coord[0]),int(coord[1])), (int(coord[2]),int(coord[3])),(255, 255, 255),2)
        xCenter = int(coord[0]) + int((int(coord[2]) - int(coord[0]))/2)
        yCenter = int(coord[1]) + int((int(coord[3]) - int(coord[1]))/2)
        # print("xcenter, ycenter: ",xCenter, yCenter)
        centre_pts.append((xCenter, yCenter))
        obj_coordinates.append([int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])])
    
	# Show depth info of the objects
	# mrcnn.draw_object_info(bgr_frame, depth_frame)
    bgr = draw_object_info(bgr_frame, depth_frame, obj_coordinates, classes, centre_pts)
    total_fps.stop()
    
    cv2.putText(bgr, f"FPS: {total_fps.getFPS():.2f}", (7,40), cv2.FONT_HERSHEY_COMPLEX, 1.4, (100, 255, 0), 3, cv2.LINE_AA)

# 	# cv2.imshow("depth frame", depth_frame)
    cv2.imshow("Bgr frame", bgr)
# 	cv2.imshow("yolov5", yolov5)

    key = cv2.waitKey(1)
    if key == 27:
	    break

rs.release()
cv2.destroyAllWindows()

