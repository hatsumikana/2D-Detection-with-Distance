import cv2
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
            depth = (depth_mm / 10)
            
            cv2.rectangle(bgr_frame, (x, y), (x + 250, y + 70), color, -1)
            cv2.putText(bgr_frame, class_name.capitalize(), (x + 5, y + 25), 0, 0.8, (255, 255, 255), 2)
            cv2.putText(bgr_frame, "{} cm".format(depth), (x + 5, y + 60), 0, 1.0, (255, 255, 255), 2)
            cv2.rectangle(bgr_frame, (x, y), (x2, y2), color, 1)
        return bgr_frame