import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi import File, UploadFile
app = FastAPI()

def check_to_left_lane(detected_lines: set) -> bool:
    left_options = {
        "left_solid_white",
        "left_solid_yellow",
        "left_double_solid_white",
        "left_double_solid_yellow"
    }

    right_options = {
        "right_broken_white",
        "right_broken_yellow"
    }

    return any(left in detected_lines for left in left_options) and \
           any(right in detected_lines for right in right_options)

def check_to_middle_lane(detected_lines: set) -> bool:
    left_options ={
        "left_broken_white",
        "left_broken_yellow"
    }

    right_options = {
        "right_broken_white",
        "right_broken_yellow"
    }

    return any(left in detected_lines for left in left_options) and \
           any(right in detected_lines for right in right_options)

def check_to_right_lane(detected_lines: set) -> bool:
    left_options ={
        "left_broken_white",
        "left_broken_yellow"
    }

    right_options = {
        "right_solid_white",
        "right_solid_yellow"
    }

    return any(left in detected_lines for left in left_options) and \
           any(right in detected_lines for right in right_options)

model = YOLO("lane_line_detection_v1.pt")

@app.post("/detect/")
async def detect_objects(file: UploadFile):

    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    width = image.shape[1]
    image_center_x = width // 2
    detected_lines = set()
    
    results = model.predict(image)
    for result in results:
        classes_names = result.names
        if len(result.boxes) == 0:
            return {"lane": "no detection"}
        for box in result.boxes:
            [x1, y1, x2, y2] = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            x_center = (x1 + x2) / 2
            class_name = classes_names[cls]
            if  x_center < image_center_x:
                class_name = "left_"+class_name
            else:
                class_name = "right_"+class_name
            detected_lines.add(class_name)
    if check_to_left_lane(detected_lines):
        return {"lane": "left lane"}
    elif check_to_middle_lane(detected_lines):
        return {"lane": "middle lane"}
    elif check_to_right_lane(detected_lines):
        return {"lane": "right lane"}
    else:
        return {"lane": "lane type not detected"}