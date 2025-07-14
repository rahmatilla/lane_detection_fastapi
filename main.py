import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi import HTTPException
app = FastAPI()

LANE_RULES = {
    "left_lane": {
        "left": {"left_solid_white", "left_solid_yellow", "left_double_solid_white", "left_double_solid_yellow"},
        "right": {"right_broken_white", "right_broken_yellow"}
    },
    "middle_lane": {
        "left": {"left_broken_white", "left_broken_yellow"},
        "right": {"right_broken_white", "right_broken_yellow"}
    },
    "right_lane": {
        "left": {"left_broken_white", "left_broken_yellow"},
        "right": {"right_solid_white", "right_solid_yellow"}
    }
}

def check_lane_type(detected_lines: set) -> str:
    for lane, rules in LANE_RULES.items():
        if any(left in detected_lines for left in rules['left']) and any(right in detected_lines for right in rules['right']):
            return lane
    return "Lane type not detected."

model = YOLO("lane_line_detection_v1.pt")

@app.post("/detect/")
async def detect_objects(file: UploadFile):
    try:
        image_bytes = await file.read()
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Uploaded file is not a valid image.")

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
                if x_center < image_center_x:
                    class_name = "left_" + class_name
                else:
                    class_name = "right_" + class_name
                detected_lines.add(class_name)

        return {"lane": check_lane_type(detected_lines)}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")