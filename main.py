from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
import cv2, os
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()



app = FastAPI()

API_KEY = os.getenv("API_KEY")

async def verify_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split("Bearer ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

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

@app.post("/detect/", summary="Detect Lane Type", dependencies=[Depends(verify_api_key)])
async def detect_objects(files: List[UploadFile] = File(...)):
    results_list = []

    for file in files:
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
                class_names = result.names
                if len(result.boxes) == 0:
                    lane_type = "no detection"
                    break
                for box in result.boxes:
                    [x1, y1, x2, y2] = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    x_center = (x1 + x2) / 2
                    class_name = class_names[cls]
                    if x_center < image_center_x:
                        class_name = "left_" + class_name
                    else:
                        class_name = "right_" + class_name
                    detected_lines.add(class_name)

                lane_type = check_lane_type(detected_lines)

            results_list.append({
                "file_name": file.filename,
                "lane": lane_type
            })

        except Exception as e:
            results_list.append({
                "file_name": file.filename,
                "lane": f"error: {str(e)}"
            })

    return results_list
