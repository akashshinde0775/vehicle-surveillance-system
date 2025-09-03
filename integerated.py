import cv2
import os
import csv
import json
import base64
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr

# Paths and Models
video_path = "input video3.mp4"
vehicle_model = YOLO("yolov8n.pt")  # Vehicle model
plate_model = YOLO("license_plate_detector.pt")  # License plate model
reader = easyocr.Reader(['en'], gpu=False)

# Setup folders
output_folder = "plates_output"
os.makedirs(output_folder, exist_ok=True)
json_output = []

# Deep SORT Tracker
tracker = DeepSort(max_age=30)

# Display constants
STANDARD_WIDTH = 1280
STANDARD_HEIGHT = 720
VEHICLE_CLASSES = ["car", "motorbike", "bus", "truck", "bicycle"]
COLORS = {
    "car": (0, 255, 0),
    "motorbike": (255, 0, 0),
    "bus": (0, 255, 255),
    "truck": (255, 255, 0),
    "bicycle": (255, 0, 255)
}
vehicle_data = {}

# Helper: Fancy green vehicle box with ID
def draw_fancy_box(img, x1, y1, x2, y2, color, label):
    thickness = 2
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness + 3)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, 1)
    label_y = y1 - text_height - 10 if y1 - text_height - 10 > 0 else y1 + text_height + 10
    bg_color = tuple(int(c * 0.6) for c in color)

    cv2.rectangle(img, (x1, label_y), (x1 + text_width + 10, label_y + text_height + baseline + 6), bg_color, -1)
    text_org = (x1 + 5, label_y + text_height + baseline)
    cv2.putText(img, label, text_org, font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label, text_org, font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (x1, y1), 6, color, -1)

# Helper: Red license plate box with OCR text
def draw_plate_style(frame, x1, y1, x2, y2, text):
    text = text.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    text_x = x1
    text_y = max(y1 - 10, 0)
    text_w = text_size[0] + 10
    text_h = text_size[1] + 10

    cv2.rectangle(frame, (text_x, text_y - text_h), (text_x + text_w, text_y), (255, 255, 255), -1)
    cv2.putText(frame, text, (text_x + 5, text_y - 5), font, font_scale, (0, 0, 0), thickness)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Helper: Encode image to base64
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Video stream
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (STANDARD_WIDTH, STANDARD_HEIGHT))

    # --- VEHICLE DETECTION & TRACKING ---
    vehicle_results = vehicle_model(frame)[0]
    vehicle_detections = []

    for box in vehicle_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        class_id = int(class_id)
        class_name = vehicle_model.names[class_id]

        if class_name in VEHICLE_CLASSES and score > 0.4:
            vehicle_detections.append([[x1, y1, x2 - x1, y2 - y1], score, class_name])

    tracks = tracker.update_tracks(vehicle_detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        vehicle_type = track.get_det_class()
        color = COLORS.get(vehicle_type, (255, 255, 255))
        label = f"{vehicle_type.capitalize()} ID:{track_id}"
        draw_fancy_box(frame, x1, y1, x2, y2, color, label)

        if vehicle_type not in vehicle_data:
            vehicle_data[vehicle_type] = set()
        vehicle_data[vehicle_type].add(track_id)

    # --- LICENSE PLATE DETECTION ---
    plate_results = plate_model(frame)[0]

    for box in plate_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_plate = frame[y1:y2, x1:x2]

        ocr_result = reader.readtext(cropped_plate)
        recognized_text = ocr_result[0][1] if ocr_result else ""

        draw_plate_style(frame, x1, y1, x2, y2, recognized_text)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        encoded_img = encode_image_to_base64(cropped_plate)

        json_output.append({
            "timestamp": timestamp,
            "license_plate_image": encoded_img,
            "text": recognized_text
        })

    cv2.imshow("Vehicle & License Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# --- SAVE OUTPUT FILES ---

# Save vehicle data to CSV
csv_path = os.path.abspath("vehicle_data.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Vehicle Type", "Count", "IDs"])
    for vehicle_type, ids in vehicle_data.items():
        writer.writerow([vehicle_type, len(ids), ", ".join(str(i) for i in sorted(ids))])
print(f"✅ Vehicle data saved to: {csv_path}")

# Save license plate JSON
with open("detected_plates.json", "w") as f:
    json.dump(json_output, f, indent=4)
print("✅ License plate data saved to: detected_plates.json")
