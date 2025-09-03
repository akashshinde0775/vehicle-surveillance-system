import cv2
import os
import csv
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr

# --- Paths and Models ---
video_path = "input video3.mp4"
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(['en'], gpu=False)

# --- Setup folders ---
output_folder = "plates_output"
os.makedirs(output_folder, exist_ok=True)

# --- Deep SORT Tracker ---
tracker = DeepSort(max_age=30)

# --- Display Constants ---
STANDARD_WIDTH = 1280
STANDARD_HEIGHT = 720
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck", "bicycle"]
COLORS = {
    "car": (0, 255, 0),
    "motorcycle": (255, 0, 0),
    "bus": (0, 255, 255),
    "truck": (255, 255, 0),
    "bicycle": (255, 0, 255)
}
vehicle_data = {}

# --- Helper Functions ---
def draw_fancy_box(img, x1, y1, x2, y2, color, label):
    thickness = 2
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness + 3)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), base = cv2.getTextSize(label, font, font_scale, 1)
    label_y = y1 - th - 10 if y1 - th - 10 > 0 else y1 + th + 10
    bg_color = tuple(int(c * 0.6) for c in color)
    cv2.rectangle(img, (x1, label_y), (x1 + tw + 10, label_y + th + base + 6), bg_color, -1)
    text_org = (x1 + 5, label_y + th + base)
    cv2.putText(img, label, text_org, font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label, text_org, font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (x1, y1), 6, color, -1)

def draw_plate_style(frame, x1, y1, x2, y2, text):
    text = text.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = x1, max(y1 - 10, 0)
    text_w, text_h = text_size[0] + 10, text_size[1] + 10
    cv2.rectangle(frame, (text_x, text_y - text_h), (text_x + text_w, text_y), (255, 255, 255), -1)
    cv2.putText(frame, text, (text_x + 5, text_y - 5), font, font_scale, (0, 0, 0), thickness)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

def draw_vehicle_counts(frame, vehicle_data, COLORS):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    y_offset = 40
    bg_color = (30, 30, 30)
    start_x, start_y = 20, 20
    box_height = (len(vehicle_data) + 1) * y_offset
    box_width = 220
    cv2.rectangle(frame, (start_x - 5, start_y - 5),
                  (start_x + box_width, start_y + box_height), bg_color, -1)
    cv2.putText(frame, "Vehicle Counts", (start_x, start_y + 15), font, 0.7, (255, 255, 255), 2)
    for idx, (vehicle_type, ids) in enumerate(vehicle_data.items()):
        color = COLORS.get(vehicle_type, (255, 255, 255))
        text = f"{vehicle_type.capitalize()}: {len(ids)}"
        y = start_y + (idx + 1) * y_offset
        cv2.putText(frame, text, (start_x, y), font, font_scale, color, 2)

# --- Initialize Video ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0.0:
    fps = 25  # fallback default
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎥 Total Frames in Input Video: {total_frames}")
print(f"⏱️ Detected FPS: {fps}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("processed_output.mp4", fourcc, fps, (STANDARD_WIDTH, STANDARD_HEIGHT))

# --- Frame Processing ---
frame_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (STANDARD_WIDTH, STANDARD_HEIGHT))

    # --- Vehicle Detection ---
    vehicle_results = vehicle_model(frame)[0]
    vehicle_detections = []
    for box in vehicle_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        class_id = int(class_id)
        class_name = vehicle_model.names[class_id]
        if class_name in VEHICLE_CLASSES and score > 0.4:
            vehicle_detections.append([[x1, y1, x2 - x1, y2 - y1], score, class_name])

    # --- Tracking ---
    tracks = tracker.update_tracks(vehicle_detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        vehicle_type = track.get_det_class()
        color = COLORS.get(vehicle_type, (255, 255, 255))
        label = f"{vehicle_type.capitalize()} ID:{track_id}"
        draw_fancy_box(frame, x1, y1, x2, y2, color, label)
        if vehicle_type not in vehicle_data:
            vehicle_data[vehicle_type] = set()
        vehicle_data[vehicle_type].add(track_id)

    # --- License Plate Detection and Save Full Vehicle Image ---
    plate_results = plate_model(frame)[0]
    for i, box in enumerate(plate_results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_plate = frame[y1:y2, x1:x2]
        ocr_result = reader.readtext(cropped_plate)
        recognized_text = ocr_result[0][1] if ocr_result else ""
        draw_plate_style(frame, x1, y1, x2, y2, recognized_text)

        for track in tracks:
            if not track.is_confirmed():
                continue
            vx1, vy1, vx2, vy2 = map(int, track.to_ltrb())
            if x1 > vx1 and y1 > vy1 and x2 < vx2 and y2 < vy2:
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                vehicle_type = track.get_det_class()
                track_id = track.track_id
                plate_clean = recognized_text.replace(' ', '_') if recognized_text else "plate"
                filename = f"{vehicle_type}_{plate_clean}_{track_id}_{timestamp}.jpg"
                if vehicle_crop.size > 0 and vehicle_crop.shape[0] > 0 and vehicle_crop.shape[1] > 0:
                    cv2.imwrite(os.path.join(output_folder, filename), vehicle_crop)
                else:
                    print(f"⚠️ Skipped saving vehicle crop for {filename}: empty or invalid image.")

                break

    # --- Draw Counts and Save ---
    draw_vehicle_counts(frame, vehicle_data, COLORS)
    out.write(frame)
    frame_counter += 1

    # Display
    cv2.imshow("Vehicle & License Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()

# --- Save CSV ---
csv_path = os.path.abspath("vehicle_data.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Vehicle Type", "Count", "IDs"])
    for vehicle_type, ids in vehicle_data.items():
        writer.writerow([vehicle_type, len(ids), ", ".join(str(i) for i in sorted(ids))])

print(f"\n✅ Processed frames: {frame_counter}")
print(f"✅ Vehicle data saved to: {csv_path}")
print(f"✅ Full vehicle images saved to: {output_folder}")
print("✅ Processed video saved as: processed_output.mp4")
