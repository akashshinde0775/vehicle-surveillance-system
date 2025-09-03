import cv2
import os
import csv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model for vehicle detection
vehicle_model = YOLO("yolov8n.pt")

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Vehicle classes to detect
VEHICLE_CLASSES = ["car", "motorbike", "bus", "truck", "bicycle"]

# Assign unique colors for different vehicle types
COLORS = {
    "car": (0, 255, 0),
    "motorbike": (255, 0, 0),
    "bus": (0, 255, 255),
    "truck": (255, 255, 0),
    "bicycle": (255, 0, 255)
}

# Dictionary to hold tracked vehicle data
vehicle_data = {}

# Open video file or webcam
video_path = "sample.mp4"  # Use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Define standard display resolution
STANDARD_WIDTH = 1280
STANDARD_HEIGHT = 720

# Draw fancy bounding box with background label
def draw_fancy_box(img, x1, y1, x2, y2, color, label):
    thickness = 2
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw outer box (shadow)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness + 3)

    # Draw inner colored bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Prepare label text
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, 1)
    label_y = y1 - text_height - 10 if y1 - text_height - 10 > 0 else y1 + text_height + 10

    # Draw filled rectangle for label background (darker version of bounding box color)
    bg_color = tuple(int(c * 0.6) for c in color)
    cv2.rectangle(
        img,
        (x1, label_y),
        (x1 + text_width + 10, label_y + text_height + baseline + 6),
        bg_color,
        -1
    )

    # Put white text with black outline (shadow effect)
    text_org = (x1 + 5, label_y + text_height + baseline)
    cv2.putText(img, label, text_org, font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)  # Shadow
    cv2.putText(img, label, text_org, font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)  # Text

    # Circle dot at top-left
    cv2.circle(img, (x1, y1), 6, color, -1)

# Processing video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (STANDARD_WIDTH, STANDARD_HEIGHT))

    # Vehicle Detection
    vehicle_results = vehicle_model(frame)[0]
    vehicle_detections = []

    for box in vehicle_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        class_id = int(class_id)
        class_name = vehicle_model.names[class_id]

        if class_name in VEHICLE_CLASSES and score > 0.4:
            vehicle_detections.append([[x1, y1, x2 - x1, y2 - y1], score, class_name])

    # Tracking
    tracks = tracker.update_tracks(vehicle_detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        vehicle_type = track.get_det_class()
        color = COLORS.get(vehicle_type, (255, 255, 255))  # default white

        label = f"{vehicle_type.capitalize()} ID:{track_id}"
        draw_fancy_box(frame, x1, y1, x2, y2, color, label)

        # Update vehicle data
        if vehicle_type not in vehicle_data:
            vehicle_data[vehicle_type] = set()
        vehicle_data[vehicle_type].add(track_id)

    # Show frame
    cv2.imshow("Vehicle Detection & Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Save CSV
csv_path = os.path.abspath("vehicle_data.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Vehicle Type", "Count", "IDs"])
    for vehicle_type, ids in vehicle_data.items():
        writer.writerow([vehicle_type, len(ids), ", ".join(str(i) for i in sorted(ids))])

print(f"\n✅ Vehicle data saved to: {csv_path}")
