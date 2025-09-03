import cv2
import easyocr
from ultralytics import YOLO

# === Try loading your custom model, fallback to YOLOv8n ===
custom_model_path = 'PlateDetectorNano.pt'
fallback_model_path = 'yolov8n.pt'  # fallback for testing
try:
    model = YOLO(custom_model_path)
    print(f"✅ Loaded model: {custom_model_path}")
except Exception as e:
    print(f"❌ Failed to load {custom_model_path}: {e}")
    print(f"🔁 Falling back to default: {fallback_model_path}")
    model = YOLO(fallback_model_path)

# === Initialize EasyOCR (CPU) ===
reader = easyocr.Reader(['en'], gpu=False)

# === Open video ===
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

# === Resize frames for consistent speed & view ===
STANDARD_WIDTH = 960
STANDARD_HEIGHT = 540

# === Visual Settings ===
YOLO_CONFIDENCE = 0.5
OCR_MIN_LENGTH = 4

def draw_fancy_plate(frame, x1, y1, x2, y2, plate_text):
    plate_text = plate_text.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2

    (tw, th), _ = cv2.getTextSize(plate_text, font, scale, thickness)
    padding = 8
    box_x1, box_y1 = x1, max(y1 - th - 15, 0)
    box_x2, box_y2 = x1 + tw + padding * 2, y1

    # Draw white box background
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)
    cv2.putText(frame, plate_text, (box_x1 + padding, box_y2 - 5), font, scale, (0, 0, 0), thickness)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

# === Frame Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (STANDARD_WIDTH, STANDARD_HEIGHT))

    # Inference
    results = model.predict(frame, conf=YOLO_CONFIDENCE, iou=0.4, imgsz=416, verbose=False)
    detections = results[0].boxes

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = frame[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue

        # OCR with EasyOCR
        ocr_result = reader.readtext(plate_crop)
        if ocr_result:
            plate_text = ocr_result[0][1]
            if len(plate_text.strip()) >= OCR_MIN_LENGTH:
                draw_fancy_plate(frame, x1, y1, x2, y2, plate_text)

    cv2.imshow("🚘 License Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
