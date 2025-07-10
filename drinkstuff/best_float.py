import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from datetime import datetime

# Load your trained YOLOv8 model
model = YOLO("/Users/tarawang/Downloads/butlerbot/drinkstuff/best.pt")

# Class names from your training
class_names = model.model.names

print("Loaded YOLO model. Available drink classes:", list(class_names.values()))

# Video capture
cap = cv2.VideoCapture(0)
frame_queue = deque(maxlen=10)
font = cv2.FONT_HERSHEY_SIMPLEX

def get_side(bbox, width):
    x_center = (bbox[0] + bbox[2]) / 2
    return "left" if x_center < width / 2 else "right"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    frame_width = frame.shape[1]
    detections = []

    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = class_names[int(cls)]

        side = get_side((x1, y1, x2, y2), frame_width)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({side}) {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), font, 0.6, (0, 0, 0), 2)

        detections.append((label, side, conf, x1, y1))

    frame_queue.append(detections)

    # Voting logic for stability
    vote_counter = {}
    for det_list in frame_queue:
        for label, side, *_ in det_list:
            key = (label, side)
            vote_counter[key] = vote_counter.get(key, 0) + 1

    if vote_counter:
        best = max(vote_counter.items(), key=lambda x: x[1])
        label, side = best[0]
        print(f"[TRACKING] {label.upper()} on the {side.upper()} ({best[1]}/10 frames)")

    # Print current detections
    for label, side, conf, x1, y1 in detections:
        print(f"[{label}] on the {side} at ({x1}, {y1}) with {conf:.2f} confidence")

    cv2.imshow("All Beverage Detection", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        snap_path = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(snap_path, frame)
        print(f"[Snapshot saved] {snap_path}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
