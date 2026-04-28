import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
# Load model
models = YOLO("best.pt")
# model=YOLO('yolov8n.pt')
# Read image
img = cv2.imread("bicycle-1279907_1280.jpg")
# Run detection
results = models(img)[0]

for box in results.boxes:
    cls = int(box.cls[0])
    label = models.names[cls]
    conf = float(box.conf[0])

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    color=(0,255,0)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Show result
cv2.imshow("Helmet Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()