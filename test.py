import cv2
from ultralytics import YOLO
from crop import crop 
from align import align_face
from embeddings import get_embedding
import faiss
import numpy as np
import os

model = YOLO("best.pt")
cap = cv2.VideoCapture("Screen Recording 2026-04-27 180119.mp4")

# 🔥 Create folder
SAVE_DIR = "illegal_no_helmet"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔥 Initialize FAISS
dimension = 128
index = faiss.IndexFlatL2(dimension)

THRESHOLD = 0.6
person_id = 0


def built_faiss(embedding, image):
    global person_id, index

    # 🔹 Normalize
    embedding = embedding.astype('float32')
    embedding = embedding / np.linalg.norm(embedding)
    embedding = embedding.reshape(1, -1)

    # 🔴 Case 1: First entry
    if index.ntotal == 0:
        print("First illegal person stored")

        index.add(embedding)

        cv2.imwrite(f"{SAVE_DIR}/person_{person_id}.jpg", image)
        person_id += 1
        return

    # 🔴 Case 2: Search
    D, I = index.search(embedding, 1)
    dist = D[0][0]

    print("Distance:", dist)

    if dist < THRESHOLD:
        print("Same person → ignore")
    else:
        print("New illegal person → store")

        index.add(embedding)

        cv2.imwrite(f"{SAVE_DIR}/person_{person_id}.jpg", image)
        person_id += 1


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls.item())
        label = results.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "no-helmet":
            print("⚠️ NO HELMET")

            cropped_face = crop(frame[y1:y2, x1:x2])
            aligned = align_face(cropped_face)
            embedding = get_embedding(aligned)

            built_faiss(embedding, cropped_face)

            cv2.imshow("crop", cropped_face)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()