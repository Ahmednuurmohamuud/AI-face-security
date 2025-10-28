import cv2
import numpy as np
import os
import datetime
import sqlite3
import sys
import face_recognition

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
DETECTED_FACES_DIR = os.path.join(BASE_DIR, "detected_faces")
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "faces.db")

# --- Create required directories ---
os.makedirs(DETECTED_FACES_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# --- Load Known Faces ---
known_faces = []
known_names = []

if os.path.exists(KNOWN_FACES_DIR):
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            img = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_faces.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])

print(f"[INFO] Loaded {len(known_faces)} known faces.")

# --- Setup SQLite DB ---
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS face_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    timestamp TEXT
)
""")
conn.commit()

# --- Start Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam")
    sys.exit(1)

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Draw rectangle + name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save unknown face
        if name == "Unknown":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            face_img = frame[top:bottom, left:right]
            if face_img.size > 0:
                cv2.imwrite(f"{DETECTED_FACES_DIR}/face_{timestamp}.jpg", face_img)

        # Log to SQLite
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO face_logs (name, timestamp) VALUES (?, ?)", (name, timestamp))
        conn.commit()

    # Show frame
    cv2.imshow("Real-Time Face Recognition & Logging", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
