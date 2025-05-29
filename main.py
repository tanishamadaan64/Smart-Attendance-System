import cv2
import os
import face_recognition
import pandas as pd
from datetime import datetime

# Paths̥
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.xlsx"

# Ensure folder exists
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Load or create attendance DataFrame
if os.path.exists(ATTENDANCE_FILE):
    attendance_df = pd.read_excel(ATTENDANCE_FILE)
else:
    attendance_df = pd.DataFrame(columns=["Name", "Timestamp"])

# Save attendance to file
def save_attendance():
    attendance_df.to_excel(ATTENDANCE_FILE, index=False)

# Load all known face encodings and names
def load_known_faces():
    known_encodings = []
    known_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
            else:
                print(f"[❌] No face found in {filename}")
    return known_encodings, known_names

# Register a new face
def register_face():
    cap = cv2.VideoCapture(0)
    name = input("Enter your name: ").strip()

    print("📸 Press 's' to capture and save your face, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if len(face_locations) != 1:
                print("❌ Please ensure only one face is visible.")
                continue

            face_image = frame
            filepath = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            cv2.imwrite(filepath, face_image)
            print(f"✅ Face saved as {filepath}")
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start attendance system
def start_attendance():
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        print("❌ No known faces found. Please register first.")
        return

    cap = cv2.VideoCapture(0)
    print("📷 Starting attendance. Press 'q' to quit.")

    already_marked = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if matches:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            y1, x2, y2, x1 = face_location
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if name != "Unknown" and name not in already_marked:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance_df.loc[len(attendance_df)] = [name, timestamp]
                print(f"✅ Marked present: {name} at {timestamp}")
                already_marked.add(name)
                save_attendance()

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main menu
def main():
    print("=== Face Recognition Attendance System ===")
    print("1. Register new face")
    print("2. Start attendance")
    print("3. Quit")
    choice = input("Choose an option (1/2/3): ").strip()

    if choice == '1':
        register_face()
    elif choice == '2':
        start_attendance()
    elif choice == '3':
        print("👋 Exiting. Goodbye!")
    else:
        print("Invalid choice. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()
