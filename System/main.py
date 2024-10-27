import cv2
from deepface import DeepFace
import serial
import time
from pymongo import MongoClient
from datetime import datetime
import uuid
from gridfs import GridFS
import json
import os
import threading
from threading import Lock
from dotenv import load_dotenv

load_dotenv()

# Initialize video capture
video_capture = cv2.VideoCapture(0)
camera_lock = Lock()

output_filename = "recorded_video.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
fps = 20.0


# Function to get frame size
def get_frame_size(video_capture):
    return (int(video_capture.get(3)), int(video_capture.get(4)))


# Function to initialize VideoWriter
def initialize_video_writer(frame_size):
    return cv2.VideoWriter(output_filename, fourcc, fps, frame_size)


# Function to append existing video frames
def append_existing_video():
    if os.path.exists(output_filename):
        existing_video = cv2.VideoCapture(output_filename)
        frame_size = get_frame_size(existing_video)
        video_writer = initialize_video_writer(frame_size)

        while existing_video.isOpened():
            ret, frame = existing_video.read()
            if not ret:
                break
            video_writer.write(frame)

        existing_video.release()
        print("Appended existing video frames.")

    return initialize_video_writer(get_frame_size(video_capture))


# Function to capture and display video
def camera():
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return

    # Check if the video file already exists and append if it does
    video_writer = append_existing_video()

    while True:
        with camera_lock:
            ret, frame = video_capture.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Write the frame to the video file
        video_writer.write(frame)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    video_writer.release()  # Release the video writer when done
    cv2.destroyAllWindows()


# Start the camera thread
t1 = threading.Thread(target=camera)
t1.start()

SYSTEM_ID = None
if not os.path.exists("id.json"):
    new_uuid = str(uuid.uuid4())
    with open("id.json", "w") as file:
        json.dump({"uuid": new_uuid}, file)

with open("id.json", "r") as file:
    data = json.load(file)
    SYSTEM_ID = data["uuid"]


def capture_and_upload_image():
    with camera_lock:
        ret, frame = video_capture.read()

    if ret:
        _, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes()
    return None


def log_event_to_mongodb(event_type, message, user_name):
    image_data = capture_and_upload_image()
    image_id = str(uuid.uuid4())
    if image_data:
        log_entry = {
            "event_type": event_type,
            "message": message,
            "user_name": user_name,
            "image_id": image_id,
            "timestamp": str(datetime.now()),
            "systemID": SYSTEM_ID,
        }

        client = MongoClient(os.environ['MONGO'])
        db = client["cu_security_logs"]
        fs = GridFS(db)

        images_collection = db["images"]
        images_collection.insert_one(
            {
                "_id": image_id,
                "image_data": image_data,
                "filename": image_id + ".jpg",
                "uploaded_at": str(datetime.now()),
            }
        )

        logs_collection = db["logs"]
        logs_collection.insert_one(log_entry)
        print(f"Log entry added to MongoDB with image ID: {image_id}")


serialPort = "COM5"
baud_rate = 9600
ser = serial.Serial(serialPort, baud_rate)
time.sleep(2)


def recognise():
    with camera_lock:
        ret, frame = video_capture.read()

    if not ret:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    retVal = None

    try:
        res = DeepFace.extract_faces(rgb_frame, enforce_detection=False)
        r = DeepFace.find(rgb_frame, "./faces", "Dlib", enforce_detection=False)
        for result in res:
            x, y, w, h, left_eye, right_eye = result["facial_area"].values()
            if r and not r[0].empty:
                name = r[0]["identity"].iloc[0][8:-4]
                retVal = name

    except Exception as e:
        print(f"Error: {e}")

    return retVal


try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode("utf-8").strip()
            print("Received:", line)
            if line == "1":
                val = recognise()
                if val:
                    print(val)
                    password = input("Enter the pass code ")
                    log_event_to_mongodb("Person Detected", "", val)
                    if password == os.environ['PASS']:
                        ser.write("open".encode("utf-8"))
                else:
                    print("Unknown")
                    log_event_to_mongodb(
                        "Person Detected", "Unknown person spotted", "Unknown"
                    )
except KeyboardInterrupt:
    print("Closing connection...")
finally:
    ser.close()
    video_capture.release()
    cv2.destroyAllWindows()
