from flask import Flask, Response, render_template
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 model
model = YOLO("yolov5s.pt")

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Error: Could not open camera.")

def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on the current frame
        results = model(frame)  # Perform inference using the frame as input

        # Get predictions (boxes, class labels, and confidences)
        boxes = results[0].boxes  # Detected bounding boxes
        labels = results[0].names  # Class names
        probs = boxes.conf  # Confidence scores

        # Draw bounding boxes if any objects are detected
        if len(boxes) > 0:
            for box, label, prob in zip(boxes.xyxy, boxes.cls, probs):
                x1, y1, x2, y2 = map(int, box.tolist())  # Get coordinates
                label_name = labels[int(label)]  # Get class name
                confidence = prob.item()  # Get confidence score

                # Draw the bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
                cv2.putText(frame, f'{label_name} {confidence:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame as part of a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # Render the index.html template
    
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    try:
        #app = Flask(__name__, template_folder="C:/Desktop/NRIIT/templates")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"An error occurred: {e}")
