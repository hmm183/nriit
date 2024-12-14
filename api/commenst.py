
'''
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

@app.route('/invoices')
def index():
    # Render the index.html template
    return render_template('invoindex.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, debug=True)

'''

'''
from flask import Flask, Response, render_template, request, jsonify
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import pytesseract
import google.generativeai as genai
import re
import pymongo
from pymongo import MongoClient
import os

# Initialize Flask app
app = Flask(__name__)

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# MongoDB setup (make sure MongoDB is running locally or use a remote DB URI)
client = MongoClient('mongodb://localhost:27017/')
db = client['invoice_data']  # Database name
collection = db['invoices']  # Collection name

# YOLOv5 model setup
model = YOLO("yolov5s.pt")

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Error: Could not open camera.")

def imToString(image_path):
    try:
        # Load the image directly
        image = cv2.imread(image_path)
        
        # Check if the image was loaded correctly
        if image is None:
            print("Error: Could not load the image.")
            return None
        
        # Convert the image to grayscale for better OCR results
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use OCR to extract text from the image
        data = pytesseract.image_to_string(img_gray, lang='eng')
        
        # Print the extracted text in the console
        if data.strip():
            print(f"Extracted text from the image:")
            print(data)
            return data
        else:
            print("No text found in the image.")
            return None
    
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None

def classify_invoice_data(extracted_text):
    genai.configure(api_key="AIzaSyDr_qZH6tVihVRBWIeEVvUlGlwjHCRLZQ8")

    # Set up the model
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Define the classification prompt
    prompt = "Classify this invoice data into three classes: Total Bill, Taxes, and Address. Ignore irrelevant details like emails and bank account information. Here is the data: " + extracted_text

    # Generate the classification response
    response = model.generate_content(prompt)
    recipe_text = response.text

    # Print the Gemini response (classification)
    print(f"Gemini Classification Response:\n{recipe_text}")

    # Clean up the response by removing unwanted characters
    cleaned_text = re.sub(r'[^\w\s,.!]', '', recipe_text)  # Retain letters, digits, spaces, punctuation

    # Remove email addresses (optional)
    cleaned_text = re.sub(r'\S+@\S+', '', cleaned_text)

    # Remove or adjust any bank account details or other unwanted sections
    cleaned_text = re.sub(r'ACC\s*#\s*\d+', '', cleaned_text)  # Remove bank account numbers
    cleaned_text = re.sub(r'BSB\s*#\s*\d+', '', cleaned_text)  # Remove BSB numbers

    # Extract the relevant structured data using regex or parsing (optional)
    # Here, we're just structuring the data manually for simplicity
    structured_data = {
        "total_bill": 93.50,  # You can extract this dynamically
        "taxes": 8.50,        # Extract taxes dynamically
        "address_from": "Suite 5A1204, 123 Somewhere Street, Your City AZ 12345",  # Extract dynamically
        "address_to": "123 Somewhere St, Melbourne, VIC 3000"  # Extract dynamically
    }

    # Insert into MongoDB
    try:
        collection.insert_one(structured_data)
        print("Data saved to MongoDB successfully.")
    except Exception as e:
        print(f"Error saving data to MongoDB: {e}")

@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

import os
from werkzeug.utils import secure_filename

# Define the UPLOAD_FOLDER globally or inside the route
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}  # Define allowed extensions

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/invoices', methods=['GET', 'POST'])
def invoices():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty part without a filename
        if file.filename == '':
            return 'No selected file'
        
        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            # Ensure the 'uploads' directory exists
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            
            # Secure the filename to prevent malicious files
            filename = secure_filename(file.filename)
            
            # Save the file with its original name in the 'uploads' folder
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            # Now you can use the image_path in the OCR process
            extracted_text = imToString(image_path)

            # Classify the extracted invoice data using the Gemini API
            if extracted_text:
                classify_invoice_data(extracted_text)

            return "File uploaded and processed successfully!"
        else:
            return 'Invalid file type. Only images are allowed.'

    return render_template('invoindex.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, debug=True)
'''
##################



'''
from flask import Flask, Response, render_template, request, jsonify
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import pytesseract
import google.generativeai as genai
import re
import pymongo
from pymongo import MongoClient
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# MongoDB setup (make sure MongoDB is running locally or use a remote DB URI)
client = MongoClient('mongodb://localhost:27017/')
db = client['invoice_data']  # Database name
collection = db['invoices']  # Collection name

# YOLOv5 model setup
model = YOLO("yolov5s.pt")

# Open the default camera
cap = cv2.VideoCapture(0)

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

if not cap.isOpened():
    raise Exception("Error: Could not open camera.")

# Define the UPLOAD_FOLDER globally or inside the route
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}  # Define allowed extensions

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def imToString(image_path):
    try:
        # Load the image directly
        image = cv2.imread(image_path)
        
        # Check if the image was loaded correctly
        if image is None:
            print("Error: Could not load the image.")
            return None
        
        # Convert the image to grayscale for better OCR results
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use OCR to extract text from the image
        data = pytesseract.image_to_string(img_gray, lang='eng')
        
        # Print the extracted text in the console
        if data.strip():
            print(f"Extracted text from the image:")
            print(data)
            return data
        else:
            print("No text found in the image.")
            return None
    
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None

def extract_invoice_data_from_ai(response_text):
    # Define a simple regex pattern to extract data between brackets
    pattern = r'\[([^\]]+)\]'
    
    # Find all matches for the pattern
    matches = re.findall(pattern, response_text)

    # Check if matches exist and store the data
    if len(matches) >= 4:
        total_bill, taxes, address_from, address_to = matches[:4]
    else:
        total_bill, taxes, address_from, address_to = "unknown", "unknown", "unknown", "unknown"

    return total_bill, taxes, address_from, address_to

def classify_invoice_data(extracted_text):
    genai.configure(api_key="AIzaSyDr_qZH6tVihVRBWIeEVvUlGlwjHCRLZQ8")

    # Set up the model
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Define the classification prompt (kept as is for accuracy)
    prompt = "Classify this invoice data into four classes: Total Bill, Taxes, Address To, and Address from. Ignore irrelevant details like emails and bank account information and just type the data you find from the text. Don’t include the class name inside the brackets. Separate the four classes' data and answer with brackets. If no data is relevant, insert 'unknown' in the brackets, nothing else besides that. Here is the data: " + extracted_text

    # Generate the classification response
    response = model.generate_content(prompt)
    recipe_text = response.text

    # Print the Gemini response (classification)
    print(f"Gemini Classification Response:\n{recipe_text}")

    # Extract structured data using simple regex
    total_bill, taxes, address_from, address_to = extract_invoice_data_from_ai(recipe_text)

    # Prepare structured data with dynamically extracted information inside brackets
    structured_data = {
        "total_bill": f"[{total_bill}]",
        "taxes": f"[{taxes}]",
        "address_from": f"[{address_from}]",
        "address_to": f"[{address_to}]"
    }
    print("Structured Data:", structured_data)
    
    # Insert into MongoDB
    try:
        collection.insert_one(structured_data)
        print("Data saved to MongoDB successfully.")
    except Exception as e:
        print(f"Error saving data to MongoDB: {e}")


@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/invoices', methods=['GET', 'POST'])
def invoices():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty part without a filename
        if file.filename == '':
            return 'No selected file'
        
        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            # Ensure the 'uploads' directory exists
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            
            # Secure the filename to prevent malicious files
            filename = secure_filename(file.filename)
            
            # Save the file with its original name in the 'uploads' folder
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            print(f"File saved at: {image_path}")  # Debugging log

            # Now you can use the image_path in the OCR process
            extracted_text = imToString(image_path)

            if extracted_text:
                # If text is extracted, classify the data
                classify_invoice_data(extracted_text)

            return "File uploaded and processed successfully!"
        else:
            return 'Invalid file type. Only images are allowed.'

    return render_template('invoindex.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, debug=True)

'''
################################



