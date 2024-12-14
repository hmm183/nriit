import re
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoTokenizer
from pymongo import MongoClient
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
dbinvo = client['invoice_data']  # Database name
collectioninvo = dbinvo['invoices']  # Collection name

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
ALLOWED_EXTENSIONS = {'pdf','jpg', 'jpeg', 'png', 'gif'}  # Define allowed extensions

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
        collectioninvo.insert_one(structured_data)
        print("Data saved to MongoDB successfully.")
    except Exception as e:
        print(f"Error saving data to MongoDB: {e}")

########################################medical##################################################
def ask_gpt_and_get_data(raw_data, clean_data):
    import google.generativeai as genai
    import os
    import sys
    import json
    import re 
    # Cleaned text from GPT model
    prompt = "Given the patient data " + raw_data + " more data in the reference of this in cleaner version " + clean_data + " parse the data in 3 parts first is patient_name second is diagnosis and third is medication if you are unable to find the data in the given text leave it as unknown and the format your answer in this form, Patient Name : (data) , Diagnosis :(data) , Prescription :(data) and have no newline characters in the answer as well have space after each word"
    

    genai.configure(api_key="AIzaSyDr_qZH6tVihVRBWIeEVvUlGlwjHCRLZQ8")
    model = genai.GenerativeModel("gemini-1.5-flash")
    # Generate content using GPT (Gemini)
    response = model.generate_content(prompt)
    recipe_text = response.text
    
    # Remove unwanted special characters
    cleaned_text = re.sub(r'[^\w\s,.!]', '', recipe_text)  # Retain letters, digits, spaces, punctuation
    
    return cleaned_text

# Step 1: Define text cleaning function
def clean_text(text):
    """
    Cleans clinical text by removing unnecessary symbols, normalizing whitespace, 
    and handling optional case normalization.
    """
    text = re.sub(r"[^a-zA-Z0-9\s.,'-]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()        # Normalize whitespace

    return text

# Step 2: Extract fields from clinical text
def extract_medical_fields(text):
    """
    Extracts important fields like patient name, diagnosis, and medication
    from the clinical text using regex.
    """
    # Example regex patterns (customize as needed)
    name_match = re.search(r"Patient Name\s*[:\-]?\s*(.*?)(?=,|Diagnosis|$)", text)
    diagnosis_match = re.search(r"Diagnosis\s*[:\-]?\s*(.*?)(?=,|Prescription|$)", text)
    medication_match = re.search(r"Prescription\s*[:\-]?\s*(.*?)(?=$|,)", text)
    
    # Extracted data
    return {
        "patient_name": name_match.group(1).strip() if name_match else "Unknown",
        "diagnosis": diagnosis_match.group(1).strip() if diagnosis_match else "Unknown",
        "medication": medication_match.group(1).strip() if medication_match else "Unknown",
    }


# Step 3: Process the document with OCR
def process_document(pdf_path):
    """
    Converts PDF pages to images, extracts text using OCR, and processes the text.
    """
    # Convert PDF to images (each page becomes an image)
    images = convert_from_path(pdf_path)

    # Extract text from each image
    extracted_text = ""
    for image in images:
        extracted_text += pytesseract.image_to_string(image)
    #print(extracted_text)
    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)
    #print(cleaned_text)
    # Extract fields from the text
    new_data=ask_gpt_and_get_data(extracted_text,cleaned_text)
    print(new_data)
    medical_data = extract_medical_fields(new_data)
    

    return medical_data

# Step 4: Insert data into MongoDB
def insert_into_mongodb(data):
    """
    Inserts processed medical data into MongoDB.
    """
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["medical_data"]
    collection = db["medical_records"]

    # Insert data
    insert_result = collection.insert_one(data)
    print("Inserted record ID:", insert_result.inserted_id)

    # Close the connection
    client.close()
###############################################ROUTES###############################################
    
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





# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['medical_data']  # Database for medical records
medical_collection = db['medical_records']  # Collection for medical records

# Define upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf','jpg', 'jpeg', 'png', 'gif'}  # Medical route only processes PDFs
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Medical Route

from bson import ObjectId

# Function to serialize MongoDB data to JSON
def serialize_mongodb_data(medical_data):
    # Convert ObjectId to string for JSON serialization
    if isinstance(medical_data, list):
        for item in medical_data:
            if '_id' in item:
                item['_id'] = str(item['_id'])
    elif isinstance(medical_data, dict):
        if '_id' in medical_data:
            medical_data['_id'] = str(medical_data['_id'])
    return medical_data

@app.route('/medical', methods=['GET', 'POST'])

def upload_medical():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                # Process the PDF using main.py's functions
                medical_data = process_document(file_path)
                
                # Insert data into MongoDB
                insert_into_mongodb(medical_data)

                # Serialize MongoDB data before sending as JSON
                medical_data = serialize_mongodb_data(medical_data)

                # Return the extracted data as JSON
                return jsonify({
                    "message": "Medical data processed successfully.",
                    "data": medical_data
                }), 200

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        else:
            return jsonify({"error": "Invalid file format. Only PDFs are allowed."}), 400

    else:
        # Handle GET request if needed
        return render_template('mediindex.html')

@app.route('/chatbot')
def chat():
    return render_template('chatbot.html')

db_medical = client['medical_data']  # Medical Database
collection_medical = db_medical['medical_records']  # Medical Collection

@app.route('/medical-chat', methods=['GET', 'POST'])
def medical_chat():
    response_text = None  # Initialize response_text for GET requests

    if request.method == 'POST':
        user_name = request.form.get('user_name').strip()

        # Check if the user exists in the medical records database
        user_data = collection_medical.find_one({"patient_name": {"$regex": f"^{user_name}$", "$options": "i"}})

        if user_data:
            # Extract relevant medical data
            medical_history = user_data.get("diagnosis", "No known medical history.")
            previous_diagnosis = user_data.get("previous_diagnosis", "None.")
            previous_treatments = user_data.get("previous_treatments", "None.")
            symptoms = user_data.get("symptoms", "None.")
        else:
            medical_history = "No medical history found."
            previous_diagnosis = "No previous diagnosis found."
            previous_treatments = "No previous treatments found."
            symptoms = "None."

        # Construct the prompt for Gemini based on user query and medical history
        prompt = f"""
        User: {user_name}
        Symptoms: {symptoms}
        Medical History: {medical_history}
        Previous Diagnosis: {previous_diagnosis}
        Previous Treatments: {previous_treatments}
        Given the symptoms and history, suggest a possible diagnosis and treatment plan.
        """
        genai.configure(api_key="AIzaSyDr_qZH6tVihVRBWIeEVvUlGlwjHCRLZQ8") # Ensure to replace with your Gemini API key
        # Get response from Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        response_text = re.sub(r'[^\w\s,.!]', '', response.text)  # Clean response

    return render_template('medical_chat.html', response_text=response_text)


predictions = []

@app.route('/real', methods=['GET', 'POST'])
def classi():
    global predictions

    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part", 400

        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        if file:
            image = Image.open(file)
            results = model(image)

            # Prepare predictions
            predictions = []
            for result in results:
                boxes = result.boxes  # Detected bounding boxes
                labels = result.names  # Detected class names
                probs = boxes.conf  # Confidence scores

                for box, label, prob in zip(boxes.xyxy, boxes.cls, probs):
                    predictions.append({
                        "object": labels[int(label)],
                        "confidence": prob.item(),
                        "bounding_box": box.tolist()
                    })

            return render_template('chat.html', predictions=predictions)

    # GET method - render upload form
    return render_template('upload.html')

@app.route('/chat', methods=['POST'])
def chating():
    global predictions

    user_query = request.json.get('query', '').lower()
    response = chatbot_response(user_query, predictions)
    return jsonify({"response": response})

def chatbot_response(query, predictions):
    if not predictions:
        return "No objects detected yet. Please upload an image first."

    if "object" in query or "objects" in query:
        objects = [prediction['object'] for prediction in predictions]
        return f"The objects detected in the image are: {', '.join(objects)}."

    if "where" in query:
        for prediction in predictions:
            if prediction['object'].lower() in query:
                location = prediction['bounding_box']
                return f"The {prediction['object']} is located at {location}."
        return "I'm sorry, I couldn't find that object in the image."

    if "confidence" in query:
        confidence_levels = []
        for prediction in predictions:
            confidence_levels.append(f"{prediction['object']}: {prediction['confidence']:.2f}")
        result = "The confidence levels of the objects are: " + ", ".join(confidence_levels)
        print(result)

        return result
    return "Sorry, I don't understand the question. Can you ask something else?"



if __name__ == '__main__':
    app.run(debug=True)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)


