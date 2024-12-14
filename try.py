import pytesseract
import cv2
import google.generativeai as genai
import re
import pymongo
from pymongo import MongoClient

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['invoice_data']  # Database name
collection = db['invoices']  # Collection name

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

# Image path you want to test
image_path = "inf.jpg"

# Get the extracted text from the image
extracted_text = imToString(image_path)

# Classify the extracted invoice data using the AI model
if extracted_text:
    classify_invoice_data(extracted_text)
