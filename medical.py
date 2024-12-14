import re
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoTokenizer
from pymongo import MongoClient

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



    
pdf_path=r"C:\Users\tiwar\OneDrive\Desktop\NRIIT Hackathon\clinical_bert\dr.pdf"
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

# Main function for demonstration
def main():
    # Path to the medical document (PDF)
    pdf_path = r"C:\Users\tiwar\OneDrive\Desktop\NRIIT Hackathon\clinical_bert\dr.pdf"  # Replace with the uploaded file path

    # Step 1: Process the document
    medical_data = process_document(pdf_path)

    # Step 2: Print extracted data
    print("Extracted Medical Data:", medical_data)

    # Step 3: Store data in MongoDB
    insert_into_mongodb(medical_data)

# Run the script
if _name_ == "_main_":
    main()
