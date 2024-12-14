import google.generativeai as genai
import os
import json
import re
from pymongo import MongoClient

# Configure Gemini API key
genai.configure(api_key="AIzaSyDr_qZH6tVihVRBWIeEVvUlGlwjHCRLZQ8") # Ensure to replace with your Gemini API key

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
db = client["medical_data"]  # Database name
collection = db["medical_records"]  # Collection name

# Set up the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Take user input for their name
user_name = input("Please enter your name: ").strip()  # Strip spaces for better matching

# Debug: Print the name entered by user
print(f"User Name Entered: {user_name}")

# Check if the name exists in the database (case insensitive search)
user_data = collection.find_one({"patient_name": {"$regex": f"^{user_name}$", "$options": "i"}})

# Debug: Check if any user data is found
if user_data:
    print(f"User Data Found: {user_data}")
else:
    print("No user data found.")

# Retrieve medical history, diagnosis, treatments, symptoms
if user_data:
    medical_history = user_data.get("diagnosis", "No known medical history.")
    previous_diagnosis = user_data.get("previous_diagnosis", "None.")
    previous_treatments = user_data.get("previous_treatments", "None.")
    symptoms = user_data.get("symptoms", "None.")
else:
    medical_history = "No medical history found."
    previous_diagnosis = "No previous diagnosis found."
    previous_treatments = "No previous treatments found."
    symptoms = "None."

# Display the user's medical information (for debugging or confirmation)
print(f"Medical History: {medical_history}")
print(f"Previous Diagnosis: {previous_diagnosis}")
print(f"Previous Treatments: {previous_treatments}")
print(f"Symptoms: {symptoms}")

# Construct the prompt for Gemini based on user query and medical history
prompt = f"""
User: {user_name}
Symptoms: {symptoms}
Medical History: {medical_history}
Previous Diagnosis: {previous_diagnosis}
Previous Treatments: {previous_treatments}
Given the symptoms and history, suggest a possible diagnosis and treatment plan.
"""

# Generate the response from Gemini API
response = model.generate_content(prompt)

# Get the response text (diagnosis, treatments, etc.)
response_text = response.text

# Clean the response (optional: remove unwanted characters or fix formatting)
cleaned_text = re.sub(r'[^\w\s,.!]', '', response_text)  # Retain letters, digits, spaces, punctuation

# Print the cleaned response as JSON
print(json.dumps({"response": cleaned_text}))
