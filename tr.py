from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os

# Initialize Flask app
app = Flask(__name__)

# Set up Gemini API Key (Ensure to replace with your actual API key)
GEMINI_API_KEY = "AIzaSyDr_qZH6tVihVRBWIeEVvUlGlwjHCRLZQ8"  # Replace with your actual API Key
#os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY  # Set Gemini API key as environment variable

# Initialize the Gemini API model (e.g., gemini-1.5-flash)
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route('/')
def home():
    return render_template('chatbot.html')  # Renders chatbot.html from templates folder

@app.route('/ask-chatbot', methods=['POST'])
def ask_chatbot():
    user_question = request.json.get('question')

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Use Gemini API to get an answer
        response = model.generate_content(user_question)

        # Get the response text (answer to the user's question)
        answer = response.text

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": f"Error from Gemini API: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
