import os
import json
import glob
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 1. SETUP & CONFIGURATION ---

load_dotenv() # Loads .env locally (Render will use Environment Variables)

app = Flask(__name__)
CORS(app) # Enable Cross-Origin requests

# Initialize Gemini Client with Safety Checks
client = None  # Define it first so the app doesn't crash if init fails

try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("CRITICAL ERROR: 'GOOGLE_API_KEY' is missing. Chat will not work.")
    else:
        client = genai.Client(api_key=api_key)
        print("Gemini Client initialized successfully.")

except Exception as e:
    print(f"Error initializing Gemini Client: {e}")

# Use the stable model name. If you have specific access to 2.0, change this back.
# 'gemini-1.5-flash' is the safest, fastest option for public use right now.
MODEL_NAME = 'gemini-1.5-flash' 

# --- 2. DATA LOADING FUNCTIONS ---

def load_and_parse_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        parts = [f"=== Interview File: {os.path.basename(file_path)} ==="]
        for entry in data:
            # Handle list of dicts structure common in interview files
            if isinstance(entry, dict):
                speaker = entry.get("speaker", "Unknown")
                question = entry.get("question", "")
                answer = entry.get("answer", "")
                parts.append(f"\n{speaker}:\nQ: {question}\nA: {answer}")
        return "\n".join(parts)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def load_survey_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        parts = [f"=== Survey Analytics File: {os.path.basename(file_path)} ==="]
        
        if "survey_summary" in data:
            parts.append("\n--- SURVEY SUMMARY ---")
            for section, values in data["survey_summary"].items():
                parts.append(f"\n[{section.upper()}]")
                parts.append(json.dumps(values, indent=2))

        if "free_text_insights" in data:
            parts.append("\n--- FREE TEXT INSIGHTS ---")
            for question, summary in data["free_text_insights"].items():
                parts.append(f"\nQ: {question}\nSummary: {summary}")
        return "\n".join(parts)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def load_context():
    """
    Loads all JSON context on server startup.
    """
    directory = os.getcwd() # Looks in the current folder (Root of your repo)
    json_paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    parts = []
    
    print(f"Scanning directory for JSON files: {directory}")
    
    for path in json_paths:
        # Skip package files or non-data files
        if "package" in path or "lock" in path: 
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Detect type of JSON and parse accordingly
            if isinstance(data, list):
                # Usually interview transcripts are lists
                parts.append(load_and_parse_json(path))
            elif isinstance(data, dict) and ("survey_summary" in data or "free_text_insights" in data):
                # Survey data structure
                parts.append(load_survey_json(path))
        except Exception as e:
            print(f"Skipping file {path} due to load error: {e}")
            continue
                
    print(f"Loaded {len(parts)} JSON context blocks.")
    return "\n\n".join(parts)

# Load context ONCE when app starts
FULL_INTERVIEW_CONTEXT = load_context()

SYSTEM_INSTRUCTION = f"""
You are an analytical assistant, built to answer questions that cafe entreprenuers have when starting a new cafe. 
You DO NOT just repeat or summarize the context provided.

Your goals:
1. Use the interview transcript contexts as background knowledge.
2. Use the survey analytics as a customer perspective to cafe-going.
3. Use both transcripts and survey to give holistic answers.
4. Think beyond explicit text.
5. Infer patterns, motives, insights, and deeper meanings.
6. Provide thoughtful, evaluative, and analytical answers.
7. If the user asks about something subjective (e.g., fonts, design decisions), use the context to think of an answer.

Be concise, analytical, and insight-driven.

--- INTERVIEW CONTEXT START ---
{FULL_INTERVIEW_CONTEXT}
--- INTERVIEW CONTEXT END ---
"""

# --- 3. THE WEB ROUTE ---

@app.route('/', methods=['GET'])
def home():
    return "Gemini RAG Server is Running!"

@app.route('/chat', methods=['POST'])
def chat():
    # SAFETY CHECK: Stop here if client didn't load
    if client is None:
        return jsonify({
            "error": "Backend Configuration Error: Google API Key is missing or invalid on the server."
        }), 500

    try:
        user_data = request.json
        user_query = user_data.get('message')

        if not user_query:
            return jsonify({"error": "No message provided"}), 400

        # Generate content
        response = client.models.generate_content(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2
            ),
            contents=[user_query]
        )

        return jsonify({"response": response.text})

    except Exception as e:
        print(f"Error during chat generation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Local testing
    app.run(debug=True, port=5000)