import os
import json
import glob
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai  # <-- Using the stable library

# --- 1. SETUP & CONFIGURATION ---

load_dotenv() 

app = Flask(__name__)
CORS(app) 

# Configure Gemini with the Stable Library
# Ensure GOOGLE_API_KEY is set in Render
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("WARNING: GOOGLE_API_KEY not found!")

genai.configure(api_key=api_key)

# We use the standard 1.5 Flash model which never gives 404s on this library
MODEL_NAME = 'gemini-1.5-flash'

# --- 2. DATA LOADING FUNCTIONS ---

def load_and_parse_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        parts = [f"=== Interview File: {os.path.basename(file_path)} ==="]
        for entry in data:
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
    directory = os.getcwd()
    json_paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    parts = []
    
    print(f"Scanning directory: {directory}")
    
    for path in json_paths:
        if "package" in path or "lock" in path: 
            continue

        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    parts.append(load_and_parse_json(path))
                elif isinstance(data, dict) and ("survey_summary" in data or "free_text_insights" in data):
                    parts.append(load_survey_json(path))
            except:
                continue
                
    print(f"Loaded {len(parts)} JSON context blocks.")
    return "\n\n".join(parts)

FULL_INTERVIEW_CONTEXT = load_context()

SYSTEM_INSTRUCTION = f"""
You are an analytical assistant.
Your goals:
1. Use the provided context as background knowledge.
2. Infer patterns, motives, insights, and deeper meanings.
3. Be concise and analytical.

--- CONTEXT START ---
{FULL_INTERVIEW_CONTEXT}
--- CONTEXT END ---
"""

# --- 3. THE WEB ROUTE ---

@app.route('/', methods=['GET'])
def home():
    return "Gemini RAG Server is Running (Stable Version)!"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_data = request.json
        user_query = user_data.get('message')

        if not user_query:
            return jsonify({"error": "No message provided"}), 400

        # Initialize Model (Old Library Syntax)
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION
        )
        
        # Generate Response
        response = model.generate_content(
            user_query,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2
            )
        )

        return jsonify({"response": response.text})

    except Exception as e:
        print(f"Error during chat generation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)