import os
import fitz
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import json
import re
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

app = Flask(__name__)
CORS(app)
# HuggingFace (open source) set up
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Model to use
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)

# Extract information from the PDF that the user inputs
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = []

    for page in doc:
        width = page.rect.width
        height = page.rect.height
        # Read it in left column and then right column because patents have 2 columns
        left_col = fitz.Rect(0, 0, width / 2, height)
        right_col = fitz.Rect(width / 2, 0, width, height)

        left_text = page.get_text("text", clip=left_col)
        right_text = page.get_text("text", clip=right_col)

        combined = left_text.strip() + "\n" + right_text.strip()
        text.append(combined)
    # Return the whole text
    return "\n\n".join(text)

# Read it in chunks so it doesn't eat up all of the tokens
def chunk_text(text, max_tokens=2000):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        start = end
    return chunks

# Created a nested output so the taxonomy looks better
def nest_taxonomy(flat_list):
    def recursive_dict():
        return defaultdict(recursive_dict)

    hierarchy = recursive_dict()
    # Organize in levels
    for item in flat_list:
        l1 = item["Level 1"] or "null"
        l2 = item["Level 2"] or "null"
        l3 = item["Level 3"] or "null"
        l4 = item["Level 4"] or "null"
        l5 = item["Level 5"] or "null"

        hierarchy[l1][l2][l3][l4][l5].setdefault("items", []).append({
            "Comment": item["Comment"]
        })

    def to_dict(d):
        if isinstance(d, defaultdict):
            return {k: to_dict(v) for k, v in d.items()}
        return d

    return to_dict(hierarchy)

# Query to ask hugging face what it should do
def query_huggingface(prompt_chunk):
    system_message = """
You are a patent analysis assistant. Your task is to extract a structured taxonomy from patent text and return it in valid JSON format. There should be approximately 100 steps in the taxonomy.

There should be at least three entries at each level, including Level 1.

Each item in the JSON array must contain the following keys in this exact order:
- "Level 1"
- "Level 2"
- "Level 3"
- "Level 4"
- "Level 5"
- "Level 6"
- "Level 7"
- "Comment"

### Instructions:

1. The taxonomy should be hierarchical:
   - Level 1: Broad domain (e.g., Materials, Processes, Applications)
   - Level 2: High-level category (e.g., 1. Composite Substrates)
   - Level 3: Subcategory or classification (e.g., 1.1. By Function)
   - Level 4: Specific technical feature (e.g., 1.1.1. Thermal Regulation)
   - Level 5: Further refinement (e.g., 1.1.1.1. Phase Change Materials)
   - Level 6: Implementation detail (e.g., 1.1.1.1.1. Encapsulation Techniques)
   - Level 7: Variant or embodiment (e.g., 1.1.1.1.1.1. Microencapsulation using Urea-Formaldehyde)

2. All levels must be present in each JSON item.
   - If a level does not apply, use `null`.

3. The "Comment" field should include:
   - A technical explanation from the patent
   - Interested parties (e.g., inventors, assignees, competitors)
   - Any cited prior art (e.g., other patents, academic papers)

Format the comment like this:
"Comment: [Technical summary]. Related Art: [References]. Interested Parties: [Stakeholders]."

4. Output must be a valid JSON array only — no markdown, no extra commentary.

5. Extract ALL information from the patent. The larger and deeper the taxonomy is, the better it is.
"""

    full_prompt = f"{system_message}\n\nAnalyze this patent text chunk and extract structured taxonomy in JSON:\n{prompt_chunk}"
    result = text_generator(full_prompt, do_sample=False)[0]["generated_text"]
    # Return in JSON
    return result

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    # Make sure it is a PDF
    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400

    try:
        full_text = extract_text_from_pdf(file)
        chunks = chunk_text(full_text, max_tokens=2000)

        combined_results = []
        for chunk in chunks:
            reply = query_huggingface(chunk)
            # Chunk response for tests
            print("\n--- Chunk Response ---\n")
            print(reply)
            print("\n----------------------\n")

            clean_reply = re.sub(r'^```(?:json)?|```$', '', reply.strip(), flags=re.MULTILINE).strip()

            try:
                chunk_data = json.loads(clean_reply)
                if isinstance(chunk_data, list):
                    combined_results.extend(chunk_data)
            except json.JSONDecodeError as e:
                return jsonify({
                    "error": "Failed to parse JSON from Hugging Face response",
                    "details": str(e),
                    "raw_response": reply
                }), 500
        nested = nest_taxonomy(combined_results)
        # Final response for data
        print("\n--- Final Response ---\n")
        print(nested)
        print("\n----------------------\n")
        # return in JSON
        return jsonify(nested)

    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
# Main
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
