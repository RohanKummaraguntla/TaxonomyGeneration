#Patent Taxonomy Generation (taxonomy.py)

A Flask-based API that uses Hugging Face language models to extract structured taxonomies from patent PDF files. 

---

## Features

- Parses PDF patent documents with two-column layout support
- Outputs 7-level nested JSON taxonomies with rich comments
- JSON structure includes technical summaries, prior art, and stakeholder mentions

---

## Setup Instructions

### 1. Clone the repository and navigate into it

bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

### 2. Install requirements

pip install flask flask-cors transformers torch python-dotenv PyMuPDF
