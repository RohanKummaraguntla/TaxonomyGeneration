# Patent Taxonomy Generation (taxonomy.py)

## What is this?

This repository contains the code, data processing pipeline, and LaTeX manuscript for a paper titled:

**“Enhancing Patent Readability: Leveraging Large Language Model-Generated Taxonomies for Structured Outputs”**

A Flask-based API that uses large language models to extract structured taxonomies from patent PDF files. 

---

## Features

- Parses PDF patent documents with two-column layout support
- Outputs 7-level nested JSON taxonomies with rich comments
- JSON structure includes technical summaries, prior art, and stakeholder mentions

---

## Setup instructions

### 1. Clone the repository and navigate into it

bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

### 2. Prerequisites

pip install flask flask-cors transformers torch python-dotenv PyMuPDF

### 3. Running the program

Requires Python 3.9+

PyPDF2, pandas, tqdm (install via pip) Run: pip install openai PyPDF2 pandas tqdm
Running the Program

Setup:
1. Activate a virtual environment to run the python program
2. Input a Patent PDF into the same folder
3. Run the program and the output will be printed and sent to an external file
4. Save the external file to read the outputed taxonomy


## Paper

Compile the paper and run it
