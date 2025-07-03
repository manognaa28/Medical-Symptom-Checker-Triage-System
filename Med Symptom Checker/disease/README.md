AI Medical Symptom Checker & Triage System
==========================================

This project is a command-line tool that asks the user for symptoms and gives a risk level with care recommendations. It uses a decision tree model trained on a synthetic dataset.

Setup and Usage
---------------

1. Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows

2. Install dependencies:
   pip install -r requirements.txt

3. Generate the synthetic dataset:
   python dataset_generator.py

4. Train the model and run the interactive symptom checker:
   python app.py
