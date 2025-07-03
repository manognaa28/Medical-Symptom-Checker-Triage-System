import pandas as pd
import joblib

# Load your trained model
clf = joblib.load("models/decision_tree.pkl")

# Load the dataset just to get column info
df = pd.read_csv("data/synthetic_medical_symptoms.csv")
X_columns = pd.get_dummies(df.drop(["ID", "Risk_Level"], axis=1), drop_first=True).columns

def triage_system(input_symptoms):
    """
    input_symptoms: dict of patient symptoms
    """
    input_df = pd.DataFrame([input_symptoms])
    input_df = pd.get_dummies(input_df, drop_first=True).reindex(columns=X_columns, fill_value=0)
    risk = clf.predict(input_df)[0]
    if risk == "High":
        recommendation = "Seek immediate medical attention or call emergency services."
    elif risk == "Moderate":
        recommendation = "Consult your doctor within the next 24-48 hours."
    else:
        recommendation = "Manage symptoms at home and monitor for any worsening."
    return f"Risk Level: {risk}\nRecommendation: {recommendation}"

def get_binary_input(symptom_name):
    while True:
        val = input(f"Do you have {symptom_name}? (0 = No, 1 = Yes): ")
        if val in ["0", "1"]:
            return int(val)
        print("Invalid input, please enter 0 or 1.")

def get_int_input(name, min_val, max_val):
    while True:
        val = input(f"Enter {name} ({min_val}-{max_val}): ")
        if val.isdigit() and min_val <= int(val) <= max_val:
            return int(val)
        print(f"Invalid input, please enter a number between {min_val} and {max_val}.")

def get_yesno_input(name):
    while True:
        val = input(f"Do you have {name}? (Yes/No): ").strip().lower()
        if val in ["yes", "no"]:
            return val.capitalize()
        print("Invalid input, please enter Yes or No.")

# Interactive CLI for live user input
print("🩺 Welcome to the AI Medical Symptom Checker!\nPlease answer the following questions:")

example_patient = {
    "Fever": get_binary_input("Fever"),
    "Cough": get_binary_input("Cough"),
    "Chest_Pain": get_binary_input("Chest Pain"),
    "Headache": get_binary_input("Headache"),
    "SOB": get_binary_input("Shortness of Breath"),
    "Nausea": get_binary_input("Nausea"),
    "Vomiting": get_binary_input("Vomiting"),
    "Diarrhea": get_binary_input("Diarrhea"),
    "Rash": get_binary_input("Rash"),
    "Severity": get_int_input("Symptom severity (1 mild - 10 severe)", 1, 10),
    "Duration": get_int_input("Duration of symptoms in days", 1, 30),
    "Risk_Factor": get_yesno_input("any chronic risk factors (e.g., diabetes, heart disease)"),
}

print(" Patient Triage Result:")
print(triage_system(example_patient))
