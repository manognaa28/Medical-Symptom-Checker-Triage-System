import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 1000
data = {
    "ID": np.arange(1, n_samples + 1),
    "Fever": np.random.choice([0, 1], size=n_samples),
    "Cough": np.random.choice([0, 1], size=n_samples),
    "Chest_Pain": np.random.choice([0, 1], size=n_samples),
    "Headache": np.random.choice([0, 1], size=n_samples),
    "SOB": np.random.choice([0, 1], size=n_samples),
    "Nausea": np.random.choice([0, 1], size=n_samples),
    "Vomiting": np.random.choice([0, 1], size=n_samples),
    "Diarrhea": np.random.choice([0, 1], size=n_samples),
    "Rash": np.random.choice([0, 1], size=n_samples),
    "Severity": np.random.randint(1, 11, size=n_samples),
    "Duration": np.random.randint(1, 15, size=n_samples),
    "Risk_Factor": np.random.choice(["Yes", "No"], size=n_samples),
}

df = pd.DataFrame(data)

def assign_risk(row):
    symptom_sum = row[["Fever","Cough","Chest_Pain","SOB"]].sum()
    if row["Severity"] > 7 or row["Duration"] > 10 or (row["Risk_Factor"] == "Yes" and symptom_sum >= 2):
        return "High"
    elif row["Severity"] > 4 or symptom_sum >= 2:
        return "Moderate"
    else:
        return "Low"

df["Risk_Level"] = df.apply(assign_risk, axis=1)
df.to_csv("data/synthetic_medical_symptoms.csv", index=False)
print(" Dataset saved at data/synthetic_medical_symptoms.csv")
