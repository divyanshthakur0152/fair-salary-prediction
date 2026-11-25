import joblib
import pandas as pd

model = joblib.load("model.pkl")

# Change these values to test a new applicant
applicant = {
    "Total_Experience": 10,
    "Total_Experience_in_field_applied": 6,
    "Current_CTC": 1800000,
    "No_Of_Companies_worked": 3,
    "Number_of_Publications": 2,
    "Certifications": 1,
    "International_degree_any": 0,
    "Education": "PG",              # Under Grad / Grad / PG / Doctorate
    "Inhand_Offer": "N",            # Y / N
    "Last_Appraisal_Rating": "B",   # A / B / C / D / Key_Performer
    "Department": "Marketing",
    "Role": "Consultant",
    "Industry": "Training",
}

pred = model.predict(pd.DataFrame([applicant]))[0]
print(f"Predicted Expected CTC: Rs {pred:,.0f}")