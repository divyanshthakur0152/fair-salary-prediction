import os
import glob
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

CSV = glob.glob("expected_ctc*.csv")[0]   # finds expected_ctc (1).csv too
TARGET = "Expected_CTC"

NUM_COLS = [
    "Total_Experience", "Total_Experience_in_field_applied", "Current_CTC",
    "No_Of_Companies_worked", "Number_of_Publications",
    "Certifications", "International_degree_any",
]
CAT_COLS = [
    "Education", "Inhand_Offer", "Last_Appraisal_Rating",
    "Department", "Role", "Industry",
]

df = pd.read_csv(CSV)
X = df[NUM_COLS + CAT_COLS]
y = df[TARGET]

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ]), CAT_COLS),
])

model = Pipeline([
    ("pre", pre),
    # small forest so model.pkl stays under 100 KB
    ("rf", RandomForestRegressor(n_estimators=10, max_depth=7,
                                 min_samples_leaf=20, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("R2 score :", round(r2_score(y_test, pred), 4))
print("MAE (Rs) :", round(mean_absolute_error(y_test, pred)))

joblib.dump(model, "model.pkl", compress=3)
print("model.pkl size (KB):", round(os.path.getsize("model.pkl") / 1024, 1))