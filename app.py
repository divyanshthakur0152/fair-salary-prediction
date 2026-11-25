import glob
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Expected CTC Dashboard", page_icon="💰", layout="wide")

NUM_COLS = [
    "Total_Experience", "Total_Experience_in_field_applied", "Current_CTC",
    "No_Of_Companies_worked", "Number_of_Publications",
    "Certifications", "International_degree_any",
]
CAT_COLS = ["Education", "Inhand_Offer", "Last_Appraisal_Rating",
            "Department", "Role", "Industry"]
TARGET = "Expected_CTC"


@st.cache_data
def load_data():
    files = glob.glob("expected_ctc*.csv")      # finds "expected_ctc (1).csv" too
    return pd.read_csv(files[0])


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


def inr(x):
    return f"₹{x:,.0f}"


df = load_data()
model = load_model()

st.title("💰 Expected CTC Prediction Dashboard")
st.caption("Small Random Forest model (model.pkl under 100 KB) trained on 25,000 applicants")

# ---------------- Sidebar: applicant form ----------------
st.sidebar.header("Applicant details")
exp = st.sidebar.slider("Total experience (years)", 0, 25, 10)
exp_field = st.sidebar.slider("Experience in applied field (years)", 0, 25, min(6, exp))
current = st.sidebar.number_input("Current CTC (₹)", 0, 5_000_000, 1_800_000, step=50_000)
companies = st.sidebar.slider("Companies worked", 0, 6, 3)
pubs = st.sidebar.slider("Publications", 0, 8, 2)
certs = st.sidebar.slider("Certifications", 0, 5, 1)
intl = st.sidebar.selectbox("International degree", [0, 1],
                            format_func=lambda v: "Yes" if v else "No")
edu = st.sidebar.selectbox("Education", ["Under Grad", "Grad", "PG", "Doctorate"], index=2)
offer = st.sidebar.selectbox("Inhand offer", ["N", "Y"])
rating = st.sidebar.selectbox("Last appraisal rating",
                              ["A", "B", "C", "D", "Key_Performer"], index=1)
dept = st.sidebar.selectbox("Department", sorted(df["Department"].dropna().unique()))
role = st.sidebar.selectbox("Role", sorted(df["Role"].dropna().unique()))
industry = st.sidebar.selectbox("Industry", sorted(df["Industry"].dropna().unique()))

applicant = pd.DataFrame([{
    "Total_Experience": exp, "Total_Experience_in_field_applied": exp_field,
    "Current_CTC": current, "No_Of_Companies_worked": companies,
    "Number_of_Publications": pubs, "Certifications": certs,
    "International_degree_any": intl, "Education": edu, "Inhand_Offer": offer,
    "Last_Appraisal_Rating": rating, "Department": dept, "Role": role,
    "Industry": industry,
}])
pred = model.predict(applicant)[0]

tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Data insights", "🧪 Model performance"])

# ---------------- Tab 1: prediction ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Expected CTC", inr(pred))
    c2.metric("Current CTC", inr(current))
    if current > 0:
        c3.metric("Expected hike", f"{(pred - current) / current * 100:.1f}%",
                  inr(pred - current))
    else:
        c3.metric("Expected hike", "N/A (fresher)")

    st.subheader("How you compare with the dataset")
    similar = df[df["Total_Experience"].between(max(exp - 2, 0), exp + 2)]
    a, b = st.columns(2)
    a.metric("Average expected CTC (similar experience)", inr(similar[TARGET].mean()))
    b.metric("Applicants with similar experience", f"{len(similar):,}")
    st.info("Change any value in the sidebar and the prediction updates instantly.")

# ---------------- Tab 2: data insights ----------------
with tab2:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Applicants", f"{len(df):,}")
    k2.metric("Avg current CTC", inr(df["Current_CTC"].mean()))
    k3.metric("Avg expected CTC", inr(df[TARGET].mean()))
    k4.metric("Avg experience", f"{df['Total_Experience'].mean():.1f} yrs")

    l, r = st.columns(2)
    with l:
        st.subheader("Avg expected CTC by education")
        st.bar_chart(df.groupby("Education")[TARGET].mean())
    with r:
        st.subheader("Avg expected CTC by experience")
        st.line_chart(df.groupby("Total_Experience")[TARGET].mean())

    l, r = st.columns(2)
    with l:
        st.subheader("Avg expected CTC by department")
        st.bar_chart(df.groupby("Department")[TARGET].mean().sort_values())
    with r:
        st.subheader("Current vs expected CTC (sample of 800)")
        st.scatter_chart(df.sample(800, random_state=1),
                         x="Current_CTC", y=TARGET)

    with st.expander("View raw data"):
        st.dataframe(df.head(200))

# ---------------- Tab 3: model performance ----------------
with tab3:
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    test_pred = model.predict(X_test)

    m1, m2, m3 = st.columns(3)
    m1.metric("R² score", f"{r2_score(y_test, test_pred):.4f}")
    m2.metric("Mean absolute error", inr(mean_absolute_error(y_test, test_pred)))
    m3.metric("Test applicants", f"{len(y_test):,}")

    l, r = st.columns(2)
    with l:
        st.subheader("Actual vs predicted (sample of 500)")
        cmp_df = pd.DataFrame({"Actual": y_test.values, "Predicted": test_pred})
        st.scatter_chart(cmp_df.sample(500, random_state=1), x="Actual", y="Predicted")
    with r:
        st.subheader("Feature importance")
        names = model.named_steps["pre"].get_feature_names_out()
        imp = pd.Series(model.named_steps["rf"].feature_importances_, index=names)
        grouped = {}
        for n, v in imp.items():
            col = n.split("__", 1)[1]
            for c in CAT_COLS:
                if col.startswith(c + "_"):
                    col = c
                    break
            grouped[col] = grouped.get(col, 0) + v
        st.bar_chart(pd.Series(grouped).sort_values(ascending=False).head(10))
        