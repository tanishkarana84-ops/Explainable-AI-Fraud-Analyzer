import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Credit Card Fraud Detector")
st.markdown("*Built with Random Forest + SHAP Explanations*")
st.markdown("---")

# ── LOAD DATA ──
@st.cache_data
def load_data():
    file_id = "1mKfUcSkrubVJ_wR0O-XpyRURe-yzloAE"
    url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

data = load_data()
st.write("Columns:", data.columns.tolist())
st.stop()


# ── BASIC STATS ──
total  = len(data)
fraud  = data["Class"].sum()
legit  = total - fraud

col1, col2, col3 = st.columns(3)
col1.metric("💳 Total Transactions", f"{total:,}")
col2.metric("🚨 Fraud Cases",        f"{fraud:,}")
col3.metric("✅ Legitimate Cases",    f"{legit:,}")

st.markdown("---")

# ── PIE CHART ──
st.subheader("📊 Fraud vs Legitimate")

pie_data = pd.DataFrame({
    "Type":  ["Legitimate", "Fraud"],
    "Count": [legit, fraud]
})

fig_pie = px.pie(
    pie_data,
    values="Count",
    names="Type",
    hole=0.4,
    color="Type",
    color_discrete_map={
        "Legitimate": "#2ecc71",
        "Fraud":      "#e74c3c"
    }
)
st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ── PREPARE DATA ──
X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ── TRAIN MODEL ──
@st.cache_resource
def train_model(_X_train, _y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(_X_train, _y_train)
    return model

with st.spinner("🤖 Training model..."):
    model = train_model(X_train, y_train)

st.success("✅ Model Ready!")

# ── MODEL ACCURACY ──
st.subheader("📈 Model Performance")

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

m1, m2, m3 = st.columns(3)
m1.metric("✅ Accuracy",         f"{report['accuracy']*100:.2f}%")
m2.metric("🎯 Precision",        f"{report['1']['precision']*100:.2f}%")
m3.metric("📊 Recall",           f"{report['1']['recall']*100:.2f}%")

st.markdown("---")

# ── TRANSACTION CHECKER ──
st.subheader("🔍 Check a Transaction")

transaction_index = st.slider(
    "Select Transaction Number",
    0,
    len(X_test) - 1,
    0
)

selected = X_test.iloc[[transaction_index]]

prediction  = model.predict(selected)[0]
probability = model.predict_proba(selected)[0][1]

if prediction == 1:
    st.error(f"🚨 FRAUD DETECTED — {probability:.2%} confidence")
else:
    st.success(f"✅ Legitimate Transaction — {1-probability:.2%} confidence")

st.markdown("---")

# ── SHAP EXPLANATION ──
st.subheader("🧠 Why did the model decide this?")
st.caption("Red = pushed toward FRAUD | Blue = pushed toward LEGITIMATE")

@st.cache_data
def get_shap(_model, _X_test):
    explainer   = shap.TreeExplainer(_model)
    shap_values = explainer(_X_test)
    return shap_values

with st.spinner("Calculating explanation..."):
    shap_values = get_shap(model, X_test)

fig, ax = plt.subplots()
shap.plots.waterfall(
    shap_values[transaction_index, :, 1],
    show=False
)
st.pyplot(fig)
plt.close()