import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",          # wide layout = instantly looks more professional
    initial_sidebar_state="expanded"
)

# ── Custom CSS — makes it look NOTHING like a default Streamlit app ───────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%); }
    .metric-card {
        background: linear-gradient(135deg, #1e2433, #252d3d);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .survive-box {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .die-box {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    h1 { color: #e2e8f0 !important; }
    .stSelectbox label, .stSlider label, .stNumberInput label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model & feature columns ─────────────────────────────────────────────
@st.cache_resource   # cache so model loads only ONCE — shows you know optimization
def load_model():
    model = pickle.load(open('titanic_model.pkl', 'rb'))
    feature_cols = pickle.load(open('feature_columns.pkl', 'rb'))
    return model, feature_cols

model, feature_cols = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🚢 Titanic Survival Predictor")
st.markdown("##### *ML-powered prediction using Random Forest · 83% Accuracy · 0.897 ROC-AUC*")
st.markdown("---")

# ── Layout: two columns ───────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 🧍 Passenger Details")

    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1st Class (Upper)", 2: "2nd Class (Middle)", 3: "3rd Class (Lower)"}[x]
    )

    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)

    age = st.slider("Age", min_value=1, max_value=80, value=30,
                    help="Passenger age in years")

    col_a, col_b = st.columns(2)
    with col_a:
        sibsp = st.number_input("Siblings / Spouse", min_value=0, max_value=8, value=0)
    with col_b:
        parch = st.number_input("Parents / Children", min_value=0, max_value=6, value=0)

    fare = st.slider("Fare Paid (£)", min_value=0.0, max_value=512.0, value=32.0, step=0.5)

    embarked = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        format_func=lambda x: {"S": "Southampton 🇬🇧", "C": "Cherbourg 🇫🇷", "Q": "Queenstown 🇮🇪"}[x]
    )

    predict_btn = st.button("🔮 Predict My Survival", use_container_width=True, type="primary")

# ── Feature Engineering (EXACTLY matching training pipeline) ──────────────────
def engineer_features(pclass, sex, age, sibsp, parch, fare, embarked):
    """
    Mirrors the exact feature engineering done during model training.
    This consistency is what separates good DS from great DS.
    """
    sex_lower = sex.lower()

    # Title extraction — the RIGHT way (covers all title groups)
    if sex_lower == 'male':
        title = 'Master' if age < 16 else 'Mr'
    else:
        title = 'Miss' if age < 18 else 'Mrs'

    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    # Age grouping — adds non-linear age signal
    if age <= 12:
        age_group = 'Child'
    elif age <= 18:
        age_group = 'Teen'
    elif age <= 35:
        age_group = 'Young Adult'
    elif age <= 60:
        age_group = 'Adult'
    else:
        age_group = 'Senior'

    # Fare binning — high fare = higher class signal
    fare_bin = pd.cut([fare], bins=[0, 7.9, 14.4, 31.0, 512], labels=[0, 1, 2, 3])[0]

    df = pd.DataFrame({
        'Pclass':      [pclass],
        'Sex':         [sex_lower],
        'Age':         [age],
        'SibSp':       [sibsp],
        'Parch':       [parch],
        'Fare':        [fare],
        'Embarked':    [embarked],
        'FamilySize':  [family_size],
        'IsAlone':     [is_alone],
        'Title':       [title],
        'AgeGroup':    [age_group],
        'FareBin':     [int(fare_bin)]
    })

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'AgeGroup'], drop_first=False)

    # Align to training feature columns — handles any missing dummies
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df

# ── Prediction + Results ──────────────────────────────────────────────────────
with col_result:
    st.markdown("### 📊 Prediction Result")

    if predict_btn:
        input_df = engineer_features(pclass, sex, age, sibsp, parch, fare, embarked)

        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        survive_prob = probability[1]
        die_prob     = probability[0]

        # ── Big result card ───────────────────────────────────────────────────
        if prediction == 1:
            st.markdown(f"""
            <div class="survive-box">
                <h1 style="color:#10b981; margin:0">✅ SURVIVED</h1>
                <h2 style="color:#d1fae5; margin:8px 0">{survive_prob:.1%} survival chance</h2>
                <p style="color:#6ee7b7">This passenger likely made it off the Titanic.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="die-box">
                <h1 style="color:#ef4444; margin:0">❌ DID NOT SURVIVE</h1>
                <h2 style="color:#fee2e2; margin:8px 0">{die_prob:.1%} chance of not surviving</h2>
                <p style="color:#fca5a5">Historical data suggests this passenger did not survive.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability gauge bar ─────────────────────────────────────────────
        st.markdown("**Survival Probability Breakdown**")
        prob_df = pd.DataFrame({
            'Outcome':     ['Survived ✅', 'Did Not Survive ❌'],
            'Probability': [survive_prob, die_prob]
        })
        st.bar_chart(prob_df.set_index('Outcome'))

        # ── Key factors that influenced this prediction ────────────────────────
        st.markdown("**🔍 Key Factors in This Prediction**")

        factors = []
        if sex.lower() == 'female':
            factors.append("✅ Female — Women had 74% survival rate vs 19% for men")
        else:
            factors.append("❌ Male — Men had only 19% survival rate")

        if pclass == 1:
            factors.append("✅ 1st Class — 63% survival rate")
        elif pclass == 2:
            factors.append("🟡 2nd Class — 47% survival rate")
        else:
            factors.append("❌ 3rd Class — Only 24% survival rate")

        family_size = sibsp + parch + 1
        if 2 <= family_size <= 4:
            factors.append(f"✅ Family size {family_size} — Small families survived better")
        elif family_size == 1:
            factors.append("🟡 Travelling alone — Slightly lower survival odds")
        else:
            factors.append(f"❌ Large family ({family_size}) — Lower survival rate")

        if age < 16:
            factors.append("✅ Child — Children were prioritised in lifeboats")
        elif age > 60:
            factors.append("❌ Senior — Older passengers had lower survival rates")

        for f in factors:
            st.markdown(f"- {f}")

    else:
        # Placeholder before prediction
        st.info("👈 Fill in passenger details and click **Predict My Survival** to see the result.")

        # ── Show historical survival stats as context ──────────────────────────
        st.markdown("**📈 Historical Titanic Survival Rates**")
        stats = pd.DataFrame({
            'Group':        ['Women', 'Men', '1st Class', '2nd Class', '3rd Class', 'Children'],
            'Survival Rate': [74, 19, 63, 47, 24, 58]
        })
        st.bar_chart(stats.set_index('Group'))

# ── Footer with your brand ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a5568; font-size:13px; padding: 10px'>
    Built by <strong style='color:#667eea'>Om Prakash Chhotray</strong> · 
    <a href='https://github.com/omprakash-ds' style='color:#667eea'>GitHub</a> · 
    Model: Random Forest · Accuracy: 83% · ROC-AUC: 0.897
</div>
""", unsafe_allow_html=True)
