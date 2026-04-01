import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival chance")

# Load model
model = pickle.load(open('titanic_model.pkl', 'rb'))

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Siblings / Spouse aboard", 0, 8, 0)
parch = st.number_input("Parents / Children aboard", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

if st.button("Predict Survival"):
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    # Feature Engineering
    input_data['FamilySize'] = input_data['SibSp'] + input_data['Parch'] + 1
    input_data['IsAlone'] = (input_data['FamilySize'] == 1).astype(int)
    input_data['Title'] = input_data['Sex'].map({'male': 'Mr', 'female': 'Mrs'})

    input_data = pd.get_dummies(input_data, drop_first=True)

    # Align columns with model training
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ **Would Survive** (Probability: {probability:.1%})")
    else:
        st.error(f"❌ **Would Not Survive** (Probability: {1-probability:.1%})")
