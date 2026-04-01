# 🚢 Titanic Survival Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://titanic-survival-eda-fogo26ksmkvn73sxghhjdq.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-83%25-green)

> End-to-end ML project — from raw data to a live deployed web app.
> Predicts whether a Titanic passenger would have survived based on
> demographic and ticket features.

---

## 🎯 Results At A Glance

| Metric | Score |
|--------|-------|
| Accuracy | 83% |
| ROC-AUC | 0.897 |
| CV Folds | 5-fold |
| Best Model | Random Forest |

---

## 🔧 Feature Engineering

- **Title** extracted from name (Mr / Mrs / Miss / Master)
- **FamilySize** = SibSp + Parch + 1
- **IsAlone** — flag for solo travellers
- **AgeGroup** — Child / Teen / Young Adult / Adult / Senior
- **FareBin** — fare quartile grouping

---

## 📊 Key Insights From EDA

- Women had **74% survival rate** vs 19% for men
- 1st class passengers: **63% survival** vs 24% in 3rd class
- Children under 16 had **58% survival rate**
- Small families (2–4 members) survived better than solo travellers

---

## 📁 Project Structure
```
├── titanic_eda.ipynb     # Full EDA + model training notebook
├── app.py                # Streamlit web application
├── titanic_model.pkl     # Trained Random Forest model
├── feature_columns.pkl   # Saved feature column order
├── train.csv             # Training dataset
└── requirements.txt      # Dependencies
```

---

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠 Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit

---

*Built by Om Prakash Chhotray — BSc CS | Aspiring Data Scientist*
*[GitHub Profile](https://github.com/omprakash-ds)*
