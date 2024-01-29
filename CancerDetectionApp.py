import streamlit as st
import pickle
import pandas as pd

# import xgboost as xgb

st.set_page_config(page_icon="♋")

model_cancer = pickle.load(open("cancer.pkl", "rb"))

# List of features used during training
trained_features_cancer = [
    "Age",
    "AirPollution",
    "Alcoholuse",
    "DustAllergy",
    "GeneticRisk",
    "chronicLungDisease",
    "Obesity",
    "Smoking",
]


def get_risk_color(prediction):
    if prediction == 0:
        return "green"  # Low
    elif prediction == 1:
        return "yellow"  # Medium
    else:
        return "red"  # High


def main():
    st.markdown(
        "<h1 style='text-align: center;'>Cancer Detection App</h1>",
        unsafe_allow_html=True,
    )

    age = st.slider("Enter Age:", min_value=1, max_value=100, value=50)
    air_pollution = st.slider(
        "Enter Air Pollution level:", min_value=0, max_value=10, value=5
    )
    alcohol_use = st.slider(
        "Enter Alcohol Use level:", min_value=0, max_value=10, value=5
    )
    dust_allergy = st.slider(
        "Enter Dust Allergy level:", min_value=0, max_value=10, value=5
    )
    genetic_risk = st.slider(
        "Enter Genetic Risk level:", min_value=0, max_value=10, value=5
    )
    chronic_lung_disease = st.slider(
        "Enter Chronic Lung Disease level:", min_value=0, max_value=10, value=5
    )
    obesity = st.slider("Enter Obesity level:", min_value=0, max_value=10, value=5)
    smoking = st.slider("Enter Smoking level:", min_value=0, max_value=10, value=5)

    user_data_cancer = pd.DataFrame(
        {
            "Age": [age],
            "AirPollution": [air_pollution],
            "Alcoholuse": [alcohol_use],
            "DustAllergy": [dust_allergy],
            "GeneticRisk": [genetic_risk],
            "chronicLungDisease": [chronic_lung_disease],
            "Obesity": [obesity],
            "Smoking": [smoking],
        }
    )

    user_data_cancer = user_data_cancer[trained_features_cancer]

    prediction_cancer = model_cancer.predict(user_data_cancer)


    risk_color = get_risk_color(prediction_cancer[0])
    prediction_text = f"<p style='color:{risk_color}; text-align: center; font-weight: bold; width: 50%; margin: 0 auto; padding: 10px; border: 2px solid {risk_color}; border-radius: 5px;'>Prediction: {'Low' if prediction_cancer[0] == 0 else 'Medium' if prediction_cancer[0] == 1 else 'High'} Cancer Risk</p>"
    st.markdown(prediction_text, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
