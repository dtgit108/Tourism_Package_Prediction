import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(
    repo_id="Dtapkir/TourismPackagePrediction",
    filename="best_tourism_package_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the
**Wellness Tourism Package** based on their profile and interaction details.
Please enter the customer information below to get a prediction.
""")

# User input fields
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business", "Other"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Designation = st.text_input("Designation", "Executive")
ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe"]
)

Age = st.number_input("Age", min_value=18, max_value=100, value=35)
CityTier = st.selectbox("City Tier", [1, 2, 3])
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
Passport = st.selectbox("Passport Available", [0, 1])
OwnCar = st.selectbox("Owns a Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=50000)
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=300, value=30)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'ProductPitched': ProductPitched
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Likely to Purchase" if prediction == 1 else "Not Likely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
