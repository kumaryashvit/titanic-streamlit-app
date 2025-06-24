import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.joblib")

# App title
st.title("Titanic Survival Prediction")

# Input form
st.header("Enter Passenger Details")
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0.42, 80.0, 30.0)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, step=1)
Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, step=1)
Fare = st.number_input("Fare", min_value=0.0, max_value=600.0, step=1.0)
Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Predict button
if st.button("Predict"):
    # Create dataframe
    input_df = pd.DataFrame([{
        "Pclass": Pclass,
        "Sex": 0 if Sex == "male" else 1,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": {"S": 0, "C": 1, "Q": 2}[Embarked]
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Display result
    st.success(f"Survived: {'Yes' if prediction else 'No'}")