import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("model.joblib")
ve = joblib.load("vectorizer.joblib")

# Page configuration
st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered"
)

# Title
st.title("📧 Spam Email Detection")
st.write("Enter an email message below to check whether it is Spam or Ham.")

# Input box
email = st.text_area("Enter Email Message")

# Prediction button
if st.button("Predict"):

    if email.strip() == "":
        st.warning("Please enter an email message.")
    else:
        email_vector = ve.transform([email])
        prediction = model.predict(email_vector)

        # Adjust this condition if your labels are different
        if prediction[0] == "spam" or prediction[0] == 1:
            st.error("🚫 Spam Email")
        else:
            st.success("✅ Not Spam (Ham)")
            