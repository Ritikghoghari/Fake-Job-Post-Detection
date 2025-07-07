import streamlit as st
import pickle

# Load model and vectorizer
with open('fake_job_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# App title
st.title("🕵️‍♂️ Fake Job Post Detector")
st.write("Paste a job description below to check if it's real or fake.")

# Input box
job_description = st.text_area("Enter job description here...", height=300)

if st.button("Predict"):
    if job_description.strip() == "":
        st.warning("Please enter a job description.")
    else:
        # Preprocess and predict
        text_vec = vectorizer.transform([job_description])
        prediction = model.predict(text_vec)[0]
        prob = model.predict_proba(text_vec)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 Warning: This looks like a **FAKE** job post. (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ This job post looks **REAL**. (Confidence: {prob:.2f})")
