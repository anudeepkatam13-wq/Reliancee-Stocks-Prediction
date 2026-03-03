import streamlit as st
import pickle

st.title("Reliance Stock Prediction App")

@st.cache_resource
def load_model():
    with open("sarima_mod_dep.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

steps = st.number_input(
    "Enter number of days to predict:",
    min_value=1,
    max_value=365,
    value=5
)

if st.button("Predict"):
    try:
        forecast = model.forecast(steps=steps)
        st.success("Prediction Successful!")
        st.write(forecast)
    except Exception as e:
        st.error(f"Error: {e}")
