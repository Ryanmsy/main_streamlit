import streamlit as st

st.set_page_config(page_title="Ryan's Portfolio", layout="wide")

st.title("Ryan's Portfolio")
st.write("Welcome! Use the sidebar to navigate between projects.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Analysis")
    st.write("Compare two models: SVM and Transforme — trained on Amazon product reviews. Input any text and see how each model classifies the sentiment.")

with col2:
    st.subheader("Sous Chef Agent")
    st.write("AI agent: Gemini 2.5 Flash. Tell it what ingredients you have and it will find real recipes using the Spoonacular API.")
