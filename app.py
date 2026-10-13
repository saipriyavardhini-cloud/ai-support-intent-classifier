import streamlit as st
from transformers import pipeline

# Page Config
st.set_page_config(page_title="AI Support Intent Classifier", page_icon="🤖", layout="centered")

st.title("🤖 AI Customer Support Intent Classifier")
st.write("Classify customer support tickets or messages instantly using Hugging Face transformers.")

@st.cache_resource
def load_model():
    # Load a lightweight zero-shot classification model
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

with st.spinner("Loading AI Model... Please wait."):
    classifier = load_model()

# User input text box
user_input = st.text_area("Enter customer message or ticket:", "Hey, my payment went through twice for my last order, please refund my money!")

# Define potential categories
candidate_labels = ["Billing & Refunds", "Technical Issue", "Account Access", "General Inquiry"]

if st.button("Classify Intent", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Analyzing text..."):
            result = classifier(user_input, candidate_labels)
            
            st.success("Analysis Complete!")
            
            # Display top prediction
            top_label = result['labels'][0]
            top_score = result['scores'][0] * 100
            
            st.metric(label="Predicted Intent", value=top_label, delta=f"{top_score:.1f}% Confidence")
            
            # Show breakdown chart/scores
            st.subheader("Detailed Breakdown")
            for label, score in zip(result['labels'], result['scores']):
                st.progress(score, text=f"{label}: {score * 100:.1f}%")