import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(page_title="Tanglish Sentiment Analyzer", page_icon="🎭")
st.title("🎭 Tanglish Sentiment Analyzer")
st.write("Analyze sentiment in code-mixed Tanglish (Tamil + English) text!")

@st.cache_resource
def load_model():
    # Path to your fine-tuned model or xlm-roberta-base
    model_path = "./fine_tuned_tanglish"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()
label_map = {0: "Positive", 1: "Negative"}

user_input = st.text_area("Enter Tanglish text:", "Intha movie semma mass ah irundhuchu!")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()

        sentiment = label_map[pred]
        if sentiment == "Positive":
            st.success(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2%})")
        else:
            st.error(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2%})")
    else:
        st.warning("Please enter some text.")