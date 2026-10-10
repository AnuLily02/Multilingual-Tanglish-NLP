import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(page_title="Tanglish Sentiment Analyzer", page_icon="🎭")
st.title("🎭 Tanglish Sentiment Analyzer")
st.write("Analyze sentiment in code-mixed Tanglish (Tamil + English) text!")


# Load fine-tuned local model
@st.cache_resource
def load_model():
    model_path = "./fine_tuned_tanglish"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


try:
    tokenizer, model = load_model()
    # Exact mapping used in train.py: 0 -> Positive, 1 -> Negative
    LABEL_MAPPING = {0: "Positive", 1: "Negative"}
except Exception as e:
    st.error(
        f"Error loading model from './fine_tuned_tanglish'. Make sure train.py has completed! Details: {e}"
    )

user_input = st.text_area(
    "Enter Tanglish text:", "Intha movie semma mass ah irundhuchu!"
)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Tokenize user input
        inputs = tokenizer(
            user_input, return_tensors="pt", truncation=True, max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(
                outputs.logits, dim=-1
            )
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        sentiment = LABEL_MAPPING.get(predicted_class, "Unknown")

        if sentiment == "Positive":
            st.success(
                f"**Sentiment:** {sentiment} (Confidence: {confidence:.2%})"
            )
        else:
            st.error(
                f"**Sentiment:** {sentiment} (Confidence: {confidence:.2%})"
            )
    else:
        st.warning("Please enter some text to analyze.")