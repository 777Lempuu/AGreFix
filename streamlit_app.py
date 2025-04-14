import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import gdown
import os

# Set up the app
st.set_page_config(page_title="AG News Classifier", layout="wide")
st.title("AG News Text Classification with ReFixMatch")

# Download model weights from Google Drive
@st.cache_resource
def download_model():
    model_url = "https://drive.google.com/uc?id=1GH9voK2QGYeoF7VI5UUaDcZNtXr0ZVHX"
    output_path = "AG_ReFix.pt"
    
    if not os.path.exists(output_path):
        with st.spinner("Downloading model weights (this may take a few minutes)..."):
            try:
                gdown.download(model_url, output_path, quiet=False)
            except Exception as e:
                st.error(f"Failed to download model weights: {str(e)}")
                return False
    return True

# Load model and tokenizer
@st.cache_resource
def load_model():
    if not download_model():
        return None, None
    
    model_name = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        # Load the saved model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
        state_dict = torch.load('AG_ReFix.pt', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model_state_dict'])
        return model, tokenizer
    except Exception as e:
        st.error(f"Could not load model: {str(e)}")
        return None, None

model, tokenizer = load_model()

# Define a simple dataset class for inference
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

# Class labels for AG News
class_labels = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Tech"
}

# App functionality
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header("Classify a Single News Text")
    input_text = st.text_area("Enter news text to classify:", height=150, 
                            placeholder="Paste news article text here...")
    
    if st.button("Classify") and input_text:
        if model is None:
            st.error("Model not loaded. Please check if the model weights downloaded correctly.")
        else:
            with st.spinner("Classifying..."):
                # Create dataset and dataloader
                dataset = InferenceDataset([input_text], tokenizer)
                loader = torch.utils.data.DataLoader(dataset, batch_size=1)
                
                # Get prediction
                model.eval()
                with torch.no_grad():
                    for batch in loader:
                        inputs = {k: v for k, v in batch.items()}
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                        pred_class = torch.argmax(probs).item()
                        confidence = torch.max(probs).item()
                
                # Display results
                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Class", f"{class_labels[pred_class]} (Class {pred_class})")
                    st.metric("Confidence", f"{confidence:.2%}")
                
                with col2:
                    # Show probability distribution
                    prob_data = {class_labels[i]: probs[0][i].item() for i in range(4)}
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=list(prob_data.values()), y=list(prob_data.keys()), palette="Blues_d", ax=ax)
                    ax.set_xlabel("Probability")
                    ax.set_title("Class Probabilities")
                    ax.set_xlim(0, 1)
                    st.pyplot(fig)

with tab2:
    st.header("Classify Multiple News Texts")
    st.info("Upload a CSV file containing news articles in a column named 'text'")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("Error: CSV file must contain a 'text' column")
            else:
                st.success(f"Successfully loaded {len(df)} records")
                st.dataframe(df.head(3))
                
                if st.button("Classify All") and model is not None:
                    with st.spinner(f"Classifying {len(df)} texts..."):
                        # Create dataset and dataloader
                        dataset = InferenceDataset(df['text'].tolist(), tokenizer)
                        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
                        
                        # Get predictions
                        model.eval()
                        predictions = []
                        confidences = []
                        with torch.no_grad():
                            for batch in loader:
                                inputs = {k: v for k, v in batch.items()}
                                outputs = model(**inputs)
                                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                                batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
                                batch_confs = torch.max(probs, dim=1).values.cpu().numpy()
                                predictions.extend(batch_preds)
                                confidences.extend(batch_confs)
                        
                        # Add results to dataframe
                        df['predicted_class'] = predictions
                        df['predicted_label'] = df['predicted_class'].map(class_labels)
                        df['confidence'] = confidences
                        
                        # Show results
                        st.subheader("Classification Results")
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Results as CSV",
                            csv,
                            "classification_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
                        # Show statistics
                        st.subheader("Prediction Statistics")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Class Distribution**")
                            fig1, ax1 = plt.subplots(figsize=(6, 4))
                            sns.countplot(x='predicted_label', data=df, ax=ax1, 
                                        order=list(class_labels.values()))
                            ax1.set_title("Predicted Class Distribution")
                            ax1.set_xlabel("Class")
                            ax1.set_ylabel("Count")
                            plt.xticks(rotation=45)
                            st.pyplot(fig1)
                        
                        with col2:
                            st.write("**Confidence Distribution**")
                            fig2, ax2 = plt.subplots(figsize=(6, 4))
                            sns.histplot(df['confidence'], bins=20, kde=True, ax=ax2)
                            ax2.set_title("Confidence Score Distribution")
                            ax2.set_xlabel("Confidence")
                            ax2.set_ylabel("Count")
                            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Add some info about the model
st.sidebar.header("About the Model")
st.sidebar.write("""
This app uses a BERT-based model fine-tuned on the AG News dataset using ReFixMatch, 
a semi-supervised learning approach that combines labeled and unlabeled data.
""")

st.sidebar.header("Class Labels")
for k, v in class_labels.items():
    st.sidebar.write(f"**{k}**: {v}")

st.sidebar.header("How to Use")
st.sidebar.write("""
1. **Single Prediction**: Paste text and click Classify
2. **Batch Prediction**: Upload CSV with 'text' column
""")

if model is None:
    st.error("Model failed to load. Please check the following:")
    st.write("- Internet connection for downloading model weights")
    st.write("- Google Drive link is accessible")
    st.write("- Sufficient disk space (model is ~60MB)")
else:
    st.sidebar.success("Model loaded successfully!")
