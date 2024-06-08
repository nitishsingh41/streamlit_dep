import streamlit as st
import numpy as np

import torch


from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("AdamCodd/tinybert-sentiment-amazon")
model = AutoModelForSequenceClassification.from_pretrained("AdamCodd/tinybert-sentiment-amazon")
st.write('model loaded')

#tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Predict")



st.write('here')
if user_input and button :

    inputs = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    st.write(model.config.id2label[predicted_class_id])
