import streamlit as st
import numpy as np
#from transformers import BertTokenizer, BertForSequenceClassification
import torch

#@st.cache(allow_output_mutation=True)
#def get_model():
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertForSequenceClassification.from_pretrained("pnichite/YTFineTuneBert")
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("agi-css/distilbert-base-uncased-finetuned-toxicity")
model = AutoModelForSequenceClassification.from_pretrained("agi-css/distilbert-base-uncased-finetuned-toxicity")
#    return tokenizer,model
st.write('model loaded')

#tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Predict")

d = {
    
  1:'Toxic',
  0:'Non Toxic'
}
st.write('here')
if user_input and button :
    st.write(user_input)
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    st.write('2')
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])
    st.stop()
