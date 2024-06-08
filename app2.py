import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch



user_input = st.text_area('Enter Text to Analyze')
button = st.button("Predict")


st.write('here')
if user_input and button :
    st.write(user_input)
