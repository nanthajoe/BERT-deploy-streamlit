
import streamlit as st
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

PRE_TRAINED_MODEL = 'hasil-fine-tuning'

# @st.cache(allow_output_mutation=True)

@st.cache_resource(experimental_allow_widgets=True)

def get_model():
    bert_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
    bert_model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=6)
    return bert_tokenizer, bert_model

bert_tokenizer, bert_model = get_model()

user_input = st.text_area("Enter Text to Analyze")
button = st.button("Analyze")

LABEL_NAME = ['Anger','Disgust','Fear','Joy','Sadness','Surprise']

if user_input and button:
    predict_input = bert_tokenizer.encode(
        user_input,
        max_length=80,
        truncation=True,
        padding=True,
        return_tensors="tf")
    
    tf_output = bert_model.predict(predict_input)[0]
    tf_prediction = tf.nn.softmax(tf_output, axis=1)
    label = tf.argmax(tf_prediction, axis=1)
    label = label.numpy()
    
    st.write("Prediction: ", LABEL_NAME[label[0]]) 