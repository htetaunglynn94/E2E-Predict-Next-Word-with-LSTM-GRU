# Essential libraries
import pandas as pd
import numpy as np
import pickle

# NLP libraries
from nltk.corpus import gutenberg  # text document from Gutenberg project

# TensorFlow libraries
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Import Stramlit
import streamlit as st

# Load LSTM model
lstm_model = load_model('model/LSTM_RNN_model.h5')

# Load GRU model
gru_model = load_model('model/GRU_RNN_model.h5')

# Load tokenizer
with open("preprocessor/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Prediction function
def predict_next_word(model, tokenizer, text):
    max_seq = model.input_shape[1]
    tkn_lst = tokenizer.texts_to_sequences([text])[0]
    pad_seq = pad_sequences(sequences=[tkn_lst], maxlen=max_seq)
    pred = model.predict(pad_seq, verbose=0)[0]
    pred_word_index = np.argmax(pred)

    return tokenizer.index_word.get(pred_word_index, None)

# Streamlit App
st.title("Next word prediction using LSTM RNN and early stopping")
text = st.text_input("Enter the sequence of words")
if st.button("Predict next word"):
    next_word_lstm = predict_next_word(lstm_model, tokenizer, text)
    next_word_gru = predict_next_word(gru_model, tokenizer, text)
    st.write(f"Next word using LSTM: {next_word_lstm}")
    st.write(f"Next word using GRU: {next_word_gru}")
    
