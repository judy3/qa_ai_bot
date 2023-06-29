import requests
import json
from typing import Optional
import streamlit as st

from constants import BOT_API_URL, BOT_API_URL_WITHOUT_KB
#from bot_utils import ask_bot, load_embedding_model, load_rwkv_model


# load the models into cache
#@st.cache_resource
#def load_models():
#    embeddings = load_embedding_model()
#    rwkv_model = load_rwkv_model()
#    return embeddings, rwkv_model

def call_bot_api(
    question: str,
    api_url: Optional[str] = BOT_API_URL
    ):
    data = {
        "question": question
    }
    response = requests.post(api_url, data=json.dumps(data))
    if response.status_code == 200:
        result = response.text
        return result
    else:
        result = f"Request failed with status code: {response.status_code}"
        return result
    

def simple_ui():
    # write a sample webpage to ask questions
    st.title("ITT Q&A Bot:")
    query = st.text_input("Enter your question below:")

    button_container = st.container()

    with button_container:
        col1, col2 = st.columns(2)  # Create two columns

        if col1.button("Ask with KB"):
            response = call_bot_api(question=query)
            st.write(response)

        if col2.button("Ask directly"):
            response = call_bot_api(question=query, api_url=BOT_API_URL_WITHOUT_KB)
            st.write(response)



#embeddings, rwkv_model = load_models()
simple_ui()


