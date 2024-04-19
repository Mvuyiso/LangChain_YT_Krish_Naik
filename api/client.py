import requests
import streamlit as st

def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
                             json={'input':{"topic": input_text}})

    return response.json()['output']['content']

def get_llama_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke",
                             json={'input':{"topic": input_text}})

    return response.json()['output']

#streamlit

st.title("More Langchain demo")
input_text=st.text_input("Enter your essay topic")
input_text1=st.text_input("Enter your poem topic")

if input_text:
    st.write(get_openai_response(input_text))


if input_text1:
    st.write(get_llama_response(input_text1))