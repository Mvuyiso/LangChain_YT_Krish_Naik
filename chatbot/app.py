from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


os.environ["OPEN_AI_API_KEY"]=os.getenv("OPENAI_API_KEY")

#LAngsmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]='True'

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an assistant that helps as a chatbot"),
        ("user", "Question:{question}")
    ]
)


# Streamlit app

st.title('Langchain demo with OpenAI')
input_text=st.text_input("Enter your question")

#openAI LLM
llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))

