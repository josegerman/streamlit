# ==================================
# About:
# ==================================
# RAG application using Langchain, GPT4(OpenAI), streamlit, python

# ==================================
# Imports
# ==================================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os

#from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
#from langchain_community.vectorstores import FAISS
#from langchain_community.vectorstores import Chroma # <-- deprecated
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==================================
# Load the variables from .env
# ==================================
#load_dotenv()
OPENAI_API_KEY = st.secrets('OPENAI_ACCESS_KEY') # <- changed

st.write("Hello World!")

template = """You are a veterinarian for question-answering tasks. Answer the question based on the following context. If you don't know the answer, just say that you don't know. Use four sentences maximum and keep the answer concise:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY) # <- changed
embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY) # <- changed

# Define the persistent directory
#current_dir = os.path.dirname(os.path.abspath(__file__))
#persistent_directory = os.path.join(current_dir, "db", "chroma_petmed_db")

#db = FAISS.from_texts(splitted_data,embedding=embedding)
db = Chroma(persist_directory="./db", embedding_function=embedding)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3},)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
) 

question = st.text_input("How can we help your pet today?")

result = chain.invoke(question)

st.write(result)



# ==================================
# Run streamlit app
# ==================================
# streamlit run C:\_DEV\VSCode\Workspaces\streamlit\streamlit\streamlit_rag_app.py


