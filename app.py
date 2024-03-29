# Import necessary libraries
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader


from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import faiss
import constants
import csv
import pandas as pd


os.environ['OPENAI_API_KEY'] = constants.APIKEY

#Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global vectorstore, conversation_chain, chat_history

    if request.method == 'POST':
        user_question = request.form['user_question']
        response = conversation_chain({'question': user_question})
        response_text = response.get('answer', 'No answer found') if isinstance(response, dict) else 'Response format not recognized'
        
        chat_history.append({'type': 'question', 'content': user_question})
        chat_history.append({'type': 'answer', 'content': response_text})
        
        # Define CSV file path
        csv_file_path = 'chat_history.csv'
        
        # Read existing data or initialize an empty DataFrame
        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            df = pd.DataFrame()
        
        # Find the next available column (for a new question-answer pair)
        next_col = len(df.columns) // 2  # Assuming each Q&A occupies 2 columns
        # Append the new question and answer as new columns
        df[f'Question {next_col+1}'] = [user_question] + [None] * (len(df.index) - 1)
        df[f'Answer {next_col+1}'] = [response_text] + [None] * (len(df.index) - 1)
        
        # Write the updated DataFrame back to CSV
        df.to_csv(csv_file_path, index=False)

    return render_template('chat.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)