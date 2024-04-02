# Import necessary libraries
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text

from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import faiss
import constants
import csv
import pandas as pd
import numpy as np
import faiss
import openai


# Setup environment variables for API access, etc.
os.environ['OPENAI_API_KEY'] = constants.APIKEY



class ConversationalChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        # Initialize any additional components needed

    def __call__(self, input_dict):
        question = input_dict['question']
        # Logic to use vectorstore to find relevant information and generate a response
        # This is a placeholder. Replace with your actual logic.
        response = "This is a placeholder response to: " + question
        return {'answer': response}



def get_pdf_text(pdf_path):
    return extract_text(pdf_path)

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    end = chunk_size
    while start < len(text):
        if start > 0:
            start -= chunk_overlap
        chunks.append(text[start:end])
        start += chunk_size
        end += chunk_size
    return chunks




client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
   openai.api_key = os.getenv("OPENAI_API_KEY")
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding




def get_vectorstore(text_chunks):
    # Generate embeddings for each text chunk
    embeddings_list = [get_embedding(chunk) for chunk in text_chunks]
    # Convert list of embeddings to a NumPy array
    embeddings = np.vstack(embeddings_list)

    # Ensure embeddings are in float32 for FAISS
    embeddings = embeddings.astype(np.float32)

    # Create a FAISS index and add embeddings
    D = embeddings.shape[1]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(D)
    index.add(embeddings)

    return index



def get_conversation_chain(vectorstore):
    return ConversationalChain(vectorstore)


pdf_path = '/Users/paulc/Documents/Paul/Stanford/RA/Work/Keast.pdf'

raw_text = get_pdf_text(pdf_path)
text_chunks = get_text_chunks(raw_text)
vectorstore = get_vectorstore(text_chunks)
# Initialize conversation chain with the vectorstore
conversation_chain = get_conversation_chain(vectorstore)




def main():
    # Load and process PDF documents to initialize conversation_chain (if needed)
    # This part is skipped for brevity

    # Path to the questions text file
    questions_file_path = '/Users/paulc/pdfGPT/texts/questions.txt'
    # Path to the output CSV file
    csv_file_path = '/Users/paulc/pdfGPT/results/qa_history.csv'
    
    # Initialize an empty list to store the chat history
    chat_history = []

    # Read questions from the text file
    with open(questions_file_path, 'r') as file:
        questions = file.readlines()

    # Assuming conversation_chain is initialized and ready to use
    # Now, in your main loop, you can use conversation_chain as intended
    for question in questions:
        question = question.strip()
        if question:
            # Use the callable conversation_chain with the correct input
            response = conversation_chain({'question': question})
            response_text = response.get('answer', 'No answer found')
            
            chat_history.append({'question': question, 'answer': response_text})


    # Convert chat history to DataFrame
    df = pd.DataFrame(chat_history)

    # Write the DataFrame to CSV
    df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    main()
