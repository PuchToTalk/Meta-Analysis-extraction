#### Meta-Analysis data extraction

Supervised by Dr. M. Nguyen, Vy Nguyen

### Steps
- **PDF Upload**: Users can upload PDF files, which are then converted into text for further processing.
- **Semantic Search**: The system creates embeddings for the text and uses them for efficient semantic search in a vector database.
- **Vector Store Generation**: Learn how to generate a vector store from text chunks using embeddings and FAISS for efficient retrieval.
- **Conversational Chain**: The system initializes a conversational chain for handling chat interactions.
- **Information Extraction**: The application is capable of extracting information from documents based on user queries.
- **Results**: Store the user questions, and the answers generated, into a CSV file


### Prerequisites
Before getting started, ensure that you have the following prerequisites installed:
- Python 3.x
- LangChain framework
- OpenAI GPT-3 or other large language models
- FAISS library for efficient similarity search

### Installation
1. Clone the repository to your local machine:
   ```
   git clone https://github.com/PuchToTalk/Meta-Analysis-extraction
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
