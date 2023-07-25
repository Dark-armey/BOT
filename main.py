# app.py
from flask import Flask, render_template, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import openai
import time

app = Flask(__name__)

openai_api_key = ""

def load_db(file, chain_type, k):
    # Load documents
    loader = PyPDFLoader('Prospectus2020.pdf')
    documents = loader.load()
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # Define embedding
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # Define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # Create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            model_name="gpt-3.5-turbo-0301",
            temperature=0,
            openai_api_key=openai_api_key  # Provide the API key here
        ),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )

    return qa

# Load the Chatbot with University-related data and questions.
qa = load_db("Prospectus2020.pdf", "stuff", 4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/LandingPage')
def landing_page():
    return render_template('LandingPage.html')

@app.route('/query', methods=['POST'])
def handle_query():
    user_input = request.json['query']
    chat_history = []

    try:
        # Use the Quest Bot for answering queries related to Quaid-e-Awam University.
        result = qa({"question": user_input, "chat_history": chat_history})
        answer = result["answer"]
    except openai.RateLimitError as e:
        return jsonify({"answer": f"Rate limit reached. Waiting for {e.retry_after_seconds} seconds..."}), 429

    # Store the conversation history for future interactions.
    chat_history.append((user_input, answer))

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
