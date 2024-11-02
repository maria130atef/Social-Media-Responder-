from flask import Flask, render_template, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
import os
from datetime import datetime, timedelta
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langdetect import detect


app = Flask(__name__)

# Cache variables
cache = {
    "content": None,
    "timestamp": None
}

CACHE_EXPIRY = timedelta(minutes=30)  # Cache expires after 30 minutes

def setup_selenium_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def extract_content_from_url(url):
    driver = setup_selenium_driver()
    driver.get(url)
    time.sleep(5)

    page_source = driver.page_source
    driver.quit()

    soup = BeautifulSoup(page_source, 'html.parser')
    title = soup.title.string if soup.title else 'No Title'

    content = []
    for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        section_title = header.get_text().strip()
        section_content = []
        for sibling in header.find_next_siblings():
            if sibling.name and sibling.name.startswith('h'):
                break
            section_content.append(sibling.get_text().strip())
        content.append(f"{section_title}\n{''.join(section_content)}")

    full_content = "\n\n".join(content)
    return {"title": title, "content": full_content}

def get_cached_content(url):
    # Check if cache is expired
    if cache["timestamp"] and datetime.now() - cache["timestamp"] < CACHE_EXPIRY:
        return cache["content"]
    
    # Otherwise, fetch and cache the content
    content = extract_content_from_url(url)
    cache["content"] = content
    cache["timestamp"] = datetime.now()
    return content

def is_question(sentence):
    question_words = ["متى","اين","؟","?","امتي","ازاي","فين","ما","ماذا", "كيف", "لماذا", "أين", " من هو", "هل"]

    for word in question_words:
        if word in sentence:
            return True
    else:    
      return False  

url = "https://www.nbe.com.eg/NBE/E/#/AR/Home"
document = get_cached_content(url)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(document['content'])

documents = [Document(page_content=chunk) for chunk in chunks]

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

vectordb = FAISS.from_documents(documents, embeddings)
retriever = vectordb.as_retriever()

groq_api_key = "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"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
prompt = ChatPromptTemplate.from_template("""
Answer the question in Arabic using only the provided context.
Provide a clear and concise answer, and ensure it addresses the question accurately.
<context>
{context}
</context>
Question: {input}
""")

def get_response(question):
    

   language = detect(question)
   qu=is_question(question)
   if language == 'ar' and qu==True:
        document_chain=create_stuff_documents_chain(llm,prompt)
        retrieval_chain=create_retrieval_chain(retriever,document_chain)    
        response=retrieval_chain.invoke({"input":question})
        return response
   else: 
       return None

    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Safeguard against empty input
    data = request.get_json()
    question = data.get('question', None)

    if not question:
        return jsonify({"error": "No question provided."}), 400

  
    # document_chain = create_stuff_documents_chain(llm, prompt)
    # retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    try:
        result = get_response(question)
        
        # Ensure result and answer are valid
        if result is None or result.get('answer') is None:
            return '', 204 

        # Extracting the answer from the result
        answer = result.get('answer', 'No answer found.')
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    return jsonify({"response": answer})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
