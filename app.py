from flask import Flask, render_template, request, jsonify
from config import Config
from langchain_groq import ChatGroq
from transcript_processor import TranscriptProcessor
from qa_generator import QAGenerator
from vector_store import VectorStoreManager
from chat_handler import ChatHandler
import os

app = Flask(__name__)
app.config.from_object(Config)

# Initialize components
llm = ChatGroq(
    model=app.config['LLM_MODEL'],
    temperature=app.config['LLM_TEMPERATURE'],
    verbose=True,
    timeout=None,
    api_key=app.config['GROQ_API_KEY']
)

vector_store_manager = VectorStoreManager(app.config['EMBEDDING_MODEL'])
transcript_processor = TranscriptProcessor()
qa_generator = QAGenerator(llm)

# Initialize chat handler after first transcript is processed
chat_handler = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_transcript', methods=['POST'])
def process_transcript():
    youtube_url = request.json.get('youtube_url')
    if not youtube_url:
        return jsonify({"error": "YouTube URL is required"}), 400
        
    df = transcript_processor.get_transcript(youtube_url)
    if df is None:
        return jsonify({"error": "Failed to process transcript"}), 500
        
    # Prepare documents and create vector store
    docs = transcript_processor.prepare_documents(df)
    vector_store = vector_store_manager.create_vector_store(docs)
    
    # Generate QA pairs
    inputs, outputs = qa_generator.generate_qa_pairs(df["text"].tolist())
    
    # Initialize chat handler
    global chat_handler
    chat_handler = ChatHandler(llm, vector_store)
    
    return jsonify({
        "message": "Transcript processed successfully",
        "qa_pairs": list(zip([i['question'] for i in inputs], [o['answer'] for o in outputs]))
    })

@app.route('/chat', methods=['POST'])
def chat():
    if chat_handler is None:
        return jsonify({"error": "Please process a transcript first"}), 400
        
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
        
    response = chat_handler.process_chat(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)