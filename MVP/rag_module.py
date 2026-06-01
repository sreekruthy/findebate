import os
import json
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import chromadb

# Download tokenizer once
nltk.download('punkt')
nltk.download('punkt_tab')

# -------- TEXT CHUNKING --------
def chunk_text(text, max_words=400):
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()

        if current_length + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# -------- GLOBAL STORAGE --------
all_chunks = []

# -------- PROCESS DATA --------
def process_folder(folder, company):
    for filename in os.listdir(folder):

        if not filename.startswith("clean_"):
            continue

        file_path = os.path.join(folder, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)

        if "earnings" in filename:
            data_type = "earnings_q1"
        elif "news" in filename:
            data_type = "news_q1"
        else:
            data_type = "other"

        for chunk in chunks:
            all_chunks.append({
                "company": company,
                "type": data_type,
                "text": chunk
            })


# -------- INITIALIZE VECTOR DB --------
def initialize_rag():
    global collection, model
    
    #all_chunks = [] # we did this beacuse, when other initialise it might crash
    """
    If initialize_rag() is called a second time (which will happen when the 
    full pipeline runs multiple agents), it appends chunks again to the same 
    list, so now you have duplicate chunks. Then when it tries to add them to 
    ChromaDB with the same IDs ("0", "1", "2"...), ChromaDB crashes because 
    IDs must be unique.
    """
    
    process_folder("data/apple", "Apple")
    process_folder("data/tesla", "Tesla")

    print("Total chunks:", len(all_chunks))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client(
        settings=chromadb.Settings(
            persist_directory="./chroma_db"
        )
    )

    collection = client.get_or_create_collection(name="finance_data")
    """
    try:
        client.delete_collection("finance_data")
    except:
        pass
        collection = client.create_collection(name="finance_data")
    """
    for i, chunk in enumerate(all_chunks):
        embedding = model.encode(chunk["text"]).tolist()

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "company": chunk["company"],
                "type": chunk["type"]
            }]
        )

    print("RAG initialized successfully")


# -------- RETRIEVAL --------
def retrieve_filtered(query, company, data_type=None, k=3):
    query_embedding = model.encode(query).tolist()

    if data_type:
        where_filter = {
            "$and": [
                {"company": company},
                {"type": data_type}
            ]
        }
    else:
        where_filter = {"company": company}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_filter
    )

    return results["documents"][0]  # RETURN CLEAN TEXT