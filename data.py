# requirements: pip install langchain openai pinecone-client tiktoken
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pinecone

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "patient-record")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp")

# Init embeddings + pinecone
emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=emb.embed_query("test").shape[0])
index = pinecone.Index(INDEX_NAME)

def ingest_patient_note(patient_id: str, doc_id: str, text: str, metadata: dict):
    # split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        meta = dict(metadata)
        meta.update({
            "patient_id": patient_id,
            "doc_id": doc_id,
            "chunk_index": i
        })
        docs.append(Document(page_content=chunk, metadata=meta))
    # embed and upsert in batches
    for i in range(0, len(docs), 10):
        batch = docs[i:i+10]
        texts = [d.page_content for d in batch]
        embs = emb.embed_documents(texts)
        items = []
        for j, e in enumerate(embs):
            cid = f"{patient_id}::{doc_id}::chunk::{i+j}"
            items.append((cid, e, batch[j].metadata))
        index.upsert(vectors=items)
    print(f"Ingested {len(docs)} chunks for {patient_id}/{doc_id}")
