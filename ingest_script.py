import os, json
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "patient-record")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp")

# --- Initialize ---
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# if not PINECONE_API_KEY:
#     raise SystemExit("PINECONE_API_KEY not set. Set it in your environment or in a .env file (PINECONE_API_KEY=...).")


pc = Pinecone(
        api_key="pcsk_3ipCfP_CxxV6bzXtgeznmoJDJ8CxvntuX2MgR2DgzCDLyg5CZGeHbuTvLcziLwHNfcsjD9"
    )
if not pc.has_index(INDEX_NAME):
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )
dense_index = pc.Index(INDEX_NAME)

# splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# def ingest_patient_note(patient_id: str, doc_id: str, text: str, metadata: dict):
#     """Split, embed, and upload one patient's document into Pinecone."""
#     chunks = splitter.split_text(text)
#     for i in range(0, len(chunks), 10):  # batch of 10
#         batch = chunks[i:i+10]
#         embeds = embeddings.embed_documents(batch)
#         items = []
#         for j, emb in enumerate(embeds):
#             vector_id = f"{patient_id}::{doc_id}::{i+j}"
#             meta = dict(metadata)
#             meta.update({"patient_id": patient_id, "doc_id": doc_id, "chunk": i+j})
#             items.append((vector_id, emb, meta))
#         index.upsert(vectors=items)
#     print(f"Ingested {len(chunks)} chunks for {patient_id}/{doc_id}")




def ingest_json_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)

    dense_index.upsert_records("namespace1", records)

if __name__ == "__main__":
    ingest_json_file("patient_data.json")
