from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid, time
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import HumanMessage
import os
app = FastAPI()
llm = ChatOpenAI(model_name= os.getenv("MODEL_NAME","gpt-4o-mini"), temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "patient-record")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp")
pc = Pinecone(
        api_key="pcsk_3ipCfP_CxxV6bzXtgeznmoJDJ8CxvntuX2MgR2DgzCDLyg5CZGeHbuTvLcziLwHNfcsjD9"
    )
dense_index = pc.Index(INDEX_NAME)
# Setup retriever (from pinecone)
# vectorstore = Pinecone(dense_index, embedding=emb)  # pseudo; adapt to your versions
# retriever = vectorstore.as_retriever(search_kwargs={"k": 6})  # top 6 chunks




class AskReq(BaseModel):
    patient_id: str
    question: str
    user_id: str  # doctor id for audit

@app.post("/ask")
def ask(req: AskReq):
    trace_id = str(uuid.uuid4())
    start = time.time()
    # authorization, check doctor has access to patient_id
    # TODO: implement RBAC check
    try:
        answer = answer_patient_question(req.patient_id, req.question)
        # log minimal audit record
        audit = {
            "trace_id": trace_id,
            "user_id": req.user_id,
            "patient_id": req.patient_id,
            "question": req.question,
            "timestamp": start,
            "status": "ok"
        }
        # write audit to secure audit log (not storing full PHI)
        # audit_logger.info(audit)
        return {"trace_id": trace_id, "result": answer}
    except Exception as e:
        # log error with trace_id
        raise HTTPException(status_code=500, detail=str(e))


prompt = PromptTemplate(
    input_variables=["patient_id", "question", "context_chunks"],
    template=open("clinical_prompt.txt").read()
)

# Build QA chain that uses our prompt


def answer_patient_question(patient_id: str, question: str):
    # retrieve context for this patient
    # docs = retriever.get_relevant_documents(patient_id)  # your retriever should filter by metadata patient_id
    # context_concat = "\n\n---\n\n".join([f"doc_id:{d.metadata['doc_id']} date:{d.metadata.get('date','')} text:{d.page_content}" for d in docs])
    # result = chain.run({"patient_id": patient_id, "question": question, "context_chunks": context_concat})
    
    res = dense_index.search(
        namespace="namespace1",
        query={
            "inputs": {"text": question},
            "top_k": 2,
            "filter": {"patient_id": patient_id},
        },
        fields=["patient_id", "chunk_text"]
    )

    # convert results into Documents (LangChain style)
    docs =[]

    for match in res.get("matches", []) or res.get("results", []):  # adapt to your API response shape
        # adapt keys per your index response
        text = match.get("metadata", {}).get("chunk_text") or match.get("chunk_text") or match.get("text")
        meta = match.get("metadata", {}) or {}
        meta.update({
            "score": match.get("score") or match.get("distance"),
            "index_id": match.get("id")
        })
        if text:
            docs.append(Document(page_content=text, metadata=meta))

    if not docs:
        return {"error": "no_relevant_docs", "message": "No documents found for this patient."}

    
    context_concat = "\n\n---\n\n".join([f"doc_id:{d.metadata['doc_id']} date:{d.metadata.get('date','')} text:{d.page_content}" for d in docs])
    prompt_text = prompt.format(patient_id=patient_id, question=question, context_chunks=context_concat)
    resp = llm(messages=[HumanMessage(content=prompt_text)])
    print("LLM response:", resp)
    # Validate JSON parse
    import json
    try:
        parsed = resp.generations[0][0].text
    except Exception:
        # fallback: ask LLM to strictly return JSON (or run a post-processor)
        parsed = {"error": "invalid_json", "raw": resp}
    return parsed